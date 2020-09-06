#  MIT License
#
#  Copyright (c) 2019 Geom-GCN Authors
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F


# Adapted from https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
class GCNSingleHead(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout_prob):
        super(GCNSingleHead, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(in_feats, out_feats, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)
        self.activation = activation

    def message_func(self, edges):
        return {'m': edges.src['h']}

    def reduce_func(self, nodes):
        return {'h': th.sum(nodes.mailbox['m'], dim=1)}

    def forward(self, g, feature):
        h = self.dropout(feature)
        h = self.linear(h)
        h = h * g.ndata['norm']
        g.ndata['h'] = h
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata.pop('h')
        h = h * g.ndata['norm']
        h = self.activation(h)
        return h


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation, num_heads, dropout_prob, merge):
        super(GCN, self).__init__()
        self.attention_heads = nn.ModuleList()
        for _ in range(num_heads):
            self.attention_heads.append(GCNSingleHead(in_feats, out_feats, activation, dropout_prob))
        self.merge = merge

    def forward(self, g, feature):
        all_attention_head_outputs = [head(g, feature) for head in self.attention_heads]
        if self.merge == 'cat':
            return th.cat(all_attention_head_outputs, dim=1)
        else:
            return th.mean(th.stack(all_attention_head_outputs), dim=0)


# Adapted from https://docs.dgl.ai/tutorials/models/1_gnn/9_gat.html
class GATSingleAttentionHead(nn.Module):
    def __init__(self, in_feats, out_feats, activation, dropout_prob):
        super(GATSingleAttentionHead, self).__init__()
        self.in_feats_dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(in_feats, out_feats, bias=False)
        nn.init.xavier_uniform_(self.linear.weight)
        self.attention_linear = nn.Linear(2 * out_feats, 1, bias=False)
        nn.init.xavier_uniform_(self.attention_linear.weight)
        self.attention_head_dropout = nn.Dropout(dropout_prob)
        self.linear_feats_dropout = nn.Dropout(dropout_prob)
        self.bias = nn.Parameter(th.ones(1, out_feats, dtype=th.float32, requires_grad=True))
        nn.init.xavier_uniform_(self.bias.data)
        self.activation = activation

    def calculate_node_pairwise_attention(self, edges):
        h_concat = th.cat([edges.src['Wh'], edges.dst['Wh']], dim=1)
        e = self.attention_linear(h_concat)
        e = F.leaky_relu(e, negative_slope=0.2)
        return {'e': e}

    def message_func(self, edges):
        return {'Wh': edges.src['Wh'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        a = F.softmax(nodes.mailbox['e'], dim=1)
        a_dropout = self.attention_head_dropout(a)
        Wh_dropout = self.linear_feats_dropout(nodes.mailbox['Wh'])
        return {'h_new': th.sum(a_dropout * Wh_dropout, dim=1)}

    def forward(self, g, feature):
        Wh = self.in_feats_dropout(feature)
        Wh = self.linear(Wh)
        g.ndata['Wh'] = Wh
        g.apply_edges(self.calculate_node_pairwise_attention)
        g.update_all(self.message_func, self.reduce_func)
        h_new = g.ndata.pop('h_new')
        h_new = self.activation(h_new + self.bias)
        return h_new


# Adapted from https://docs.dgl.ai/tutorials/models/1_gnn/9_gat.html
class GAT(nn.Module):
    def __init__(self, in_feats, out_feats, activation, num_heads, dropout_prob, merge):
        super(GAT, self).__init__()
        self.attention_heads = nn.ModuleList()
        for _ in range(num_heads):
            self.attention_heads.append(GATSingleAttentionHead(in_feats, out_feats, activation, dropout_prob))
        self.merge = merge

    def forward(self, g, feature):
        all_attention_head_outputs = [head(g, feature) for head in self.attention_heads]
        if self.merge == 'cat':
            return th.cat(all_attention_head_outputs, dim=1)
        else:
            return th.mean(th.stack(all_attention_head_outputs), dim=0)


class GeomGCNSingleChannel(nn.Module):
    def __init__(self, g, in_feats, out_feats, num_divisions, activation, dropout_prob, merge):
        super(GeomGCNSingleChannel, self).__init__()
        self.num_divisions = num_divisions
        self.in_feats_dropout = nn.Dropout(dropout_prob)

        self.linear_for_each_division = nn.ModuleList()
        for i in range(self.num_divisions):
            self.linear_for_each_division.append(nn.Linear(in_feats, out_feats, bias=False))

        for i in range(self.num_divisions):
            nn.init.xavier_uniform_(self.linear_for_each_division[i].weight)

        self.activation = activation
        self.g = g
        self.subgraph_edge_list_of_list = self.get_subgraphs(self.g)
        self.merge = merge
        self.out_feats = out_feats

    def get_subgraphs(self, g):
        subgraph_edge_list = [[] for _ in range(self.num_divisions)]
        u, v, eid = g.all_edges(form='all')
        for i in range(g.number_of_edges()):
            subgraph_edge_list[g.edges[u[i], v[i]].data['subgraph_idx']].append(eid[i])

        return subgraph_edge_list

    def forward(self, feature):
        in_feats_dropout = self.in_feats_dropout(feature)
        self.g.ndata['h'] = in_feats_dropout

        for i in range(self.num_divisions):
            subgraph = self.g.edge_subgraph(self.subgraph_edge_list_of_list[i])
            subgraph.copy_from_parent()
            subgraph.ndata[f'Wh_{i}'] = self.linear_for_each_division[i](subgraph.ndata['h']) * subgraph.ndata['norm']
            subgraph.update_all(message_func=fn.copy_u(u=f'Wh_{i}', out=f'm_{i}'),
                                reduce_func=fn.sum(msg=f'm_{i}', out=f'h_{i}'))
            subgraph.ndata.pop(f'Wh_{i}')
            subgraph.copy_to_parent()

        self.g.ndata.pop('h')
        
        results_from_subgraph_list = []
        for i in range(self.num_divisions):
            if f'h_{i}' in self.g.node_attr_schemes():
                results_from_subgraph_list.append(self.g.ndata.pop(f'h_{i}'))
            else:
                results_from_subgraph_list.append(
                    th.zeros((feature.size(0), self.out_feats), dtype=th.float32, device=feature.device))

        if self.merge == 'cat':
            h_new = th.cat(results_from_subgraph_list, dim=-1)
        else:
            h_new = th.mean(th.stack(results_from_subgraph_list, dim=-1), dim=-1)
        h_new = h_new * self.g.ndata['norm']
        h_new = self.activation(h_new)
        return h_new


class GeomGCN(nn.Module):
    def __init__(self, g, in_feats, out_feats, num_divisions, activation, num_heads, dropout_prob, ggcn_merge,
                 channel_merge):
        super(GeomGCN, self).__init__()
        self.attention_heads = nn.ModuleList()
        for _ in range(num_heads):
            self.attention_heads.append(
                GeomGCNSingleChannel(g, in_feats, out_feats, num_divisions, activation, dropout_prob, ggcn_merge))
        self.channel_merge = channel_merge
        self.g = g

    def forward(self, feature):
        all_attention_head_outputs = [head(feature) for head in self.attention_heads]
        if self.channel_merge == 'cat':
            return th.cat(all_attention_head_outputs, dim=1)
        else:
            return th.mean(th.stack(all_attention_head_outputs), dim=0)


class GCNNet(nn.Module):
    def __init__(self, num_input_features, num_output_classes, num_hidden, num_heads_layer_one, num_heads_layer_two,
                 dropout_rate):
        super(GCNNet, self).__init__()
        self.gcn1 = GCN(num_input_features, num_hidden, F.relu, num_heads_layer_one, dropout_rate, 'cat')
        self.gcn2 = GCN(num_hidden * num_heads_layer_one, num_output_classes, lambda x: x, num_heads_layer_two,
                        dropout_rate, 'mean')

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x


class GATNet(nn.Module):
    def __init__(self, num_input_features, num_output_classes, num_hidden, num_heads_layer_one, num_heads_layer_two, dropout_rate):
        super(GATNet, self).__init__()
        self.gat1 = GAT(num_input_features, num_hidden, F.elu, num_heads_layer_one, dropout_rate, 'cat')
        self.gat2 = GAT(num_hidden * num_heads_layer_one, num_output_classes, lambda x: x, num_heads_layer_two,
                        dropout_rate, 'mean')

    def forward(self, g, features):
        x = self.gat1(g, features)
        x = self.gat2(g, x)
        return x


class GeomGCNNet(nn.Module):
    def __init__(self, g, num_input_features, num_output_classes, num_hidden, num_divisions, num_heads_layer_one,
                 num_heads_layer_two,
                 dropout_rate, layer_one_ggcn_merge, layer_one_channel_merge, layer_two_ggcn_merge,
                 layer_two_channel_merge):
        super(GeomGCNNet, self).__init__()
        self.geomgcn1 = GeomGCN(g, num_input_features, num_hidden, num_divisions, F.relu, num_heads_layer_one,
                                dropout_rate,
                                layer_one_ggcn_merge, layer_one_channel_merge)

        if layer_one_ggcn_merge == 'cat':
            layer_one_ggcn_merge_multiplier = num_divisions
        else:
            layer_one_ggcn_merge_multiplier = 1

        if layer_one_channel_merge == 'cat':
            layer_one_channel_merge_multiplier = num_heads_layer_one
        else:
            layer_one_channel_merge_multiplier = 1

        self.geomgcn2 = GeomGCN(g, num_hidden * layer_one_ggcn_merge_multiplier * layer_one_channel_merge_multiplier,
                                num_output_classes, num_divisions, lambda x: x,
                                num_heads_layer_two, dropout_rate, layer_two_ggcn_merge, layer_two_channel_merge)
        self.g = g

    def forward(self, features):
        x = self.geomgcn1(features)
        x = self.geomgcn2(x)
        return x
