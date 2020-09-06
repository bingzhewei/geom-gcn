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

import argparse
import json
import os
import time

import dgl.init
import numpy as np
import tensorboardX
import torch as th
import torch.nn.functional as F

import utils_data
from utils_layers import GeomGCNNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dataset_embedding', type=str)
    parser.add_argument('--num_hidden', type=int)
    parser.add_argument('--num_heads_layer_one', type=int)
    parser.add_argument('--num_heads_layer_two', type=int)
    parser.add_argument('--layer_one_ggcn_merge', type=str, default='cat')
    parser.add_argument('--layer_two_ggcn_merge', type=str, default='mean')
    parser.add_argument('--layer_one_channel_merge', type=str, default='cat')
    parser.add_argument('--layer_two_channel_merge', type=str, default='mean')
    parser.add_argument('--dropout_rate', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay_layer_one', type=float)
    parser.add_argument('--weight_decay_layer_two', type=float)
    parser.add_argument('--num_epochs_patience', type=int, default=100)
    parser.add_argument('--num_epochs_max', type=int, default=5000)
    parser.add_argument('--run_id', type=str)
    parser.add_argument('--dataset_split', type=str)
    parser.add_argument('--learning_rate_decay_patience', type=int, default=50)
    parser.add_argument('--learning_rate_decay_factor', type=float, default=0.8)
    args = parser.parse_args()
    vars(args)['model'] = 'GeomGCN_TwoLayers'

    t1 = time.time()
    if args.dataset_split == 'jknet':
        g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels = utils_data.load_data(
            args.dataset, None, 0.6, 0.2, 'GeomGCN', args.dataset_embedding)
    else:
        g, features, labels, train_mask, val_mask, test_mask, num_features, num_labels = utils_data.load_data(
            args.dataset, args.dataset_split, None, None, 'GeomGCN', args.dataset_embedding)
    print(time.time() - t1)

    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)

    net = GeomGCNNet(g=g, num_input_features=num_features, num_output_classes=num_labels, num_hidden=args.num_hidden,
                     num_divisions=9, dropout_rate=args.dropout_rate,
                     num_heads_layer_one=args.num_heads_layer_one, num_heads_layer_two=args.num_heads_layer_two,
                     layer_one_ggcn_merge=args.layer_one_ggcn_merge,
                     layer_one_channel_merge=args.layer_one_channel_merge,
                     layer_two_ggcn_merge=args.layer_two_ggcn_merge,
                     layer_two_channel_merge=args.layer_two_channel_merge)

    optimizer = th.optim.Adam([{'params': net.geomgcn1.parameters(), 'weight_decay': args.weight_decay_layer_one},
                               {'params': net.geomgcn2.parameters(), 'weight_decay': args.weight_decay_layer_two}],
                              lr=args.learning_rate)
    learning_rate_scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                      factor=args.learning_rate_decay_factor,
                                                                      patience=args.learning_rate_decay_patience)
    writer = tensorboardX.SummaryWriter(logdir=f'runs/{args.model}_{args.run_id}')

    net.cuda()
    features = features.cuda()
    labels = labels.cuda()
    train_mask = train_mask.cuda()
    val_mask = val_mask.cuda()
    test_mask = test_mask.cuda()

    # Adapted from https://github.com/PetarV-/GAT/blob/master/execute_cora.py
    patience = args.num_epochs_patience
    vlss_mn = np.inf
    vacc_mx = 0.0
    vacc_early_model = None
    vlss_early_model = None
    state_dict_early_model = None
    curr_step = 0

    # Adapted from https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
    dur = []
    for epoch in range(args.num_epochs_max):
        t0 = time.time()

        net.train()
        train_logits = net(features)
        train_logp = F.log_softmax(train_logits, 1)
        train_loss = F.nll_loss(train_logp[train_mask], labels[train_mask])
        train_pred = train_logp.argmax(dim=1)
        train_acc = th.eq(train_pred[train_mask], labels[train_mask]).float().mean().item()

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        net.eval()
        with th.no_grad():
            val_logits = net(features)
            val_logp = F.log_softmax(val_logits, 1)
            val_loss = F.nll_loss(val_logp[val_mask], labels[val_mask]).item()
            val_pred = val_logp.argmax(dim=1)
            val_acc = th.eq(val_pred[val_mask], labels[val_mask]).float().mean().item()

        learning_rate_scheduler.step(val_loss)

        dur.append(time.time() - t0)

        print(
            "Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} | Time(s) {:.4f}".format(
                epoch, train_loss.item(), train_acc, val_loss, val_acc, sum(dur) / len(dur)))

        writer.add_scalar('Train Loss', train_loss.item(), epoch)
        writer.add_scalar('Val Loss', val_loss, epoch)
        writer.add_scalar('Train Acc', train_acc, epoch)
        writer.add_scalar('Val Acc', val_acc, epoch)

        # Adapted from https://github.com/PetarV-/GAT/blob/master/execute_cora.py
        if val_acc >= vacc_mx or val_loss <= vlss_mn:
            if val_acc >= vacc_mx and val_loss <= vlss_mn:
                vacc_early_model = val_acc
                vlss_early_model = val_loss
                state_dict_early_model = net.state_dict()
            vacc_mx = np.max((val_acc, vacc_mx))
            vlss_mn = np.min((val_loss, vlss_mn))
            curr_step = 0
        else:
            curr_step += 1
            if curr_step >= patience:
                break

    net.load_state_dict(state_dict_early_model)
    net.eval()
    with th.no_grad():
        test_logits = net(features)
        test_logp = F.log_softmax(test_logits, 1)
        test_loss = F.nll_loss(test_logp[test_mask], labels[test_mask]).item()
        test_pred = test_logp.argmax(dim=1)
        test_acc = th.eq(test_pred[test_mask], labels[test_mask]).float().mean().item()
        test_hidden_features = net.geomgcn1(features).cpu().numpy()

        final_train_pred = test_pred[train_mask].cpu().numpy()
        final_val_pred = test_pred[val_mask].cpu().numpy()
        final_test_pred = test_pred[test_mask].cpu().numpy()

    results_dict = vars(args)
    results_dict['test_loss'] = test_loss
    results_dict['test_acc'] = test_acc
    results_dict['actual_epochs'] = 1 + epoch
    results_dict['val_acc_max'] = vacc_mx
    results_dict['val_loss_min'] = vlss_mn
    results_dict['total_time'] = sum(dur)
    with open(os.path.join('runs', f'{args.model}_{args.run_id}_results.txt'), 'w') as outfile:
        outfile.write(json.dumps(results_dict) + '\n')
    np.savez_compressed(os.path.join('runs', f'{args.model}_{args.run_id}_hidden_features.npz'),
                        hidden_features=test_hidden_features)
    np.savez_compressed(os.path.join('runs', f'{args.model}_{args.run_id}_final_train_predictions.npz'),
                        final_train_predictions=final_train_pred)
    np.savez_compressed(os.path.join('runs', f'{args.model}_{args.run_id}_final_val_predictions.npz'),
                        final_val_predictions=final_val_pred)
    np.savez_compressed(os.path.join('runs', f'{args.model}_{args.run_id}_final_test_predictions.npz'),
                        final_test_predictions=final_test_pred)
