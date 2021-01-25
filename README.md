# Geom-GCN: Geometric Graph Convolutional Networks

GraphDML-UIUC-JLU: Graph-structured Data Mining and Machine Learning at University of Illinois at Urbana-Champaign (UIUC) and Jilin University (JLU)

### Abstract

Message-passing neural networks (MPNNs) have been successfully applied in a wide variety of applications in the real world. However, two fundamental weaknesses of MPNNs' aggregators limit their ability to represent graph-structured data: losing the structural information of nodes in neighborhoods and lacking the ability to capture long-range dependencies in disassortative graphs. Few studies have noticed the weaknesses from different perspectives. From the observations on classical neural network and network geometry, we propose a novel geometric aggregation scheme for graph neural networks to overcome the two weaknesses.  The behind basic idea is the aggregation on a graph can benefit from a continuous space underlying the graph. The proposed aggregation scheme is permutation-invariant and consists of three modules, node embedding, structural neighborhood, and bi-level aggregation. We also present an implementation of the scheme in graph convolutional networks, termed Geom-GCN, to perform transductive learning on graphs. Experimental results show the proposed Geom-GCN achieved state-of-the-art performance on a wide range of open datasets of graphs.

![](https://github.com/graphdml-uiuc-jlu/geom-gcn/blob/master/preview.PNG)

## Code

#### Required Packages
1. PyTorch
2. NetworkX
3. Deep Graph Library
4. Numpy
5. Scipy
6. Scikit-Learn
7. Tensorflow
8. TensorboardX

#### Table 3
To replicate the Geom-GCN results from Table 3, run
```bash
bash NewTableThreeGeomGCN_runs.txt
```
To replicate the GCN results from Table 3, run
```bash
bash NewTableThreeGCN_runs.txt
```
To replicate the GAT results from Table 3, run
```bash
bash NewTableThreeGAT_runs.txt
```

Results will be stored in `runs`.
#### Combination of Embedding Methods
To replicate the results for utilizing all embedding methods simultaneously, run
```bash
bash ExperimentTwoAllGeomGCN_runs.txt
```

Results will be stored in `runs`.
