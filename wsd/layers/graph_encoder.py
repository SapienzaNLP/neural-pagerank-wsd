import json

import torch
import torch.nn as nn
from torch_sparse import spmm


class GraphEncoder(nn.Module):
    def __init__(self, hparams, graph_path='data/wn_graph.json'):
        super(GraphEncoder, self).__init__()
        num_synsets, synset_indices, synset_values = GraphEncoder._load_graph(graph_path)
        self.num_synsets = num_synsets
        self.synset_indices = torch.nn.Parameter(data=synset_indices, requires_grad=False)
        self.synset_values = torch.nn.Parameter(data=synset_values, requires_grad=hparams.use_trainable_graph)

    def forward(self, x):
        return spmm(self.synset_indices, self.synset_values, self.num_synsets, self.num_synsets, x.t()).t()

    @staticmethod
    def _load_graph(graph_path):
        with open(graph_path) as f:
            graph = json.load(f)

        num_synsets = graph['n']
        synset_indices = torch.as_tensor(graph['indices'])
        synset_values = torch.as_tensor(graph['values'])
        return num_synsets, synset_indices, synset_values
