"""SimpleGCN2

Test GCN model very basic
and will be deleted was here for
purpose of code experimenting
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# TODO: layer sizes
# make sure sizes multiples of 8 to map onto tensor cores

class SimpleGCN2(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, x, edge_index):
        #x, edge_index = data.x, data.edge_index

        print('data type')
        print(x.dtype)
        print(edge_index)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
