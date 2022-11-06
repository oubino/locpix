"""SimplePointNet

Test PointNet model very basic
and will be deleted was here for
purpose of code experimenting
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius

# TODO: layer sizes
# make sure sizes multiples of 8 to map onto tensor cores

# note the following is taken directly from example on pytorch geometric
# github


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class PointNet(torch.nn.Module):
    def __init__(self, name):
        self.name = name
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 8, 8, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([16 + 3, 16, 16, 32]))
        self.sa3_module = GlobalSAModule(MLP([32 + 3, 32, 32, 32]))

        self.mlp = MLP([32, 32, 16, 8], dropout=0.5, norm=None)

    def forward(self, data):
        print('here2')
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        print(sa1_out[0].shape)
        print('here3')
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        out = self.mlp(x).log_softmax(dim=-1)

        print(out.shape)

        return out