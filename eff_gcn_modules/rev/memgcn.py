import torch
import torch.nn as nn
import copy
try:
    from .gcn_revop import InvertibleModuleWrapper
except:
    from gcn_revop import InvertibleModuleWrapper

class GroupAdditiveCoupling(torch.nn.Module):
    def __init__(self, Fms, split_dim=-1, group=2):
        super(GroupAdditiveCoupling, self).__init__()

        self.Fms = Fms
        self.split_dim = split_dim
        self.group = group

    def forward(self, x, edge_index, *args):
        xs = torch.chunk(x, self.group, dim=self.split_dim)
        chunked_args = list(map(lambda arg: torch.chunk(arg, self.group, dim=self.split_dim), args))
        args_chunks = list(zip(*chunked_args))
        y_in = sum(xs[1:])

        ys = []
        for i in range(self.group):
            Fmd = self.Fms[i].forward(y_in, edge_index, *args_chunks[i])
            y = xs[i] + Fmd
            y_in = y
            ys.append(y)

        out = torch.cat(ys, dim=self.split_dim)

        return out

    def inverse(self, y, edge_index, *args):
        ys = torch.chunk(y, self.group, dim=self.split_dim)
        chunked_args = list(map(lambda arg: torch.chunk(arg, self.group, dim=self.split_dim), args))
        args_chunks = list(zip(*chunked_args))

        xs = []
        for i in range(self.group-1, -1, -1):
            if i != 0:
                y_in = ys[i-1]
            else:
                y_in = sum(xs)

            Fmd = self.Fms[i].forward(y_in, edge_index, *args_chunks[i])
            x = ys[i] - Fmd
            xs.append(x)

        x = torch.cat(xs[::-1], dim=self.split_dim)

        return x
