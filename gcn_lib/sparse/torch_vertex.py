import torch
from torch import nn
import torch_geometric as tg
from .torch_nn import MLP
from .torch_edge import DilatedKnnGraph


class MRConv(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751)
    """
    def __init__(self, in_channels, out_channels, act_type='relu', norm_type=None, bias=True, aggr='max'):
        super(MRConv, self).__init__()
        self.nn = MLP([in_channels*2, out_channels], act_type, norm_type, bias)
        self.aggr = aggr

    def forward(self, x, edge_index):
        """"""
        x_j = tg.utils.scatter_(self.aggr, torch.index_select(x, 0, edge_index[0]) - torch.index_select(x, 0, edge_index[1]), edge_index[1])
        return self.nn(torch.cat([x, x_j], dim=1))


class EdgConv(tg.nn.EdgeConv):
    """
    Edge convolution layer (with activation, batch normalization)
    """
    def __init__(self, in_channels, out_channels, act_type='relu', norm_type=None, bias=True, aggr='max'):
        super(EdgConv, self).__init__(MLP([in_channels*2, out_channels], act_type, norm_type, bias), aggr)

    def forward(self, x, edge_index):
        return super(EdgConv, self).forward(x, edge_index)


class GraphConv(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv_type='edge',
                 act_type='relu', norm_type=None, bias=True):
        super(GraphConv, self).__init__()
        if conv_type == 'edge':
            self.gconv = EdgConv(in_channels, out_channels, act_type, norm_type, bias)
        elif conv_type == 'mr':
            self.gconv = MRConv(in_channels, out_channels, act_type, norm_type, bias)

    def forward(self, x, edge_index):
        return self.gconv(x, edge_index)


class DynConv(GraphConv):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv_type='edge', act_type='relu',
                 norm_type=None, bias=True,  stochastic=False, epsilon=1.0, knn_type='matrix'):
        super(DynConv, self).__init__(in_channels, out_channels, conv_type, act_type, norm_type, bias)
        self.k = kernel_size
        self.d = dilation
        self.dilated_knn_graph = DilatedKnnGraph(kernel_size, dilation, stochastic, epsilon, knn_type)

    def forward(self, x, batch=None):
        edge_index = self.dilated_knn_graph(x, batch)
        return super(DynConv, self).forward(x, edge_index)


class ResDynBlock(nn.Module):
    """
    Residual Dynamic graph convolution block
        :input: (x0, x1, x2, ... , xi), batch
        :output:(x0, x1, x2, ... , xi ,xi+1) , batch
    """
    def __init__(self, channels,  kernel_size=9, dilation=1, conv_type='edge', act_type='relu', norm_type=None,
                 bias=True, stochastic=False, epsilon=1.0, knn_type='matrix'):
        super(ResDynBlock, self).__init__()
        self.body = DynConv(channels, channels, kernel_size, dilation, conv_type,
                            act_type, norm_type, bias, stochastic, epsilon, knn_type)

    # input: (x0, x1, x2, ..., xi);  (xi-1, xi), output is (xi, xi+1)
    def forward(self, x, batch):
        return self.body(x, batch) + x, batch


class DenseDynBlock(nn.Module):
    """
    Dense Dynamic graph convolution block
    """
    def __init__(self, channels,  kernel_size=9, dilation=1, conv_type='edge', act_type='relu', norm_type=None,
                 bias=True, stochastic=False, epsilon=1.0, knn_type='matrix'):
        super(DenseDynBlock, self).__init__()
        self.body = DynConv(channels*2, channels, kernel_size, dilation, conv_type,
                            act_type, norm_type, bias, stochastic, epsilon, knn_type)

    def forward(self, x, batch):
        dense = self.body(batch)
        return torch.cat((x, dense), 1), batch



