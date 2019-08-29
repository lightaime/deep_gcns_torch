import torch
from torch import nn
import torch_geometric as tg
from .torch_nn import MLP, BasicConv, batched_index_select
from .torch_edge import DenseDilatedKnnGraph


class MRConv4D(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act_type='relu', norm_type=None, bias=True):
        super(MRConv4D, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act_type, norm_type, bias)

    def forward(self, x, edge_index):
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        return self.nn(torch.cat([x, x_j], dim=1))


class EdgeConv4D(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act_type='relu', norm_type=None, bias=True):
        super(EdgeConv4D, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act_type, norm_type, bias)

    def forward(self, x, edge_index):
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphConv4D(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv_type='edge', act_type='relu', norm_type=None, bias=True):
        super(GraphConv4D, self).__init__()
        if conv_type == 'edge':
            self.gconv = EdgeConv4D(in_channels, out_channels, act_type, norm_type, bias)
        elif conv_type == 'mr':
            self.gconv = MRConv4D(in_channels, out_channels, act_type, norm_type, bias)
        else:
            raise NotImplementedError('conv_type is not supported')

    def forward(self, x, edge_index):
        return self.gconv(x, edge_index)


class DynConv4D(GraphConv4D):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv_type='edge', act_type='relu',
                 norm_type=None, bias=True, stochastic=False, epsilon=0.0):
        super(DynConv4D, self).__init__(in_channels, out_channels, conv_type, act_type, norm_type, bias)
        self.k = kernel_size
        self.d = dilation
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x):
        edge_index = self.dilated_knn_graph(x)
        return super(DynConv4D, self).forward(x, edge_index)


class ResDynBlock4D(nn.Module):
    """
    Residual Dynamic graph convolution block
        :input: (x0, x1, x2, ... , xi), batch
        :output:(x0, x1, x2, ... , xi ,xi+1) , batch
    """
    def __init__(self, channels,  kernel_size=9, dilation=1, conv_type='edge', act_type='relu', norm_type=None,
                 bias=True,  stochastic=False, epsilon=0.0):
        super(ResDynBlock4D, self).__init__()
        self.body = DynConv4D(channels, channels, kernel_size, dilation, conv_type,
                              act_type, norm_type, bias, stochastic, epsilon)

    def forward(self, x):
        return self.body(x) + x


class DenseDynBlock4D(nn.Module):
    """
    Dense Dynamic graph convolution block
    """
    def __init__(self, in_channels, out_channels=64,  kernel_size=9, dilation=1, conv_type='edge',
                 act_type='relu', norm_type=None,
                 bias=True, stochastic=False, epsilon=0.0):
        super(DenseDynBlock4D, self).__init__()
        self.body = DynConv4D(in_channels, out_channels, kernel_size, dilation, conv_type,
                              act_type, norm_type, bias, stochastic, epsilon)

    def forward(self, x):
        dense = self.body(x)
        return torch.cat((x, dense), 1)


