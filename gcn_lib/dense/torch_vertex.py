import torch
from torch import nn
from .torch_nn import BasicConv, batched_index_select
from .torch_edge import DenseDilatedKnnGraph, DilatedKnnGraph
import torch.nn.functional as F


class MRConv2d(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels*2, out_channels], act, norm, bias)

    def forward(self, x, edge_index):
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        return self.nn(torch.cat([x, x_j], dim=1))


class EdgeConv2d(nn.Module):
    """
    Edge convolution layer (with activation, batch normalization) for dense data type
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index):
        x_i = batched_index_select(x, edge_index[1])
        x_j = batched_index_select(x, edge_index[0])
        max_value, _ = torch.max(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return max_value


class GraphConv2d(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        elif conv == 'mr':
            self.gconv = MRConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index):
        return self.gconv(x, edge_index)


class DynConv2d(GraphConv2d):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, knn='matrix'):
        super(DynConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size
        self.d = dilation
        if knn == 'matrix':
            self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)
        else:
            self.dilated_knn_graph = DilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

    def forward(self, x):
        edge_index = self.dilated_knn_graph(x)
        return super(DynConv2d, self).forward(x, edge_index)


class ResDynBlock2d(nn.Module):
    """
    Residual Dynamic graph convolution block
        :input: (x0, x1, x2, ... , xi), batch
        :output:(x0, x1, x2, ... , xi ,xi+1) , batch
    """
    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, knn='matrix', res_scale=1):
        super(ResDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, in_channels, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, knn)
        self.res_scale = res_scale

    def forward(self, x):
        return self.body(x) + x*self.res_scale


class DenseDynBlock2d(nn.Module):
    """
    Dense Dynamic graph convolution block
    """
    def __init__(self, in_channels, out_channels=64,  kernel_size=9, dilation=1, conv='edge',
                 act='relu', norm=None,bias=True, stochastic=False, epsilon=0.0, knn='matrix'):
        super(DenseDynBlock2d, self).__init__()
        self.body = DynConv2d(in_channels, out_channels, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, knn)

    def forward(self, x):
        dense = self.body(x)
        return torch.cat((x, dense), 1)


class GraphPooling(nn.Module):
    """
    Dense Dynamic graph pooling block
    """
    def __init__(self, in_channels, ratio=0.5, conv='edge', **kwargs):
        super(GraphPooling, self).__init__()
        self.gnn = DynConv2d(in_channels, 1, conv=conv, **kwargs)
        self.ratio = ratio

    def forward(self, x):
        """"""
        score = torch.tanh(self.gnn(x))
        _, indices = score.topk(int(x.shape[2]*self.ratio), 2)
        return torch.gather(x, 2, indices.repeat(1, x.shape[1], 1, 1))


class VLADPool(torch.nn.Module):
    def __init__(self, in_channels, num_clusters=64, alpha=100.0):
        super(VLADPool, self).__init__()
        self.in_channels = in_channels
        self.num_clusters = num_clusters
        self.alpha = alpha

        self.lin = nn.Linear(in_channels, self.num_clusters, bias=True)

        self.centroids = nn.Parameter(torch.rand(self.num_clusters, in_channels))
        self._init_params()

    def _init_params(self):
        self.lin.weight = nn.Parameter((2.0 * self.alpha * self.centroids))
        self.lin.bias = nn.Parameter(- self.alpha * self.centroids.norm(dim=1))

    def forward(self, x, norm_intra=False, norm_L2=False):
        B, C, N, _ = x.shape
        x = x.squeeze().transpose(1, 2)  # B, N, C
        K = self.num_clusters
        soft_assign = self.lin(x)  # soft_assign of size (B, N, K)
        soft_assign = F.softmax(soft_assign, dim=1).unsqueeze(1)  # soft_assign of size (B, N, K)
        soft_assign = soft_assign.expand(-1, C, -1, -1)  # soft_assign of size (B, C, N, K)

        # input x of size (NxC)
        xS = x.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, K)  # xS of size (B, C, N, K)
        cS = self.centroids.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1).transpose(2, 3)  # cS of size (B, C, N, K)

        residual = (xS - cS)  # residual of size (B, C, N, K)
        residual = residual * soft_assign  # vlad of size (B, C, N, K)

        vlad = torch.sum(residual, dim=2, keepdim=True)  # (B, C, K)

        if (norm_intra):
            vlad = F.normalize(vlad, p=2, dim=1)  # intra-normalization
            # print("i-norm vlad", vlad.shape)
        if (norm_L2):
            vlad = vlad.view(-1, K * C)  # flatten
            vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        # return vlad.view(B, -1, 1, 1)
        return vlad


