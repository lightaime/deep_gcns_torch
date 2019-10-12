import torch
from torch import nn
from torch_cluster import knn_graph


class Dilated(nn.Module):
    """
    Find dilated neighbor from neighbor list
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(Dilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index, batch=None):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index.view(2, -1, num)
                edge_index = edge_index[:, :, randnum]
                return edge_index.view(2, -1)
            else:
                edge_index = edge_index[:, ::self.dilation]
        else:
            edge_index = edge_index[:, ::self.dilation]
        return edge_index


class DilatedKnnGraph(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0, knn='matrix'):
        super(DilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = Dilated(k, dilation, stochastic, epsilon)
        if knn == 'matrix':
            self.knn = knn_graph_matrix
        else:
            self.knn = knn_graph

    def forward(self, x, batch):
        edge_index = self.knn(x, self.k * self.dilation, batch)
        return self._dilated(edge_index, batch)


def pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    x_inner = -2*torch.matmul(x, x.transpose(2, 1))
    x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
    return x_square + x_inner + x_square.transpose(2, 1)


def knn_matrix(x, k=16, batch=None):
    """Get KNN based on the pairwise distance.
    Args:
        pairwise distance: (num_points, num_points)
        k: int
    Returns:
        nearest neighbors: (num_points*k ,1) (num_points, k)
    """
    if batch is None:
        batch_size = 1
    else:
        batch_size = batch[-1] + 1
    x = x.view(batch_size, -1, x.shape[-1])

    neg_adj = -pairwise_distance(x)
    _, nn_idx = torch.topk(neg_adj, k=k)
    del neg_adj

    n_points = x.shape[1]
    start_idx = torch.arange(0, n_points*batch_size, n_points).long().view(batch_size, 1, 1)
    if x.is_cuda:
        start_idx = start_idx.cuda()
    nn_idx += start_idx
    del start_idx

    if x.is_cuda:
        torch.cuda.empty_cache()

    nn_idx = nn_idx.view(1, -1)
    center_idx = torch.arange(0, n_points*batch_size).repeat(k, 1).transpose(1, 0).contiguous().view(1, -1)
    if x.is_cuda:
        center_idx = center_idx.cuda()
    return nn_idx, center_idx


def knn_graph_matrix(x, k=16, batch=None):
    """Construct edge feature for each point
    Args:
        x: (num_points, num_dims)
        batch: (num_points, )
        k: int
    Returns:
        edge_index: (2, num_points*k)
    """
    nn_idx, center_idx = knn_matrix(x, k, batch)
    return torch.cat((nn_idx, center_idx), dim=0)

