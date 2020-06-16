import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter, scatter_softmax


class GenMessagePassing(MessagePassing):
    def __init__(self, aggr='softmax',
                 t=1.0, learn_t=False,
                 p=1.0, learn_p=False):

        if aggr == 'softmax' or aggr == 'softmax_sg':

            super(GenMessagePassing, self).__init__(aggr=None)
            self.aggr = aggr

            if learn_t and aggr == 'softmax':
                if t < 1.0:
                    c = torch.nn.Parameter(torch.Tensor([1/t]), requires_grad=True)
                    self.t = 1 / c
                else:
                    self.t = torch.nn.Parameter(torch.Tensor([t]), requires_grad=True)
            else:
                self.t = t

        elif aggr == 'power':

            super(GenMessagePassing, self).__init__(aggr=None)
            self.aggr = aggr

            if learn_p:
                self.p = torch.nn.Parameter(torch.Tensor([p]), requires_grad=True)
            else:
                self.p = p
        else:
            super(GenMessagePassing, self).__init__(aggr=aggr)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):

        if self.aggr in ['add', 'mean', 'max', None]:
            return super(GenMessagePassing, self).aggregate(inputs, index, ptr, dim_size)

        elif self.aggr == 'softmax':
            out = scatter_softmax(inputs*self.t, index, dim=self.node_dim)
            out = scatter(inputs*out, index, dim=self.node_dim,
                          dim_size=dim_size, reduce='sum')
            return out

        elif self.aggr == 'softmax_sg':
            with torch.no_grad():
                out = scatter_softmax(inputs*self.t, index, dim=self.node_dim)
            out = scatter(inputs*out, index, dim=self.node_dim,
                          dim_size=dim_size, reduce='sum')
            return out

        elif self.aggr == 'power':
            min_value, max_value = 1e-7, 1e1
            torch.clamp_(inputs, min_value, max_value)
            out = scatter(torch.pow(inputs, self.p), index, dim=self.node_dim,
                          dim_size=dim_size, reduce='mean')
            torch.clamp_(out, min_value, max_value)
            return torch.pow(out, 1/self.p)

        else:
            raise NotImplementedError('To be implemented')


class MsgNorm(torch.nn.Module):
    def __init__(self, learn_msg_scale=False):
        super(MsgNorm, self).__init__()

        self.msg_scale = torch.nn.Parameter(torch.Tensor([1.0]),
                                            requires_grad=learn_msg_scale)

    def forward(self, x, msg, p=2):
        msg = F.normalize(msg, p=p, dim=1)
        x_norm = x.norm(p=p, dim=1, keepdim=True)
        msg = msg * x_norm * self.msg_scale
        return msg
