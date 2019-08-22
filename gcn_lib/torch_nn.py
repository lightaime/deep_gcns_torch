from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin


##############################
#    Basic layers
##############################
def act_layer(act_type, inplace=False, neg_slope=0.2, n_prelu=1):
    """
    helper selecting activation
    :param act_type:
    :param inplace:
    :param neg_slope:
    :param n_prelu:
    :return:
    """

    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return layer


def norm_layer(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm1d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm1d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return layer


class MultiSeq(Seq):
    def __init__(self, *args):
        super(MultiSeq, self).__init__(*args)

    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class MLP(Seq):
    def __init__(self, channels, act_type='relu', norm_type=None, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(Lin(channels[i - 1], channels[i], bias))
            if act_type:
                m.append(act_layer(act_type))
            if norm_type:
                m.append(norm_layer(norm_type, channels[-1]))
        super(MLP, self).__init__(*m)

