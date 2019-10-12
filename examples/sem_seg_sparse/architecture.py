import torch

from torch.nn import Linear as Lin

from gcn_lib.sparse import MultiSeq, MLP, GraphConv, ResDynBlock, DenseDynBlock, DilatedKnnGraph
from gcn_lib.dense import BasicConv, GraphConv2d, ResDynBlock2d, DenseDynBlock2d, DenseDilatedKnnGraph
from torch.nn import Sequential as Seq


class SparseDeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(SparseDeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.kernel_size
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.stochastic
        conv = opt.conv
        c_growth = channels

        self.n_blocks = opt.n_blocks

        self.knn = DilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv(opt.in_channels, channels, conv, act, norm, bias)

        if opt.block.lower() == 'res':
            self.backbone = MultiSeq(*[ResDynBlock(channels, k, 1+i, conv, act, norm, bias, stochastic=stochastic, epsilon=epsilon)
                                       for i in range(self.n_blocks-1)])
        elif opt.block.lower() == 'dense':
            self.backbone = MultiSeq(*[DenseDynBlock(channels, k, 1+i, conv, act, norm, bias, stochastic=stochastic, epsilon=epsilon)
                                       for i in range(self.n_blocks-1)])
        else:
            raise NotImplementedError('{} is not implemented. Please check.\n'.format(opt.block))
        self.fusion_block = MLP([channels + c_growth * (self.n_blocks - 1), 1024], act, None, bias)
        self.prediction = MultiSeq(*[MLP([1+channels+c_growth*(self.n_blocks-1), 512, 256], act, None, bias),
                                     MLP([256, opt.n_classes], None, None, bias)])

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, data):
        corr, color, batch = data.pos, data.x, data.batch
        x = torch.cat((corr, color), 1)
        feats = [self.head(x, self.knn(x[:, 0:3], batch))]
        for i in range(self.n_blocks-1):
            feats.append(self.backbone[i](feats[-1], batch)[0])
        feats = torch.cat(feats, 1)
        fusion, _ = torch.max(self.fusion_block(feats), 1, keepdim=True)
        return self.prediction(torch.cat((feats, fusion), 1))
