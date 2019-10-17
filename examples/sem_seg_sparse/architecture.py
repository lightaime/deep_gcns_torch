import torch
from torch.nn import Linear as Lin
import torch_geometric as tg
from gcn_lib.sparse import MultiSeq, MLP, GraphConv, ResDynBlock, DenseDynBlock, DilatedKnnGraph


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
        self.fusion_block = MLP([channels + c_growth * (self.n_blocks - 1), 1024], act, norm, bias)
        self.prediction = MultiSeq(*[MLP([channels+c_growth*(self.n_blocks-1)+1024, 512], act, norm, bias),
                                     MLP([512, 256], act, norm, bias),
                                     torch.nn.Dropout(p=opt.dropout),
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
        x = torch.cat((corr, color), dim=1)
        feats = [self.head(x, self.knn(x[:, 0:3], batch))]
        for i in range(self.n_blocks-1):
            feats.append(self.backbone[i](feats[-1], batch)[0])
        feats = torch.cat(feats, dim=1)

        fusion = tg.utils.scatter_('max', self.fusion_block(feats), batch)
        fusion = torch.repeat_interleave(fusion, repeats=feats.shape[0]//fusion.shape[0], dim=0)
        return self.prediction(torch.cat((fusion, feats), dim=1))
