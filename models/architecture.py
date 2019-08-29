import torch

from torch.nn import Linear as Lin

from gcn_lib.sparse import MultiSeq, MLP, GraphConv, ResDynBlock, DenseDynBlock, DilatedKnnGraph
from gcn_lib.dense import BasicConv, GraphConv4D, ResDynBlock4D, DenseDynBlock4D, DenseDilatedKnnGraph
from torch.nn import Sequential as Seq


class SparseDeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(SparseDeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.kernel_size
        act = opt.act_type
        norm = opt.norm_type
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.stochastic
        conv = opt.conv_type
        c_growth = channels

        self.n_blocks = opt.n_blocks

        self.knn = DilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv(opt.in_channels, channels, conv, act, norm, bias)

        if opt.block_type.lower() == 'res':
            self.backbone = MultiSeq(*[ResDynBlock(channels, k, 1+i, conv, act, norm, bias, stochastic, epsilon)
                                       for i in range(self.n_blocks-1)])
        elif opt.block_type.lower() == 'dense':
            self.backbone = MultiSeq(*[DenseDynBlock(channels, k, 1+i, conv, act, norm, bias, stochastic, epsilon)
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
        out = self.prediction(torch.cat((feats, fusion), 1))
        return out


class DenseDeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DenseDeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.kernel_size
        act = opt.act_type
        norm = opt.norm_type
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.stochastic
        conv = opt.conv_type
        c_growth = channels
        self.n_blocks = opt.n_blocks

        self.knn = DenseDilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv4D(opt.in_channels, channels, conv, act, norm, bias)

        if opt.block_type.lower() == 'res':
            self.backbone = Seq(*[ResDynBlock4D(channels, k, 1+i, conv, act, norm, bias, stochastic, epsilon)
                                  for i in range(self.n_blocks-1)])
        elif opt.block_type.lower() == 'dense':
            self.backbone = Seq(*[DenseDynBlock4D(channels+c_growth*i, c_growth, k, 1+i, conv, act,
                                                  norm, bias, stochastic, epsilon)
                                  for i in range(self.n_blocks-1)])
        else:
            raise NotImplementedError('{} is not implemented. Please check.\n'.format(opt.block))
        self.fusion_block = BasicConv([channels+c_growth*(self.n_blocks-1), 1024], act, None, bias)
        self.prediction = Seq(*[BasicConv([1+channels+c_growth*(self.n_blocks-1), 512, 256], act, None, bias),
                                BasicConv([256, opt.n_classes], None, None, bias)])

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        feats = [self.head(inputs, self.knn(inputs[:, 0:3]))]
        for i in range(self.n_blocks-1):
            feats.append(self.backbone[i](feats[-1]))
        feats = torch.cat(feats, 1)
        fusion, _ = torch.max(self.fusion_block(feats), 1, keepdim=True)
        return self.prediction(torch.cat((feats, fusion), 1)).squeeze()


