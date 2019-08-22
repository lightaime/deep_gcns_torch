import torch

from torch.nn import Linear as Lin

from gcn_lib import MultiSeq, MLP, GraphConv, ResDynBlock, DenseDynBlock, DilatedKnnGraph


class DeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.kernel_size
        act = opt.act_type
        norm = opt.norm_type
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.stochastic
        conv = opt.conv_type
        knn_type = opt.knn_type

        self.n_blocks = opt.n_blocks

        self.knn = DilatedKnnGraph(k, 1, stochastic, epsilon, knn_type)
        self.head = GraphConv(opt.in_channels, channels, conv, act, norm, bias, 'max')

        if opt.block_type.lower() == 'res':
            self.backbone = MultiSeq(*[ResDynBlock(channels, k, 1+i, conv, act, norm, bias, 'max', stochastic, epsilon, knn_type)
                                       for i in range(self.n_blocks)])
        elif opt.block_type.lower() == 'dense':
            self.backbone = MultiSeq(*[DenseDynBlock(channels, k, 1+i, conv, act, norm, bias, 'max', stochastic, epsilon, knn_type)
                                       for i in range(self.n_blocks)])
        else:
            raise NotImplementedError('{} is not implemented. Please check.\n'.format(opt.block))

        self.prediction = MultiSeq(*[MLP([channels*(self.n_blocks+1), 512, 256], act, None, bias),
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
        edge_index = self.knn(x[:, 0:3], batch)
        out = self.head(x, edge_index).unsqueeze(0)
        out = self.backbone(out, batch)[0]
        out = self.prediction(out.transpose(1, 0).contiguous().view(out.shape[1], -1))
        return out


