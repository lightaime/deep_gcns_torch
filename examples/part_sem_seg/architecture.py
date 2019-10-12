import torch
from gcn_lib.dense import BasicConv, GraphConv2d, ResDynBlock2d, DenseDynBlock2d, DilatedKnnGraph
from torch.nn import Sequential as Seq
import torch.nn.functional as F


class DenseDeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DenseDeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.kernel_size
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        knn = opt.knn
        epsilon = opt.epsilon
        stochastic = opt.stochastic
        conv = opt.conv
        c_growth = channels
        res_scale = 1

        self.n_blocks = opt.n_blocks

        self.knn = DilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv2d(opt.in_channels, channels, conv, act, norm, bias)

        if opt.block.lower() == 'dense':
            self.backbone = Seq(*[DenseDynBlock2d(channels+c_growth*i, c_growth, k, 1+i, conv, act,
                                                  norm, bias, stochastic, epsilon, knn)
                                  for i in range(self.n_blocks-1)])

        else:
            if not opt.block.lower() == 'res':  # plain gcn
                res_scale = 0
            if opt.use_dilation:
                self.backbone = Seq(*[ResDynBlock2d(channels, k, i + 1, conv, act, norm,
                                                    bias, stochastic, epsilon, knn, res_scale)
                                      for i in range(self.n_blocks - 1)])
            else:
                self.backbone = Seq(*[ResDynBlock2d(channels, k, 1, conv, act, norm,
                                                    bias, stochastic, epsilon, knn, res_scale)
                                      for _ in range(self.n_blocks - 1)])

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
        out = self.prediction(torch.cat((feats, fusion), 1)).squeeze(-1)
        return F.log_softmax(out, dim=1)

