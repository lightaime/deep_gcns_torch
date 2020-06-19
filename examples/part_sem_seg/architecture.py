import torch
from gcn_lib.dense import BasicConv, GraphConv2d, ResDynBlock2d, DenseDynBlock2d, DenseDilatedKnnGraph, PlainDynBlock2d
from torch.nn import Sequential as Seq
import torch.nn.functional as F


class DeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        knn = 'matrix'  # implement knn using matrix multiplication
        epsilon = opt.epsilon
        stochastic = opt.stochastic
        conv = opt.conv
        c_growth = channels
        emb_dims = 1024

        self.n_blocks = opt.n_blocks

        self.knn = DenseDilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv2d(opt.in_channels, channels, conv, act, norm, bias=False)

        if opt.block.lower() == 'res':
            if opt.use_dilation:
                self.backbone = Seq(*[ResDynBlock2d(channels, k, i + 1, conv, act, norm,
                                                    bias, stochastic, epsilon, knn)
                                      for i in range(self.n_blocks - 1)])
            else:
                self.backbone = Seq(*[ResDynBlock2d(channels, k, 1, conv, act, norm,
                                                    bias, stochastic, epsilon, knn)
                                      for _ in range(self.n_blocks - 1)])
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))
        elif opt.block.lower() == 'plain':
            # Plain GCN. No dilation, no stochastic
            stochastic = False
            self.backbone = Seq(*[PlainDynBlock2d(channels, k, 1, conv, act, norm,
                                                bias, stochastic, epsilon, knn)
                                  for i in range(self.n_blocks - 1)])

            fusion_dims = int(channels+c_growth*(self.n_blocks-1))
        else:
            raise NotImplementedError('{} is not supported in this experiment'.format(opt.block))

        self.fusion_block = BasicConv([fusion_dims, emb_dims], 'leakyrelu', norm, bias=False)
        self.prediction = Seq(*[BasicConv([emb_dims * 3, 512], 'leakyrelu', norm, drop=opt.dropout),
                                BasicConv([512, 256], 'leakyrelu', norm, drop=opt.dropout),
                                BasicConv([256, opt.n_classes], None, None)])

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
        fusion = self.fusion_block(feats)

        x1 = F.adaptive_max_pool2d(fusion, 1)
        x2 = F.adaptive_avg_pool2d(fusion, 1)
        feat_global_pool = torch.cat((x1, x2), dim=1)
        feat_global_pool = torch.repeat_interleave(feat_global_pool, repeats=fusion.shape[2], dim=2)
        cat_pooled = torch.cat((feat_global_pool, fusion), dim=1)
        out = self.prediction(cat_pooled).squeeze(-1)
        return F.log_softmax(out, dim=1)

