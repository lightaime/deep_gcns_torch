#!/usr/bin/env python
# -*- coding: utf-8 -*-
import __init__
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from gcn_lib.dense import BasicConv, GraphConv2d, ResDynBlock2d, DenseDynBlock2d, DilatedKnnGraph, PlainDynBlock2d


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
        emb_dims = opt.emb_dims
        self.n_blocks = opt.n_blocks

        self.knn = DilatedKnnGraph(k, 1, stochastic, epsilon)
        self.head = GraphConv2d(opt.in_channels, channels, conv, act, norm, bias=False)

        if opt.block.lower() == 'dense':
            self.backbone = Seq(*[DenseDynBlock2d(channels+c_growth*i, c_growth, k, 1+i, conv, act,
                                                  norm, bias, stochastic, epsilon, knn)
                                  for i in range(self.n_blocks-1)])
            fusion_dims = int(
                (channels + channels + c_growth * (self.n_blocks-1)) * self.n_blocks // 2)

        elif opt.block.lower() == 'res':
            if opt.use_dilation:
                self.backbone = Seq(*[ResDynBlock2d(channels, k, i + 1, conv, act, norm,
                                                    bias, stochastic, epsilon, knn)
                                      for i in range(self.n_blocks - 1)])
            else:
                self.backbone = Seq(*[ResDynBlock2d(channels, k, 1, conv, act, norm,
                                                    bias, stochastic, epsilon, knn)
                                      for _ in range(self.n_blocks - 1)])
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))
        else:
            # Plain GCN. No dilation, no stochastic
            stochastic = False

            self.backbone = Seq(*[PlainDynBlock2d(channels, k, 1, conv, act, norm,
                                                  bias, stochastic, epsilon, knn)
                                  for i in range(self.n_blocks - 1)])

            fusion_dims = int(channels+c_growth*(self.n_blocks-1))

        # fusion_dims = int((channels + channels + c_growth*self.num_backbone_layers)*(self.num_backbone_layers+1)//2)
        self.fusion_block = BasicConv([fusion_dims, emb_dims], 'leakyrelu', norm, bias=False)
        self.prediction = Seq(*[BasicConv([emb_dims * 2, 512], 'leakyrelu', norm, drop=opt.dropout),
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

        feats = torch.cat(feats, dim=1)
        fusion = self.fusion_block(feats)
        x1 = F.adaptive_max_pool2d(fusion, 1)
        x2 = F.adaptive_avg_pool2d(fusion, 1)
        return self.prediction(torch.cat((x1, x2), dim=1)).squeeze(-1).squeeze(-1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Point Cloud Segmentation')
    # ----------------- Model related
    parser.add_argument('--k', default=9, type=int, help='neighbor num (default:9)')
    parser.add_argument('--block', default='res', type=str, help='graph backbone block type {res, plain, dense}')
    parser.add_argument('--conv', default='edge', type=str, help='graph conv layer {edge, mr}')
    parser.add_argument('--act', default='relu', type=str, help='activation layer {relu, prelu, leakyrelu}')
    parser.add_argument('--norm', default='batch', type=str,
                        help='batch or instance normalization {batch, instance}')
    parser.add_argument('--bias', default=True, type=bool, help='bias of conv layer True or False')
    parser.add_argument('--n_blocks', type=int, default=14, help='number of basic blocks in the backbone')
    parser.add_argument('--n_filters', default=64, type=int, help='number of channels of deep features')
    parser.add_argument('--in_channels', type=int, default=3, help='Dimension of input ')
    parser.add_argument('--n_classes', type=int, default=40, help='Dimension of out_channels ')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    # dilated knn
    parser.add_argument('--use_dilation', default=True, type=bool, help='use dilated knn or not')
    parser.add_argument('--epsilon', default=0.2, type=float, help='stochastic epsilon for gcn')
    parser.add_argument('--stochastic', default=True, type=bool, help='stochastic for gcn, True or False')

    args = parser.parse_args()
    args.device = torch.device('cuda')

    feats = torch.rand((2, 3, 1024, 1), dtype=torch.float).to(args.device)
    num_neighbors = 20


    print('Input size {}'.format(feats.size()))
    net = DeepGCN(args).to(args.device)
    out = net(feats)

    print('Output size {}'.format(out.size()))
