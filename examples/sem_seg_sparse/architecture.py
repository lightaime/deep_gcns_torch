import __init__
import torch
from torch.nn import Linear as Lin
from gcn_lib.sparse import MultiSeq, MLP, GraphConv, PlainDynBlock, ResDynBlock, DenseDynBlock, DilatedKnnGraph
from utils.pyg_util import scatter_
from torch_geometric.data import Data


class SparseDeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(SparseDeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.k
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
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))
        elif opt.block.lower() == 'dense':
            self.backbone = MultiSeq(*[DenseDynBlock(channels+c_growth*i, c_growth, k, 1+i,
                                                     conv, act, norm, bias, stochastic=stochastic, epsilon=epsilon)
                                       for i in range(self.n_blocks-1)])
            fusion_dims = int(
                (channels + channels + c_growth * (self.n_blocks - 1)) * self.n_blocks // 2)
        else:
            # Use PlainGCN without skip connection and dilated convolution.
            stochastic = False
            self.backbone = MultiSeq(
                *[PlainDynBlock(channels, k, 1, conv, act, norm, bias, stochastic=stochastic, epsilon=epsilon)
                  for i in range(self.n_blocks - 1)])
            fusion_dims = int(channels + c_growth * (self.n_blocks - 1))

        self.fusion_block = MLP([fusion_dims, 1024], act, norm, bias)
        self.prediction = MultiSeq(*[MLP([fusion_dims+1024, 512], act, norm, bias),
                                     MLP([512, 256], act, norm, bias, drop=opt.dropout),
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

        fusion = scatter_('max', self.fusion_block(feats), batch)
        fusion = torch.repeat_interleave(fusion, repeats=feats.shape[0]//fusion.shape[0], dim=0)
        return self.prediction(torch.cat((fusion, feats), dim=1))


if __name__ == "__main__":
    import random, numpy as np, argparse
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_size = 2
    N = 1024
    device = 'cuda'

    parser = argparse.ArgumentParser(description='PyTorch implementation of Deep GCN For semantic segmentation')
    parser.add_argument('--in_channels', default=9, type=int, help='input channels (default:9)')
    parser.add_argument('--n_classes', default=13, type=int, help='num of segmentation classes (default:13)')
    parser.add_argument('--k', default=20, type=int, help='neighbor num (default:16)')
    parser.add_argument('--block', default='res', type=str, help='graph backbone block type {plain, res, dense}')
    parser.add_argument('--conv', default='edge', type=str, help='graph conv layer {edge, mr}')
    parser.add_argument('--act', default='relu', type=str, help='activation layer {relu, prelu, leakyrelu}')
    parser.add_argument('--norm', default='batch', type=str, help='{batch, instance} normalization')
    parser.add_argument('--bias', default=True, type=bool, help='bias of conv layer True or False')
    parser.add_argument('--n_filters', default=64, type=int, help='number of channels of deep features')
    parser.add_argument('--n_blocks', default=7, type=int, help='number of basic blocks')
    parser.add_argument('--dropout', default=0.5, type=float, help='ratio of dropout')
    parser.add_argument('--epsilon', default=0.2, type=float, help='stochastic epsilon for gcn')
    parser.add_argument('--stochastic', default=False, type=bool, help='stochastic for gcn, True or False')
    args = parser.parse_args()

    pos = torch.rand((batch_size*N, 3), dtype=torch.float).to(device)
    x = torch.rand((batch_size*N, 6), dtype=torch.float).to(device)

    data = Data()
    data.pos = pos
    data.x = x
    data.batch = torch.arange(batch_size).unsqueeze(-1).expand(-1, N).contiguous().view(-1).contiguous()
    data = data.to(device)

    net = SparseDeepGCN(args).to(device)
    print(net)

    out = net(data)

    print('out logits shape', out.shape)
    import time 
    st = time.time()
    runs = 1000

    with torch.no_grad():
        for i in range(runs):
            # print(i)
            out = net(data)
            torch.cuda.synchronize()
    print(time.time() - st)

