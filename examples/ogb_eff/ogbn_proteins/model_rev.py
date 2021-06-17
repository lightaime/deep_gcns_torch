import __init__
import torch
import torch.nn as nn
from gcn_lib.sparse.torch_nn import norm_layer
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import logging
import eff_gcn_modules.rev.memgcn as memgcn
from eff_gcn_modules.rev.rev_layer import GENBlock
import copy


class RevGCN(torch.nn.Module):
    def __init__(self, args):
        super(RevGCN, self).__init__()

        self.inject_input = args.inject_input
        self.num_layers = args.num_layers
        self.num_steps = args.num_steps
        self.dropout = args.dropout
        self.block = args.block
        self.group = args.group

        hidden_channels = args.hidden_channels
        num_tasks = args.num_tasks
        conv = args.conv
        aggr = args.gcn_aggr

        t = args.t
        self.learn_t = args.learn_t
        p = args.p
        self.learn_p = args.learn_p
        self.msg_norm = args.msg_norm
        learn_msg_scale = args.learn_msg_scale

        conv_encode_edge = args.conv_encode_edge
        norm = args.norm
        mlp_layers = args.mlp_layers
        node_features_file_path = args.nf_path

        self.use_one_hot_encoding = args.use_one_hot_encoding

        self.checkpoint_grad = False
        if self.num_steps > 15 or hidden_channels > 64:
            self.checkpoint_grad = True
            self.ckp_k = 10

        print('The number of layers {}'.format(self.num_layers),
              'Aggregation method {}'.format(aggr),
              'block: {}'.format(self.block))

        if self.block == 'res+':
            print('LN/BN->ReLU->GraphConv->Res')
        elif self.block == 'res':
            print('GraphConv->LN/BN->ReLU->Res')
        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')
        elif self.block == "plain":
            print('GraphConv->LN/BN->ReLU')
        else:
            raise Exception('Unknown block Type')

        self.gcns = torch.nn.ModuleList()
        self.last_norm = norm_layer(norm, hidden_channels)

        for layer in range(self.num_layers):
            Fms = nn.ModuleList()
            fm = GENBlock(hidden_channels//self.group, hidden_channels//self.group,
                          aggr=aggr,
                          t=t, learn_t=self.learn_t,
                          p=p, learn_p=self.learn_p,
                          msg_norm=self.msg_norm,
                          learn_msg_scale=learn_msg_scale,
                          encode_edge=conv_encode_edge,
                          edge_feat_dim=hidden_channels,
                          norm=norm, mlp_layers=mlp_layers,
                          dropout=self.dropout)

            for i in range(self.group):
                if i == 0:
                    Fms.append(fm)
                else:
                    Fms.append(copy.deepcopy(fm))


            invertible_module = memgcn.GroupAdditiveCoupling(Fms,
                                                             group=self.group,
                                                             pre_shuffle=False)


            gcn = memgcn.InvertibleModuleWrapper(fn=invertible_module,
                                                 keep_input=False)

            self.gcns.append(gcn)

        self.node_features = torch.load(node_features_file_path).to(args.device)

        if self.use_one_hot_encoding:
            self.node_one_hot_encoder = torch.nn.Linear(8, 8)
            self.node_features_encoder = torch.nn.Linear(8 * 2, hidden_channels)
        else:
            self.node_features_encoder = torch.nn.Linear(8, hidden_channels)

        self.edge_encoder = torch.nn.Linear(8, hidden_channels)

        self.node_pred_linear = torch.nn.Linear(hidden_channels, num_tasks)

    def forward(self, x, node_index, edge_index, edge_attr, epoch=-1):

        node_features_1st = self.node_features[node_index]

        if self.use_one_hot_encoding:
            node_features_2nd = self.node_one_hot_encoder(x)
            # concatenate
            node_features = torch.cat((node_features_1st, node_features_2nd), dim=1)
        else:
            node_features = node_features_1st

        h = self.node_features_encoder(node_features)

        edge_emb = self.edge_encoder(edge_attr)
        edge_emb = torch.cat([edge_emb]*self.group, dim=-1)

        m = torch.zeros_like(h).bernoulli_(1 - self.dropout)
        mask = m.requires_grad_(False) / (1 - self.dropout)

        h = self.gcns[0](h, edge_index, mask, edge_emb)

        for layer in range(1, self.num_layers):
            h = self.gcns[layer](h, edge_index, mask, edge_emb)

        h = F.relu(self.last_norm(h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        return self.node_pred_linear(h)


    def print_params(self, epoch=None, final=False):

        if self.learn_t:
            ts = []
            for gcn in self.gcns:
                ts.append(gcn.t.item())
            if final:
                print('Final t {}'.format(ts))
            else:
                logging.info('Epoch {}, t {}'.format(epoch, ts))
        if self.learn_p:
            ps = []
            for gcn in self.gcns:
                ps.append(gcn.p.item())
            if final:
                print('Final p {}'.format(ps))
            else:
                logging.info('Epoch {}, p {}'.format(epoch, ps))
        if self.msg_norm:
            ss = []
            for gcn in self.gcns:
                ss.append(gcn.msg_norm.msg_scale.item())
            if final:
                print('Final s {}'.format(ss))
            else:
                logging.info('Epoch {}, s {}'.format(epoch, ss))
