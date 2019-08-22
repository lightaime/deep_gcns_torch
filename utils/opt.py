import os
import datetime
import argparse
import shutil

import torch
from .tf_logger import TfLogger
from .metrics import AverageMeter


class OptInit():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch implementation of Deep GCN')

        # task
        parser.add_argument('--task', type=str, default='sem_seg')

        # dataset args
        parser.add_argument('--train_path', type=str, default='/data/deepgcn/S3DIS')
        parser.add_argument('--test_path', type=str, default='/data/deepgcn/S3DIS')
        parser.add_argument('--batch_size', default=8, type=int, help='mini-batch size (default:8)')
        parser.add_argument('--in_channels', default=9, type=int, help='the channel size of input point cloud ')

        # train args
        parser.add_argument('--total_epochs', default=150, type=int, help='number of total epochs to run')
        parser.add_argument('--iter', default=0, type=int, help='number of iteration to start')
        parser.add_argument('--lr_adjust_freq', default=20, type=int, help='decay lr after certain number of epoch')
        parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
        parser.add_argument('--print_freq', default=100, type=int, help='print frequency of training (default: 100)')
        parser.add_argument('--postname', type=str, default='', help='postname of saved file')
        parser.add_argument('--multi_gpus', action='store_true', help='use multi-gpus')
        parser.add_argument('--cpu', action='store_true', help='use cpu?')

        # test args
        parser.add_argument('--train', action='store_true', help='train or test(default)')
        parser.add_argument('--visu', action='store_true', help='set --visu if need visualization in test phase')
        parser.add_argument('--no_clutter', action='store_true', help='no clutter? set --no_clutter if ture.')

        # model args
        parser.add_argument('--pretrained_model', type=str, help='path to pretrained model(default: none)', default='')
        parser.add_argument('--model_name', type=str, default='DeepGCN')
        parser.add_argument('--kernel_size', default=16, type=int, help='neighbor num (default:16)')
        parser.add_argument('--block_type', default='res', type=str, help='graph backbone block type {res, dense}')
        parser.add_argument('--conv_type', default='edge', type=str, help='graph conv layer {edge, mr}')
        parser.add_argument('--act_type', default='relu', type=str, help='activation layer {relu, prelu, leakyrelu}')
        parser.add_argument('--norm_type', default='batch', type=str, help='batch or instance normalization')
        parser.add_argument('--knn_type', default='matrix', type=str, help='matrix or tree(pytorch geometric)')
        parser.add_argument('--bias', default=True,  type=bool, help='bias of conv layer True or False')
        parser.add_argument('--n_filters', default=64, type=int, help='number of channels of deep features')
        parser.add_argument('--n_blocks', default=28, type=int, help='number of basic blocks')

        # dilated knn
        parser.add_argument('--epsilon', default=0.2, type=float, help='stochastic epsilon for gcn')
        parser.add_argument('--stochastic', default=True,  type=bool, help='stochastic for gcn, True or False')

        args = parser.parse_args()

        args.post = '-'.join([args.model_name.lower(), args.block_type, args.conv_type])
        if args.postname:
            args.post += '-' + args.postname
        args.post += '-' + datetime.datetime.now().strftime("%y%m%d")

        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.pretrained_model = os.path.join(ROOT_DIR, args.pretrained_model)
        args.save_path = os.path.join(ROOT_DIR, args.task, 'checkpoints/checkpoints'+'-'+args.post)
        args.logdir = os.path.join(ROOT_DIR, args.task, 'logs/'+args.post)

        if args.cpu:
            args.device = torch.device('cpu')
        else:
            args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.args = args

    def initialize(self):
        self.args.valid_values = AverageMeter()
        self.args.valid_value = 0
        self.args.best_loss = 0

        if self.args.train:
            # logger
            self.args.logger = TfLogger(self.args.logdir)
            # loss
            self.args.losses = AverageMeter()
            self.args.epoch = -1
            self.make_dir()

        self.print_args()
        return self.args

    def print_args(self):
        # print args
        print("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            print("{}:{}".format(arg, content))
        print("==========     CONFIG END    =============")

        print("\n")
        if self.args.train:
            print('!!! Phase is training. If you want to test, please remove --train.')
        else:
            print('!!! Phase is testing. If you want to train, please add --train.')
        print("\n")

    def make_dir(self):
        # check for folders existence
        shutil.rmtree(self.args.logdir)
        os.makedirs(self.args.logdir)

        if not os.path.exists(self.args.save_path):		
            os.makedirs(self.args.save_path)

