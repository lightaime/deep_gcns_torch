import os
import sys
import argparse
import shutil
import random
import numpy as np
import torch
import logging
import time
import uuid
import pathlib
import glob
from torch.utils.tensorboard import SummaryWriter


class OptInit:
    def __init__(self):
        # ===> argparse
        parser = argparse.ArgumentParser(description='PyTorch implementation of Deep GCN For ModelNet Classification')
        # ----------------- Base
        parser.add_argument('--phase', type=str, default='train', metavar='N',
                            choices=['train', 'test'])
        parser.add_argument('--exp_name', type=str, default='', help='experiment name')
        parser.add_argument('--job_name', type=str, default='', help='full name of exp directory (exp_name + timetamp')
        parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu?')
        parser.add_argument('--root_dir', type=str, default='log', help='the dir of all experiment results, ckpt and logs')

        # ----------------- Dataset related
        parser.add_argument('--data_dir', type=str, default='/data/deepgcn/modelnet40',
                            help="data dir, will download dataset here automatically")
        parser.add_argument('--num_points', type=int, default=1024,
                            help='num of points to use')
        parser.add_argument('--augment', action='store_true', default=True, help='Data Augmentation')
        parser.add_argument('--in_channels', type=int, default=3, help='Dimension of input ')

        # ----------------- Training related
        parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                            help='mini-batch size (default:16))')
        parser.add_argument('--epochs', type=int, default=400, metavar='N',
                            help='number of episode to train ')
        parser.add_argument('--use_sgd', type=bool, default=True, help='Use SGD')
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 regularization')
        parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                            help='learning rate (default: 0.001)')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--multi_gpus', action='store_true', help='use multi-gpus')

        # ----------------- Testing related
        parser.add_argument('--test_batch_size', type=int, default=50, metavar='batch_size',
                            help='Size of batch)')
        parser.add_argument('--pretrained_model', type=str, default='', metavar='N',
                            help='Pretrained model path')

        # ----------------- Model related
        parser.add_argument('--k', default=9, type=int, help='neighbor num (default:9)')
        parser.add_argument('--block', default='res', type=str, help='graph backbone block type {res, plain, dense}')
        parser.add_argument('--conv', default='edge', type=str, help='graph conv layer {edge, mr}')
        parser.add_argument('--act', default='relu', type=str, help='activation layer {relu, prelu, leakyrelu}')
        parser.add_argument('--norm', default='batch', type=str,
                            help='batch or instance normalization {batch, instance}')
        parser.add_argument('--bias', default=True, type=bool, help='bias of conv layer True or False')
        parser.add_argument('--n_blocks', type=int, default=28, help='number of basic blocks in the backbone')
        parser.add_argument('--n_filters', default=64, type=int, help='number of channels of deep features')
        parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
        parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')

        # dilated knn
        parser.add_argument('--no_dilation', action="store_false", dest="use_dilation",
                            help='use dilated knn or not')
        parser.add_argument('--epsilon', default=0.2, type=float, help='stochastic epsilon for gcn')
        parser.add_argument('--no_stochastic', action="store_false", dest="use_stochastic",
                            help='stochastic for gcn, True or False')

        args = parser.parse_args()
        args.device = torch.device('cuda' if not args.use_cpu and torch.cuda.is_available() else 'cpu')
        self.args = args

        # ===> generate log dir
        if self.args.phase == 'train':
            # generate exp_dir when pretrained model does not exist, otherwise continue training using the pretrained
            if not self.args.pretrained_model:
                self._generate_exp_directory()
            else:
                self.args.exp_dir = os.path.dirname(os.path.dirname(self.args.pretrained_model))
                self.args.ckpt_dir = os.path.join(self.args.exp_dir, "checkpoint")

            # logger
            self.args.writer = SummaryWriter(log_dir=self.args.exp_dir)
            # loss
            self.args.epoch = -1
            self.args.step = -1

        else:
            self.args.job_name = os.path.basename(args.pretrained_model).split('.')[0]
            self.args.exp_dir = os.path.dirname(args.pretrained_model)
            self.args.res_dir = os.path.join(self.args.exp_dir, 'result', self.args.job_name)
            pathlib.Path(self.args.res_dir).mkdir(parents=True, exist_ok=True)

            self.args.exp_dir = self.args.root_dir
            pathlib.Path(self.args.exp_dir).mkdir(parents=True, exist_ok=True)

        self._configure_logger()
        self._print_args()
        self._set_seed()

    def get_args(self):
        return self.args

    def _generate_exp_directory(self):
        """Creates checkpoint folder. We save
        model checkpoints using the provided model directory
        but we add a sub-folder for each separate experiment:
        """
        timestamp = time.strftime('%Y%m%d-%H%M%S')

        if not self.args.exp_name:
            self.args.exp_name = '{}-{}-{}-n{}-C{}-norm_{}-k{}-drop{}-lr{}-B{}-seed{}' \
                .format(os.path.basename(os.getcwd()),
                        self.args.block, self.args.conv, self.args.n_blocks, self.args.n_filters,
                        self.args.norm, self.args.k, self.args.dropout,  self.args.lr,
                        self.args.batch_size, self.args.seed)

        if not self.args.job_name:
            self.args.job_name = '_'.join([self.args.exp_name, timestamp, str(uuid.uuid4())])
        self.args.exp_dir = os.path.join(self.args.root_dir, self.args.job_name)
        self.args.ckpt_dir = os.path.join(self.args.exp_dir, "checkpoint")
        self.args.code_dir = os.path.join(self.args.exp_dir, "code")
        pathlib.Path(self.args.exp_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.args.ckpt_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.args.code_dir).mkdir(parents=True, exist_ok=True)
        # ===> save scripts
        scripts_to_save = glob.glob('*.py')
        if scripts_to_save is not None:
            for script in scripts_to_save:
                dst_file = os.path.join(self.args.code_dir, os.path.basename(script))
                shutil.copyfile(script, dst_file)

    def _configure_logger(self):
        """
        Configure logger on given level. Logging will occur on standard
        output and in a log file saved in model_dir.
        """
        self.args.loglevel = "info"
        numeric_level = getattr(logging, self.args.loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: {}'.format(self.args.loglevel))

        log_format = logging.Formatter('%(asctime)s %(message)s')
        logger = logging.getLogger()
        logger.setLevel(numeric_level)

        file_handler = logging.FileHandler(os.path.join(self.args.exp_dir,
                                                        '{}.log'.format(os.path.basename(self.args.job_name))))
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

        file_handler = logging.StreamHandler(sys.stdout)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        logging.root = logger
        logging.info("saving log, checkpoint and back up code in folder: {}".format(self.args.exp_dir))

    def _print_args(self):
        logging.info("==========       args      =============")
        for arg, content in self.args.__dict__.items():
            logging.info("{}:{}".format(arg, content))
        logging.info("==========     args END    =============")
        logging.info("\n")
        logging.info('===> Phase is {}.'.format(self.args.phase))

    def _set_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

