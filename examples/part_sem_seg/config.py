import os
import time
import argparse
import random
import numpy as np
import torch
import logging
import logging.config
from utils.tf_logger import TfLogger
import uuid
import sys
import pathlib
import glob
import shutil


category_names = ['Bag', 'Bed', 'Bottle', 'Bowl', 'Chair', 'Clock', 'Dishwasher', 'Display', 'Door', 'Earphone',  # 0-9
                  'Faucet', 'Hat', 'Keyboard', 'Knife', 'Lamp', 'Laptop', 'Microwave', 'Mug', 'Refrigerator',  # 10-18
                  'Scissors', 'StorageFurniture', 'Table', 'TrashCan', 'Vase']  # 19-23
# We only experiment level3. The category no is  1, 2, 4, 5, 6, 7, 8,9 ; 10, 13, 14, 16, 18 ;  20, 21, 22, 23


class OptInit:
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch implementation of Deep GCN')

        # base
        parser.add_argument('--phase', default='test', type=str, help='train or test(default)')
        parser.add_argument('--use_cpu', action='store_true', help='use cpu?')

        # dataset args
        parser.add_argument('--data_dir', type=str, default='/data/deepgcn/partnet')
        parser.add_argument('--dataset', type=str, default='sem_seg_h5')
        parser.add_argument('--category', type=int, default=1, help="please check label_to_names in PartNet Class")
        parser.add_argument('--level', type=int, default=3, help="segmentation level")
        parser.add_argument('--batch_size', default=6, type=int, help='mini-batch size (default:7)')
        parser.add_argument('--test_batch_size', default=10, type=int, help='test mini-batch size (default:12)')
        parser.add_argument('--in_channels', default=3, type=int, help='the channel size of input point cloud ')
        parser.add_argument('--data_augment', action='store_true',
                            help='Default: False. Do not use Data Augmentation')

        # train args
        parser.add_argument('--total_epochs', default=500, type=int, help='number of total epochs to run')
        parser.add_argument('--iter', default=-1, type=int, help='number of iteration to start')
        parser.add_argument('--lr_adjust_freq', default=50, type=int, help='decay lr after certain number of epoch')
        parser.add_argument('--lr', default=5e-3, type=float, help='initial learning rate')
        parser.add_argument('--lr_decay_rate', default=0.9, type=float, help='learning rate decay')
        parser.add_argument('--multi_gpus', action='store_true', help='use multi-gpus')
        parser.add_argument('--seed', default=3, type=int, help='seed')

        # model args
        parser.add_argument('--pretrained_model', type=str, help='path to pretrained model(default: none)', default='')
        parser.add_argument('--k', default=9, type=int, help='neighbor num (default:9)')
        parser.add_argument('--block', default='res', type=str, help='graph backbone block type {res, plain, dense}')
        parser.add_argument('--conv', default='edge', type=str, help='graph conv layer {edge, mr}')
        parser.add_argument('--act', default='relu', type=str, help='activation layer {relu, prelu, leakyrelu}')
        parser.add_argument('--norm', default='batch', type=str,
                            help='batch or instance normalization {batch, instance}')

        parser.add_argument('--bias', default=True, type=bool, help='bias of conv layer True or False')
        parser.add_argument('--n_filters', default=64, type=int, help='number of channels of deep features')
        parser.add_argument('--n_blocks', default=28, type=int, help='number of basic blocks in the backbone')
        parser.add_argument('--dropout', default=0.5, type=float, help='ratio of dropout')
        # dilated knn
        parser.add_argument('--use_dilation', default=True, type=bool, help='use dilated knn or not')
        parser.add_argument('--epsilon', default=0.2, type=float, help='stochastic epsilon for gcn')
        parser.add_argument('--stochastic', default=True, type=bool, help='stochastic for gcn, True or False')

        # saving
        parser.add_argument('--exp_name', type=str, default='PartnetSemanticSeg', help='post name of saved file')
        parser.add_argument('--root_dir', type=str, default='log', help='the dir of experiment results')
        args = parser.parse_args()

        args.category_no = args.category
        args.category = category_names[args.category]
        args.device = torch.device('cuda' if not args.use_cpu and torch.cuda.is_available() else 'cpu')
        self.args = args

        # ===> generate log dir
        if self.args.phase == 'train':
            self._generate_exp_directory()
            # logger
            self.args.logger = TfLogger(self.args.exp_dir)
            # loss
            self.args.epoch = -1
            self.args.step = -1

        else:
            self.args.exp_dir = os.path.dirname(args.pretrained_model)
            self.args.res_dir = os.path.join(self.args.exp_dir, 'result', args.block, args.category)
            pathlib.Path(self.args.res_dir).mkdir(parents=True, exist_ok=True)

        self._configure_logger()
        self._print_args()
        self._set_seed(self.args.seed)

    def _get_args(self):
        return self.args

    def _generate_exp_directory(self):
        """
        Helper function to create checkpoint folder. We save
        model checkpoints using the provided model directory
        but we add a sub-folder for each separate experiment:
        """
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        self.args.jobname = '{}-{}-{}-L{}-{}-{}-n{}-C{}-k{}-drop{}-lr{}_B{}' \
            .format(self.args.exp_name, self.args.category_no, self.args.category, self.args.level,
                    self.args.block, self.args.conv, self.args.n_blocks, self.args.n_filters,
                    self.args.k, self.args.dropout,  self.args.lr, self.args.batch_size)

        experiment_string = '_'.join([self.args.jobname, timestamp, str(uuid.uuid4())])
        self.args.exp_dir = os.path.join(self.args.root_dir, experiment_string)
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

    def _print_args(self):
        logging.info("==========       args      =============")
        for arg, content in self.args.__dict__.items():
            logging.info("{}:{}".format(arg, content))
        logging.info("==========     args END    =============")
        logging.info("\n")
        logging.info('===> Phase is {}.'.format(self.args.phase))

    def _configure_logger(self):
        """
        Configure logger on given level. Logging will occur on standard
        output and in a log file saved in model_dir.
        """
        self.args.loglevel = "info"
        numeric_level = getattr(logging, self.args.loglevel.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: {}'.format(self.args.loglevelloglevel))

            # configure logger to display and save log data
        # log_format = logging.Formatter('%(asctime)s [%(levelname)-5.5s] [%(filename)s:%(lineno)04d] %(message)s')
        log_format = logging.Formatter('%(asctime)s %(message)s')
        logger = logging.getLogger()
        logger.setLevel(numeric_level)

        file_handler = logging.FileHandler(os.path.join(self.args.exp_dir,
                                                        '{}.log'.format(os.path.basename(self.args.exp_dir))))
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

        file_handler = logging.StreamHandler(sys.stdout)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        logging.root = logger
        logging.info("saving log, checkpoint and back up code in folder: {}".format(self.args.exp_dir))

    def _set_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



