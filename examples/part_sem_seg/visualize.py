import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
import os.path as osp
from utils.pc_viz import visualize_part_seg
import argparse


clss = ['Bag', 'Bed', 'Bottle', 'Bowl', 'Chair', 'Clock', 'Dishwasher', 'Display', 'Door', 'Earphone',  # 0-9
        'Faucet', 'Hat', 'Keyboard', 'Knife', 'Lamp', 'Laptop', 'Microwave', 'Mug', 'Refrigerator', 'Scissors',  # 10-19
        'StorageFurniture', 'Table', 'TrashCan', 'Vase']  # 20-23

parser = argparse.ArgumentParser(description='PyTorch implementation of Deep GCN')
parser.add_argument('--category', type=int, default=1)
parser.add_argument('--obj_no', default=0, type=int, help='NO. of which obj in a given category to visualize')
parser.add_argument('--folders', default='res,plain', type=str,
                    help='use "," to separate different folders, eg. "res,plain"')
args = parser.parse_args()

category = clss[args.category]
obj_no = args.obj_no
folders = list(map(lambda x: x.strip(), args.folders.split(',')))

dir_path = osp.join(os.path.dirname(os.path.abspath(__file__)), 'result')
folder_paths = list(map(lambda x: osp.join(dir_path, x, category), folders))

file_name_pred = '_'.join([category, str(obj_no), 'pred.obj'])
file_name_gt = '_'.join([category, str(obj_no), 'gt.obj'])

visualize_part_seg(file_name_pred,
                   file_name_gt,
                   folder_paths,
                   limit=-1,
                   text=['Ground Truth', 'ResGCN-28'],
                   interactive=True,
                   orientation='horizontal')
