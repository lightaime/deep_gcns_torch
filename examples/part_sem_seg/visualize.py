import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
import os.path as osp
from utils.pc_viz import visualize_part_seg
import argparse


category_names = ['Bag', 'Bed', 'Bottle', 'Bowl', 'Chair', 'Clock', 'Dishwasher', 'Display', 'Door', 'Earphone',  # 0-9
        'Faucet', 'Hat', 'Keyboard', 'Knife', 'Lamp', 'Laptop', 'Microwave', 'Mug', 'Refrigerator', 'Scissors',  # 10-19
        'StorageFurniture', 'Table', 'TrashCan', 'Vase']  # 20-23

parser = argparse.ArgumentParser(description='Qualitative comparision of ResGCN '
                                             'against PlainGCN on PartNet segmentation')

# dir_path set to the location of the result folder.
# result folder should have such structure:
# result
# ├── plain  # result folder for PlainGCN
#      ├── Bed # the obj director of category Bed
#            ├── Bed_0_pred.obj
# ├── res  # result folder for ResGCN
#      ├── Bed # the obj director of category Bed
#            ├── Bed_0_pred.obj

parser.add_argument('--category', type=int, default=4)
parser.add_argument('--obj_no', default=0, type=int, help='NO. of which obj in a given category to visualize')
parser.add_argument('--dir_path', default='../result', type=str, help='path to the result')
parser.add_argument('--folders', default='plain,res', type=str,
                    help='use "," to separate different folders, eg. "res,plain"')
args = parser.parse_args()

category = category_names[args.category]
obj_no = args.obj_no
folders = list(map(lambda x: x.strip(), args.folders.split(',')))

folder_paths = list(map(lambda x: osp.join(args.dir_path, x, category), folders))

file_name_pred = '_'.join([category, str(obj_no), 'pred.obj'])
file_name_gt = '_'.join([category, str(obj_no), 'gt.obj'])

# show Ground Truth, PlainGCN, ResGCN
visualize_part_seg(file_name_pred,
                   file_name_gt,
                   folder_paths,
                   limit=-1,
                   text=['Ground Truth', 'PlainGCN-28', 'ResGCN-28'],
                   interactive=True,
                   orientation='horizontal')
