import viz
import os.path as osp

# category = 'Bed'
# test_no = 2
category = 'Dishwasher'
test_no = 1

folder_path = 'results/'
folder = [osp.join('results/nores/', category), osp.join('results/res/', category)]

file_name_pred = '_'.join([category, str(test_no), 'pred.obj'])
file_name_gt = '_'.join([category, str(test_no), 'gt.obj'])

viz.visualize_part_seg(file_name_pred,
                       file_name_gt,
                       folder,
                       limit=-1,
                       text=['Ground Truth', 'PlainGCN-28', 'ResGCN-28'],
                       interactive=True,
                       orientation='horizontal')
