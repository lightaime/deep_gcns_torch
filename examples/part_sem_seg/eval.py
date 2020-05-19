import os
import __init__
import numpy as np
from tqdm import tqdm
import os.path as osp
import torch
from torch_geometric.data import DenseDataLoader
from config import OptInit
from architecture import DeepGCN
from utils.ckpt_util import load_pretrained_models
from utils.data_util import PartNet
import logging

g_class2color = {
                    '0': [255,  0,  0],
                    '1': [255,  255,  0],
                    '2': [0, 255, 0],
                    '3': [0, 255, 255],
                    '4': [0, 0, 255],
                    '5': [255, 0, 255],

                    '6': [255,  102,  102],
                    '7': [255,  255,  102],
                    '8': [102, 255, 102],
                    '9': [102, 255, 255],
                    '10': [102, 102, 255],
                    '11': [255, 102, 255],

                    '12': [255,  153,  153],
                    '13': [255,  255,  153],
                    '14': [153, 255, 153],
                    '15': [153, 255, 255],
                    '16': [153, 153, 255],
                    '17': [255, 153, 255],

                    '18': [153,  0,  0],
                    '19': [153,  153,  0],
                    '20': [0, 153, 0],
                    '21': [0, 153, 153],
                    '22': [0, 0, 153],
                    '23': [153, 0, 153],


                    '24': [102,  0,  0],
                    '25': [102,  102,  0],
                    '26': [0, 102, 0],
                    '27': [0, 102, 102],
                    '28': [0, 0, 102],
                    '29': [102, 0, 102],

                    '30': [255,  178,  102],
                    '31': [178,  255,  102],
                    '32': [102, 255, 178],
                    '33': [102, 178, 255],
                    '34': [178, 102, 255],
                    '35': [255, 102, 178],

                    '36': [153, 76, 0],
                    '37': [76, 153, 0],
                    '38': [0, 153, 76],
                    '39': [0, 76, 153],
                    '40': [76, 0, 153],
                    '41': [153, 0, 76],

                    '42': [255, 204, 153],
                    '43': [204, 255, 153],
                    '44': [153, 255, 204],
                    '45': [153, 204, 255],
                    '46': [204, 153, 255],
                    '47': [255, 153, 204],

                    '48': [255, 153, 51],
                    '49': [153, 255, 51],
                    '50': [51, 255, 153],
                    '51': [51, 153, 255],
                    '52': [153, 51, 255],
                    '53': [255, 51, 153]}


def test(model, loader, opt):
    save_path = opt.res_dir

    part_intersect = np.zeros(opt.n_classes, dtype=np.float32)
    part_union = np.zeros(opt.n_classes, dtype=np.float32)

    model.eval()
    shape_iou_tot = 0.
    shape_iou_cnt = 0.
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            # open files for output
            fout = open(osp.join(save_path, opt.category + '_' + str(i) + '_pred.obj'), 'w')
            fout_gt = open(osp.join(save_path, opt.category + '_' + str(i) + '_gt.obj'), 'w')

            # load data
            data = data.to(opt.device)
            inputs = data.pos.transpose(2, 1).unsqueeze(3)
            gt = data.y
            out = model(inputs.detach())
            pred = out.max(dim=1)[1]

            pos = data.pos.transpose(2, 1).squeeze(0).cpu().numpy()
            pred_np = pred.cpu().squeeze(0).numpy()
            target_np = gt.cpu().squeeze(0).numpy()

            for i in range(len(pred_np)):
                cls_pred = str(pred_np[i])
                cls_gt = str(target_np[i])
                color_pred = g_class2color[cls_pred]
                color_gt = g_class2color[cls_gt]

                fout.write('v %f %f %f %d %d %d\n' % (pos[0, i], pos[1, i], pos[2, i], color_pred[0], color_pred[1], color_pred[2]))
                fout_gt.write('v %f %f %f %d %d %d\n' % (pos[0, i], pos[1, i], pos[2, i], color_gt[0], color_gt[1], color_gt[2]))

            cur_shape_iou_tot = 0.0
            cur_shape_iou_cnt = 0

            for cl in range(opt.n_classes):
                cur_gt_mask = (target_np == cl)
                cur_pred_mask = (pred_np == cl)

                I = np.sum(np.logical_and(cur_pred_mask, cur_gt_mask), dtype=np.float32)
                U = np.sum(np.logical_or(cur_pred_mask, cur_gt_mask), dtype=np.float32)

                if U > 0:
                    part_intersect[cl] += I
                    part_union[cl] += U

                    cur_shape_iou_tot += I/U
                    cur_shape_iou_cnt += 1.

            if cur_shape_iou_cnt > 0:
                cur_shape_miou = cur_shape_iou_tot / cur_shape_iou_cnt
                shape_iou_tot += cur_shape_miou
                shape_iou_cnt += 1.

    shape_mIoU = shape_iou_tot / shape_iou_cnt
    part_iou = np.divide(part_intersect[1:], part_union[1:])
    mean_part_iou = np.mean(part_iou)
    logging.info("===> Finish Testing! Category {}-{}, Part mIOU is {:.4f} \n\n\n ".format(
                      opt.category_no, opt.category, mean_part_iou))


if __name__ == '__main__':
    opt = OptInit()._get_args()
    logging.info('===> Creating dataloader ...')
    test_dataset = PartNet(opt.data_dir, 'sem_seg_h5', opt.category, opt.level, 'test')
    test_loader = DenseDataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)
    opt.n_classes = test_loader.dataset.num_classes

    logging.info('===> Loading the network ...')
    model = DeepGCN(opt).to(opt.device)
    logging.info('===> loading pre-trained ...')
    model, opt.best_value, opt.epoch = load_pretrained_models(model, opt.pretrained_model, opt.phase)

    test(model, test_loader, opt)


