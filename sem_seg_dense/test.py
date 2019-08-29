import os
import sys
from tqdm import tqdm
import numpy as np
import torch
import torch_geometric.datasets as GeoData
from torch_geometric.data import DenseDataLoader
import torch_geometric.transforms as T

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from utils.opt import OptInit
from utils.ckpt_util import load_pretrained_models
import models.architecture as models


def main():
    opt = OptInit().initialize()

    print('===> Creating dataloader...')
    test_dataset = GeoData.S3DIS(opt.test_path, 5, False, pre_transform=T.NormalizeScale())
    test_loader = DenseDataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)
    opt.n_classes = test_loader.dataset.num_classes
    if opt.no_clutter:
        opt.n_classes -= 1

    print('===> Loading the network ...')
    model = getattr(models, opt.model_name)(opt).to(opt.device)
    model, opt.best_value, opt.epoch = load_pretrained_models(model, opt.pretrained_model, opt.phase)

    print('===> Start Evaluation ...')
    test(opt.model, test_loader, opt)


def test(model, loader, opt):
    Is = np.empty((len(loader), opt.n_classes))
    Us = np.empty((len(loader), opt.n_classes))

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            data = data.to(opt.device)
            gt = data.y
            out = model(data)
            pred = out.max(dim=1)[1]

            pred_np = pred.cpu().numpy()
            target_np = gt.cpu().numpy()

            for cl in range(opt.n_classes):
                I = np.sum(np.logical_and(pred_np == cl, target_np == cl))
                U = np.sum(np.logical_or(pred_np == cl, target_np == cl))
                Is[i, cl] = I
                Us[i, cl] = U

    ious = np.divide(np.sum(Is, 0), np.sum(Us, 0))
    ious[np.isnan(ious)] = 1
    for cl in range(opt.n_classes):
        print("===> mIOU for class {}: {}".format(cl, ious[cl]))
    print("===> mIOU is {}".format(np.mean(ious)))


if __name__ == '__main__':
    main()


# from TorchTools.DataTools import indoor3d_util
# TODO: visulazation
# if opt.visu:
#     fout = open('_pred.obj'), 'w')
#     fout_gt = open(os.path.join(DUMP_DIR, os.path.basename(room_path)[:-4]+'_gt.obj'), 'w')
# fout_data_label = open(out_data_label_filename, 'w')
# fout_gt_label = open(out_gt_label_filename, 'w')
#
#
# data.x[0:3] *= 255.
#
# for i in range(len(pred)):
#     color = indoor3d_util.g_label2color[pred[i]]
#     color_gt = indoor3d_util.g_label2color[gt[i]]
#     if opt.visu:
#         fout.write('v %f %f %f %d %d %d\n' % (data.x[i, 3], data.x[i, 4], data.x[i, 5], color[i, 0], color[i, 1], color[i, 2]))
#         fout_gt.write(
#             'v %f %f %f %d %d %d\n' % (data.x[i, 3], data.x[i, 4], data.x[i, 5], color_gt[i, 0], color_gt[i, 1], color_gt[i, 2]))
#     fout_data_label.write('%f %f %f %d %d %d %f %d\n' % (
#     data.x[i, 3], data.x[i, 4], data.x[i, 5], data.x[i, 0], data.x[i, 1], data.x[i, 2], out[i, pred], pred))
#     fout_gt_label.write('%d\n' % (gt[i]))


