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
from opt import OptInit
from architecture import DenseDeepGCN
from utils.ckpt_util import load_pretrained_models


def main():
    opt = OptInit().initialize()

    print('===> Creating dataloader...')
    test_dataset = GeoData.S3DIS(opt.test_path, 5, False, pre_transform=T.NormalizeScale())
    test_loader = DenseDataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)
    opt.n_classes = test_loader.dataset.num_classes
    if opt.no_clutter:
        opt.n_classes -= 1

    print('===> Loading the network ...')
    model = DenseDeepGCN(opt).to(opt.device)
    model, opt.best_value, opt.epoch = load_pretrained_models(model, opt.pretrained_model, opt.phase)

    print('===> Start Evaluation ...')
    test(model, test_loader, opt)


def test(model, loader, opt):
    Is = np.empty((len(loader), opt.n_classes))
    Us = np.empty((len(loader), opt.n_classes))

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            data = data.to(opt.device)
            inputs = torch.cat((data.pos.transpose(2, 1).unsqueeze(3), data.x.transpose(2, 1).unsqueeze(3)), 1)
            gt = data.y

            out = model(inputs)
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


