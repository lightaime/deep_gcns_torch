import __init__
from tqdm import tqdm
import numpy as np
import torch
import torch_geometric.datasets as GeoData
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
from config import OptInit
from architecture import SparseDeepGCN
from utils.ckpt_util import load_pretrained_models
import logging


def main():
    opt = OptInit().get_args()

    logging.info('===> Creating dataloader...')
    test_dataset = GeoData.S3DIS(opt.data_dir, 5, train=False, pre_transform=T.NormalizeScale())
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)
    opt.n_classes = test_loader.dataset.num_classes
    if opt.no_clutter:
        opt.n_classes -= 1

    logging.info('===> Loading the network ...')
    model = SparseDeepGCN(opt).to(opt.device)
    model, opt.best_value, opt.epoch = load_pretrained_models(model, opt.pretrained_model, opt.phase)

    logging.info('===> Start Evaluation ...')
    test(model, test_loader, opt)


def test(model, loader, opt):
    Is = np.empty((len(loader), opt.n_classes))
    Us = np.empty((len(loader), opt.n_classes))

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            data = data.to(opt.device)
            out = model(data)
            pred = out.max(dim=1)[1]

            pred_np = pred.cpu().numpy()
            target_np = data.y.cpu().numpy()

            for cl in range(opt.n_classes):
                I = np.sum(np.logical_and(pred_np == cl, target_np == cl))
                U = np.sum(np.logical_or(pred_np == cl, target_np == cl))
                Is[i, cl] = I
                Us[i, cl] = U

    ious = np.divide(np.sum(Is, 0), np.sum(Us, 0))
    ious[np.isnan(ious)] = 1
    for cl in range(opt.n_classes):
        logging.info("===> mIOU for class {}: {}".format(cl, ious[cl]))
    logging.info("===> mIOU is {}".format(np.mean(ious)))


if __name__ == '__main__':
    main()


