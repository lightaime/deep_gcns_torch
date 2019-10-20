import os
import sys
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch_geometric.data import DenseDataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
from opt import OptInit
from architecture import DenseDeepGCN
from utils.ckpt_util import load_pretrained_models, load_pretrained_optimizer, save_checkpoint
from utils.metrics import AverageMeter
from utils import random_points_augmentation
from utils.data_util import PartNet


def train(model, train_loader, test_loader, opt):
    opt.printer.info('===> Init the optimizer ...')
    criterion = nn.NLLLoss().to(opt.device)
    # criterion = torch.nn.CrossEntropyLoss().to(opt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_adjust_freq, opt.lr_decay_rate)
    optimizer, scheduler, opt.lr = load_pretrained_optimizer(opt.pretrained_model, optimizer, scheduler, opt.lr)
    opt.printer.info('===> Init Metric ...')
    opt.losses = AverageMeter()
    opt.test_value = 0.

    opt.printer.info('===> start training ...')
    for _ in range(opt.epoch, opt.total_epochs):
        opt.epoch += 1
        train_step(model, train_loader, optimizer, scheduler, criterion, opt)
        if opt.epoch % opt.test_freq:
            test(model, test_loader, opt)
        save_ckpt(model, optimizer, scheduler, opt)
        scheduler.step()
    opt.printer.info(
        'Saving the final model.Finish! Category {}-{}. Best part mIou is {}. Best shape mIOU is {}.'.
            format(opt.category_no, opt.category, opt.best_value, opt.best_shapeMiou))


def train_step(model, train_loader, optimizer, scheduler, criterion, opt):
    model.train()
    for i, data in enumerate(train_loader):
        opt.iter += 1
        inputs = data.pos.transpose(2, 1).unsqueeze(3)
        if opt.data_augmentation:
            inputs = random_points_augmentation(inputs, rotate=True, translate=True, mean=0, std=0.02)
        gt = data.y.to(opt.device)
        inputs = inputs.to(opt.device)
        del data
        # ------------------ zero, output, loss
        optimizer.zero_grad()
        out = model(inputs)
        if gt.max() > opt.n_classes - 1:
            gt = gt.clamp(0., opt.n_classes - 1)
        loss = criterion(out, gt)

        # ------------------ optimization
        loss.backward()
        optimizer.step()

        opt.losses.update(loss.item())

    # ------------------ show information
    opt.printer.info('Epoch:{}\t Iter:[{}/{}]\t Loss: {Losses.avg: .4f}'.format(
        opt.epoch, opt.iter, len(train_loader), Losses=opt.losses))
    info = {
        'loss': loss,
        'test_value': opt.test_value,
        'lr': scheduler.get_lr()[0]
    }
    for tag, value in info.items():
        opt.logger.scalar_summary(tag, value, opt.iter)


def test(model, loader, opt):
    part_intersect = np.zeros(opt.n_classes, dtype=np.float32)
    part_union = np.zeros(opt.n_classes, dtype=np.float32)
    model.eval()

    shape_iou_tot = 0.
    shape_iou_cnt = 0.
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            data = data.to(opt.device)
            inputs = data.pos.transpose(2, 1).unsqueeze(3)
            gt = data.y

            out = model(inputs.detach())
            pred = out.max(dim=1)[1]

            pred_np = pred.cpu().numpy()
            target_np = gt.cpu().numpy()

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

                    cur_shape_iou_tot += I / U
                    cur_shape_iou_cnt += 1.

            if cur_shape_iou_cnt > 0:
                cur_shape_miou = cur_shape_iou_tot / cur_shape_iou_cnt
                shape_iou_tot += cur_shape_miou
                shape_iou_cnt += 1.

    opt.shape_mIoU = shape_iou_tot / shape_iou_cnt
    if opt.shape_mIoU > opt.best_shapeMiou:
        opt.best_shapeMiou = opt.shape_mIoU
    part_iou = np.divide(part_intersect[1:], part_union[1:])
    mean_part_iou = np.mean(part_iou)
    opt.test_value = mean_part_iou
    opt.printer.info(
        "===> Category {}-{}, Part mIOU is{:.4f} \t Shape mIoU is{:.4f} \t best Part mIOU is {:.4f}\t".format(
            opt.category_no, opt.category, opt.test_value, opt.shape_mIoU, opt.best_value))


def save_ckpt(model, optimizer, scheduler, opt):
    # ------------------ save ckpt
    is_best = (opt.test_value > opt.best_value)
    if opt.save_best_only:
        save_flag = is_best
    else:
        save_flag = True
    if save_flag:
        opt.best_value = max(opt.test_value, opt.best_value)
        model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        save_checkpoint({
            'epoch': opt.epoch,
            'state_dict': model_cpu,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_value': opt.best_value,
        }, is_best, opt.save_path, opt.post)
        opt.losses.reset()
        opt.test_value = 0.


if __name__ == '__main__':
    opt = OptInit().initialize()
    opt.printer.info('===> Creating dataloader ...')
    test_dataset = PartNet(opt.data_dir, opt.dataset, opt.category, opt.level, 'val')
    test_loader = DenseDataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=True, num_workers=4)
    opt.n_classes = test_loader.dataset.num_classes

    opt.best_shapeMiou = 0

    opt.printer.info('===> Loading the network ...')
    model = DenseDeepGCN(opt).to(opt.device)
    if opt.multi_gpus:
        model = nn.DataParallel(model).to(opt.device)
    opt.printer.info('===> loading pre-trained ...')
    model, opt.best_value, opt.epoch = load_pretrained_models(model, opt.pretrained_model, opt.phase)

    if opt.phase == 'train':
        train_dataset = PartNet(opt.data_dir, opt.dataset, opt.category, opt.level, 'train')
        train_loader = DenseDataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8)
        opt.n_classes = test_loader.dataset.num_classes
        train(model, train_loader, test_loader, opt)

    else:
        test(model, test_loader, opt)
