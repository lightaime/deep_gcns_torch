import __init__
import numpy as np
from tqdm import tqdm
import logging

import torch
from torch import nn
from torch_geometric.data import DenseDataLoader
from config import OptInit
from architecture import DeepGCN
from utils.ckpt_util import load_pretrained_models, load_pretrained_optimizer
from utils.metrics import AverageMeter
from data import scale_translate_pointcloud, PartNet


def train(model, train_loader, val_loader, test_loader, opt):
    logging.info('===> Init the optimizer ...')
    criterion = nn.NLLLoss().to(opt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)  # weight_decay=1e-4
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_adjust_freq, opt.lr_decay_rate)
    optimizer, scheduler, opt.lr = load_pretrained_optimizer(opt.pretrained_model, optimizer, scheduler, opt.lr)
    logging.info('===> Init Metric ...')
    opt.losses = AverageMeter()

    best_val_part_miou = 0.
    best_test_part_miou = 0.
    test_part_miou_val_best = 0.

    logging.info('===> start training ...')
    for _ in range(opt.epoch, opt.total_epochs):
        opt.epoch += 1
        # reset tracker
        opt.losses.reset()

        train_epoch(model, train_loader, optimizer, criterion, opt)
        val_part_iou, val_shape_mIoU = test(model, val_loader, opt)
        test_part_iou, test_shape_mIoU = test(model, test_loader, opt)

        scheduler.step()

        # ------------------  save ckpt
        if val_part_iou > best_val_part_miou:
            best_val_part_miou = val_part_iou
            test_part_miou_val_best = test_part_iou
            logging.info("Got a new best model on Validation with Part iou {:.4f}".format(best_val_part_miou))
            save_ckpt(model, optimizer, scheduler, opt, 'val_best')
        if test_part_iou > best_test_part_miou:
            best_test_part_miou = test_part_iou
            logging.info("Got a new best model on Test with Part iou {:.4f}".format(best_test_part_miou))
            save_ckpt(model, optimizer, scheduler, opt, 'test_best')

        # ------------------ show information
        logging.info(
            "===> Epoch {} Category {}-{}, Train Loss {:.4f}, mIoU on val {:.4f}, mIoU on test {:4f}, "
            "Best val mIoU {:.4f} Its test mIoU {:.4f}. Best test mIoU {:.4f}".format(
                opt.epoch, opt.category_no, opt.category, opt.losses.avg, val_part_iou, test_part_iou,
                best_val_part_miou, test_part_miou_val_best, best_test_part_miou))

        info = {
            'loss': opt.losses.avg,
            'val_part_miou': val_part_iou,
            'test_part_miou': test_part_iou,
            'lr': scheduler.get_lr()[0]
        }
        for tag, value in info.items():
            opt.writer.scalar_summary(tag, value, opt.step)

    save_ckpt(model, optimizer, scheduler, opt, 'last')
    logging.info(
        'Saving the final model.Finish! Category {}-{}. Best val part mIoU is {:.4f}. Its test mIoU is {:.4f}. '
        'Best test part mIoU is {:.4f}. Last test mIoU {:.4f} \n\n\n'.
            format(opt.category_no, opt.category, best_val_part_miou, test_part_miou_val_best,
                   best_test_part_miou, test_part_iou))


def train_epoch(model, train_loader, optimizer, criterion, opt):
    model.train()
    for i, data in enumerate(tqdm(train_loader, desc='[{}/{}] {} '.format(opt.epoch + 1, opt.total_epochs, 'train'))):
        opt.step += 1
        inputs = data.pos.transpose(2, 1).unsqueeze(3)
        if opt.data_augment:
            inputs = scale_translate_pointcloud(inputs)

        gt = data.y.to(opt.device)
        if gt.max() > opt.n_classes - 1:  # avoid some useless label
            gt = gt.clamp(0., opt.n_classes - 1)

        inputs = inputs.to(opt.device)
        del data
        # ------------------ zero, output, loss
        optimizer.zero_grad()
        out = model(inputs)
        loss = criterion(out, gt)

        # ------------------ optimization
        loss.backward()
        optimizer.step()

        opt.losses.update(loss.item())


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

    shape_mIoU = shape_iou_tot / shape_iou_cnt
    part_iou = np.divide(part_intersect[1:], part_union[1:])
    mean_part_iou = np.nanmean(part_iou)
    return mean_part_iou, shape_mIoU


def save_ckpt(model, optimizer, scheduler, opt, name_post):
    # ------------------ save ckpt
    filename = '{}/{}_model.pth'.format(opt.ckpt_dir, opt.jobname + '-' + name_post)
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    state = {
        'epoch': opt.epoch,
        'state_dict': model_cpu,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_value': opt.best_value,
    }
    torch.save(state, filename)
    logging.info('save a new best model into {}'.format(filename))


if __name__ == '__main__':
    opt = OptInit()._get_args()
    logging.info('===> Creating dataloader ...')

    train_dataset = PartNet(opt.data_dir, 'sem_seg_h5', opt.category, opt.level, 'train')
    train_loader = DenseDataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8)

    test_dataset = PartNet(opt.data_dir, 'sem_seg_h5', opt.category, opt.level, 'test')
    test_loader = DenseDataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=8)

    val_dataset = PartNet(opt.data_dir, 'sem_seg_h5', opt.category, opt.level, 'val')
    val_loader = DenseDataLoader(val_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=8)

    opt.n_classes = train_dataset.num_classes
    logging.info('===> Loading PartNet Category {}-{}, Semantic Segmentation level {}. '
                 'Has classes {}'.format(opt.category_no, opt.category, opt.level, opt.n_classes))

    logging.info('===> Loading the network ...')
    model = DeepGCN(opt).to(opt.device)
    if opt.multi_gpus:
        model = nn.DataParallel(model).to(opt.device)
    logging.info('===> loading pre-trained ...')
    model, opt.best_value, opt.epoch = load_pretrained_models(model, opt.pretrained_model, opt.phase)

    if opt.phase == 'train':
        train(model, train_loader, val_loader, test_loader, opt)

    else:
        mean_part_iou, shape_mIoU = test(model, test_loader, opt)
        logging.info(
            'Finish Testing! Category {}-{} Part mIoU is {:.4f} Shape mIoU is {:.4f}\n\n\n'.
                format(opt.category_no, opt.category, mean_part_iou, shape_mIoU))
