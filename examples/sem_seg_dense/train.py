import __init__
import numpy as np
import torch
import torch_geometric.datasets as GeoData
from torch_geometric.data import DenseDataLoader
import torch_geometric.transforms as T
from torch.nn import DataParallel
from config import OptInit
from architecture import DenseDeepGCN
from utils.ckpt_util import load_pretrained_models, load_pretrained_optimizer, save_checkpoint
from utils.metrics import AverageMeter
import logging
from tqdm import tqdm


def main():
    opt = OptInit().get_args()
    logging.info('===> Creating dataloader ...')
    train_dataset = GeoData.S3DIS(opt.data_dir, opt.area, True, pre_transform=T.NormalizeScale())
    train_loader = DenseDataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    test_dataset = GeoData.S3DIS(opt.data_dir, opt.area, train=False, pre_transform=T.NormalizeScale())
    test_loader = DenseDataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0)
    opt.n_classes = train_loader.dataset.num_classes

    logging.info('===> Loading the network ...')
    model = DenseDeepGCN(opt).to(opt.device)
    if opt.multi_gpus:
        model = DataParallel(DenseDeepGCN(opt)).to(opt.device)

    logging.info('===> loading pre-trained ...')
    model, opt.best_value, opt.epoch = load_pretrained_models(model, opt.pretrained_model, opt.phase)
    logging.info(model)

    logging.info('===> Init the optimizer ...')
    criterion = torch.nn.CrossEntropyLoss().to(opt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_adjust_freq, opt.lr_decay_rate)
    optimizer, scheduler, opt.lr = load_pretrained_optimizer(opt.pretrained_model, optimizer, scheduler, opt.lr)

    logging.info('===> Init Metric ...')
    opt.losses = AverageMeter()
    opt.test_value = 0.

    logging.info('===> start training ...')
    for _ in range(opt.epoch, opt.total_epochs):
        opt.epoch += 1
        logging.info('Epoch:{}'.format(opt.epoch))
        train(model, train_loader, optimizer, criterion, opt)
        if opt.epoch % opt.eval_freq == 0 and opt.eval_freq != -1:
            test(model, test_loader, opt)
        scheduler.step()

        # ------------------ save checkpoints
        # min or max. based on the metrics
        is_best = (opt.test_value < opt.best_value)
        opt.best_value = max(opt.test_value, opt.best_value)
        model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        save_checkpoint({
            'epoch': opt.epoch,
            'state_dict': model_cpu,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_value': opt.best_value,
        }, is_best, opt.ckpt_dir, opt.exp_name)

        # ------------------ tensorboard log
        info = {
            'loss': opt.losses.avg,
            'test_value': opt.test_value,
            'lr': scheduler.get_lr()[0]
        }
        opt.writer.add_scalars('epoch', info, opt.iter)

    logging.info('Saving the final model.Finish!')


def train(model, train_loader, optimizer, criterion, opt):
    opt.losses.reset()
    model.train()
    with tqdm(train_loader) as tqdm_loader:
        for i, data in enumerate(tqdm_loader):
            opt.iter += 1
            desc = 'Epoch:{}  Iter:{}  [{}/{}]  Loss:{Losses.avg: .4f}'\
                .format(opt.epoch, opt.iter, i + 1, len(train_loader), Losses=opt.losses)
            tqdm_loader.set_description(desc)

            if not opt.multi_gpus:
                data = data.to(opt.device)
            inputs = torch.cat((data.pos.transpose(2, 1).unsqueeze(3), data.x.transpose(2, 1).unsqueeze(3)), 1)
            gt = data.y.to(opt.device)
            # ------------------ zero, output, loss
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion(out, gt)

            # ------------------ optimization
            loss.backward()
            optimizer.step()

            opt.losses.update(loss.item())


def test(model, loader, opt):
    Is = np.empty((len(loader), opt.n_classes))
    Us = np.empty((len(loader), opt.n_classes))

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(loader)):
            if not opt.multi_gpus:
                data = data.to(opt.device)
            inputs = torch.cat((data.pos.transpose(2, 1).unsqueeze(3), data.x.transpose(2, 1).unsqueeze(3)), 1)
            gt = data.y

            out = model(inputs)
            pred = out.max(dim=1)[1]

            pred_np = pred.cpu().numpy()
            target_np = gt.cpu().numpy()

            for cl in range(opt.n_classes):
                cur_gt_mask = (target_np == cl)
                cur_pred_mask = (pred_np == cl)
                I = np.sum(np.logical_and(cur_pred_mask, cur_gt_mask), dtype=np.float32)
                U = np.sum(np.logical_or(cur_pred_mask, cur_gt_mask), dtype=np.float32)
                Is[i, cl] = I
                Us[i, cl] = U

    ious = np.divide(np.sum(Is, 0), np.sum(Us, 0))
    ious[np.isnan(ious)] = 1
    iou = np.mean(ious)
    if opt.phase == 'test':
        for cl in range(opt.n_classes):
            logging.info("===> mIOU for class {}: {}".format(cl, ious[cl]))

    opt.test_value = iou
    logging.info('TEST Epoch: [{}]\t mIoU: {:.4f}\t'.format(opt.epoch, opt.test_value))


if __name__ == '__main__':
    main()
