import os
import sys
import torch
import torch_geometric.datasets as GeoData
from torch_geometric.data import DataLoader, DataListLoader
import torch_geometric.transforms as T
from torch_geometric.utils import mean_iou as miou
from torch_geometric.nn.data_parallel import DataParallel
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from utils.opt import OptInit
from utils.ckpt_util import load_pretrained_models, load_pretrained_optimizer, save_checkpoint
import models.architecture as models
from utils.metrics import AverageMeter
from utils import optim


def main():
    opt = OptInit().initialize()
    print('===> Creating dataloader ...')
    train_dataset = GeoData.S3DIS(opt.train_path, 5, True, pre_transform=T.NormalizeScale())
    if opt.multi_gpus:
        train_loader = DataListLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    else:
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    opt.n_classes = train_loader.dataset.num_classes

    print('===> Loading the network ...')
    model = getattr(models, opt.model_name)(opt).to(opt.device)
    if opt.multi_gpus:
        model = DataParallel(getattr(models, opt.model_name)(opt)).to(opt.device)
    print('===> loading pre-trained ...')
    model, opt.best_value, opt.epoch = load_pretrained_models(model, opt.pretrained_model, opt.phase)

    print('===> Init the optimizer ...')
    criterion = torch.nn.CrossEntropyLoss().to(opt.device)
    if opt.optim.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optim.lower() == 'radam':
        optimizer = optim.RAdam(model.parameters(), lr=opt.lr)
    else:
        raise NotImplementedError('opt.optim is not supported')
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_adjust_freq, opt.lr_decay_rate)
    optimizer, scheduler, opt.lr = load_pretrained_optimizer(opt.pretrained_model, optimizer, scheduler, opt.lr)

    print('===> Init Metric ...')
    opt.losses = AverageMeter()
    # opt.valid_metric = miou
    # opt.valid_values = AverageMeter()
    opt.valid_value = 0.

    print('===> start training ...')
    for _ in range(opt.total_epochs):
        opt.epoch += 1
        train(model, train_loader, optimizer, scheduler, criterion, opt)
        # valid_value = valid(model, valid_loader, valid_metric, opt)
        scheduler.step()
    print('Saving the final model.Finish!')


def train(model, train_loader, optimizer, scheduler, criterion, opt):
    model.train()
    for i, data in enumerate(train_loader):
        opt.iter += 1
        if not opt.multi_gpus:
            data = data.to(opt.device)
            gt = data.y
        else:
            gt = torch.cat([data_batch.y for data_batch in data], 0).to(opt.device)

        # ------------------ zero, output, loss
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, gt)

        # ------------------ optimization
        loss.backward()
        optimizer.step()

        opt.losses.update(loss.item())
        # ------------------ show information
        if opt.iter % opt.print_freq == 0:
            print('Epoch:{}\t Iter: {}\t [{}/{}]\t Loss: {Losses.avg: .4f}'.format(
                opt.epoch, opt.iter, i + 1, len(train_loader), Losses=opt.losses))
            opt.losses.reset()

        # ------------------ tensor board log
        info = {
            'loss': loss,
            'valid_value': opt.valid_value,
            'lr': scheduler.get_lr()[0]
        }
        for tag, value in info.items():
            opt.logger.scalar_summary(tag, value, opt.iter)

    # ------------------ save checkpoints
    # min or max. based on the metrics
    is_best = (opt.valid_value < opt.best_value)
    opt.best_value = min(opt.valid_value, opt.best_value)

    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    # optim_cpu = {k: v.cpu() for k, v in optimizer.state_dict().items()}
    save_checkpoint({
        'epoch': opt.epoch,
        'state_dict': model_cpu,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_value': opt.best_value,
    }, is_best, opt.save_path, opt.post)


def valid(model, valid_loader, valid_metric, opt):
    opt.valid_values.reset()
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            if not opt.multi_gpus:
                data = data.to(opt.device)
                gt = data.y
            else:
                gt = torch.cat([data_batch.y for data_batch in data], 0).to(opt.device)

            out = opt.model(data)
            valid_value = valid_metric(out.max(dim=1)[1], gt, opt.n_classes)
            opt.valid_values.update(valid_value, opt.batch_size)
        print('Epoch: [{0}]\t Iter: [{1}]\t''TEST loss: {valid_values.avg: .4f})\t'.format(
               opt.epoch, opt.iter, valid_values=opt.valid_values))

    opt.valid_value = opt.valid_values.avg


if __name__ == '__main__':
    main()


