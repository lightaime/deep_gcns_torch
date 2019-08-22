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
    opt.model = getattr(models, opt.model_name)(opt).to(opt.device)
    if opt.multi_gpus:
        opt.model = DataParallel(getattr(models, opt.model_name)(opt)).to(opt.device)
    print('===> loading pre-trained ...')
    load_pretrained_models(opt)

    print('===> Init the optimizer ...')
    opt.criterion = torch.nn.CrossEntropyLoss().to(opt.device)
    opt.valid_metric = miou
    opt.optimizer = torch.optim.Adam(opt.model.parameters(), lr=opt.lr)
    opt.scheduler = torch.optim.lr_scheduler.StepLR(opt.optimizer, opt.lr_adjust_freq, 0.5)
    load_pretrained_optimizer(opt)

    print('===> start training ...')
    for _ in range(opt.total_epochs):
        opt.epoch += 1
        train(train_loader, opt)
        # valid(train_loader, opt)
        opt.scheduler.step()
    print('Saving the final model.Finish!')


def train(train_loader, opt):
    opt.model.train()
    for i, data in enumerate(train_loader):
        opt.iter += 1
        if not opt.multi_gpus:
            data = data.to(opt.device)
            gt = data.y
        else:
            gt = torch.cat([data_batch.y for data_batch in data], 0).to(opt.device)
        # ------------------ zero, output, loss
        opt.optimizer.zero_grad()
        out = opt.model(data)
        loss = opt.criterion(out, gt)

        # ------------------ optimization
        loss.backward()
        opt.optimizer.step()

        opt.losses.update(loss.item())
        # ------------------ show information
        if opt.iter % opt.print_freq == 0:
            print('Epoch:{}\t Iter: {}\t [{}/{}]\t {Losses.avg: .4f}'.format(
                opt.epoch, opt.iter, i + 1, len(train_loader), Losses=opt.losses))
            opt.losses.reset()

        # ------------------ tensor board log
        info = {
            'loss': loss,
            'valid_value': opt.valid_value,
            'lr': opt.scheduler.get_lr()[0]
        }
        for tag, value in info.items():
            opt.logger.scalar_summary(tag, value, opt.iter)

    # ------------------ save checkpoints
    is_best = (opt.valid_value < opt.best_loss)
    # min or max. based on the metrics
    opt.best_loss = min(opt.valid_value, opt.best_loss)
    model_cpu = {k: v.cpu() for k, v in opt.model.state_dict().items()}
    optim_cpu = {k: v.cpu() for k, v in opt.optimizer.state_dict().items()}
    save_checkpoint({
        'epoch': opt.epoch,
        'state_dict': model_cpu,
        'optimizer_state_dict': optim_cpu,
        'scheduler_state_dict': opt.scheduler.state_dict(),
        'best_loss': opt.best_loss,
    }, is_best, opt.save_path, opt.post)


def valid(valid_loader, opt):
    opt.valid_values.reset()
    opt.model.eval()
    with torch.no_grad():
        for i, data in enumerate(valid_loader):
            if i > 10:
                break
            if not opt.multi_gpus:
                data = data.to(opt.device)
                gt = data.y
            else:
                gt = torch.cat([data_batch.y for data_batch in data], 0).to(opt.device)

            out = opt.model(data)
            valid_value = opt.valid_metric(out.max(dim=1)[1], gt, opt.n_classes)
            opt.valid_values.update(valid_value, opt.batch_size)
        print('Epoch: [{0}]\t Iter: [{1}]\t''TEST loss: {valid_values.avg: .4f})\t'.format(
               opt.epoch, opt.iter, valid_values=opt.valid_values))

    opt.valid_value = opt.valid_values.avg


if __name__ == '__main__':
    main()


