import os
import sys
import torch
import torch_geometric.datasets as GeoData
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn.data_parallel import DataParallel
from sklearn.metrics import f1_score

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
from opt import OptInit
from architecture import DeepGCN
from utils.ckpt_util import load_pretrained_models, load_pretrained_optimizer, save_checkpoint
from utils.metrics import AverageMeter
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train():
    info_format = 'Epoch: [{}]\t loss: {: .6f} train mF1: {: .6f} \t val mF1: {: .6f}\t test mF1: {:.6f} \t ' \
                  'best val mF1: {: .6f}\t best test mF1: {:.6f}'
    opt.printer.info('===> Init the optimizer ...')
    criterion = torch.nn.BCEWithLogitsLoss().to(opt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    scheduler = ReduceLROnPlateau(optimizer, "min", patience=opt.lr_patience, verbose=True, factor=0.5, cooldown=30,
                                  min_lr=opt.lr/100)
    opt.scheduler = 'ReduceLROnPlateau'

    optimizer, scheduler, opt.lr = load_pretrained_optimizer(opt.pretrained_model, optimizer, scheduler, opt.lr)

    opt.printer.info('===> Init Metric ...')
    opt.losses = AverageMeter()

    best_val_value = 0.
    best_test_value = 0.

    opt.printer.info('===> Start training ...')
    for _ in range(opt.epoch, opt.total_epochs):
        opt.epoch += 1
        loss, train_value = train_step(model, train_loader, optimizer, criterion, opt)
        val_value = test(model, valid_loader, opt)
        test_value = test(model, test_loader, opt)

        if val_value > best_val_value:
            best_val_value = val_value
            save_ckpt(model, optimizer, scheduler, opt.epoch, opt.save_path, opt.post, name_post='val_best')
        if test_value > best_test_value:
            best_test_value = test_value
            save_ckpt(model, optimizer, scheduler, opt.epoch, opt.save_path, opt.post, name_post='test_best')

        opt.printer.info(info_format.format(opt.epoch, loss, train_value, val_value, test_value, best_val_value,
                                            best_test_value))

        if opt.scheduler == 'ReduceLROnPlateau':
            scheduler.step(opt.losses.avg)
        else:
            scheduler.step()

    opt.printer.info('Saving the final model.Finish!')


def train_step(model, train_loader, optimizer, criterion, opt):
    model.train()
    micro_f1 = 0.
    count = 0.
    opt.losses.reset()
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

        micro_f1 += f1_score(gt.cpu().detach().numpy(),
                             (out > 0).cpu().detach().numpy(), average='micro') * len(gt)
        count += len(gt)
        # ------------------ optimization
        loss.backward()
        optimizer.step()

        opt.losses.update(loss.item())
    return opt.losses.avg, micro_f1/count


def test(model, loader, opt):
    model.eval()
    count = 0
    micro_f1 = 0.
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(opt.device)
            out = model(data)

            num_node = len(data.x)
            micro_f1 += f1_score(data.y.cpu().detach().numpy(),
                                 (out > 0).cpu().detach().numpy(), average='micro') * num_node
            count += num_node
        micro_f1 = float(micro_f1)/count
    return micro_f1


def save_ckpt(model, optimizer, scheduler, epoch, save_path, name_pre, name_post='best'):
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    state = {
            'epoch': epoch,
            'state_dict': model_cpu,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
    filename = '{}/{}_{}.pth'.format(save_path, name_pre, name_post)
    torch.save(state, filename)


if __name__ == '__main__':
    opt = OptInit().initialize()
    opt.printer.info('===> Creating dataloader ...')
    test_dataset = GeoData.PPI(opt.data_dir, split='test')
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True)
    opt.n_classes = test_loader.dataset.num_classes

    opt.printer.info('===> Loading the network ...')
    model = DeepGCN(opt).to(opt.device)
    if opt.multi_gpus:
        model = DataParallel(DeepGCN(opt)).to(opt.device)
    opt.printer.info('===> loading pre-trained ...')
    model, opt.best_value, opt.epoch = load_pretrained_models(model, opt.pretrained_model, opt.phase)

    if opt.phase == 'train':
        train_dataset = GeoData.PPI(opt.data_dir, 'train')
        if opt.multi_gpus:
            train_loader = DataListLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
        valid_dataset = GeoData.PPI(opt.data_dir, split='val')
        valid_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=True)
        train()

        # load best model on validation dataset
        opt.printer.info('\n\n=================Below is best model testing=====================')
        # opt.printer.info('Loading best model on validation dataset')
        best_model_path = '{}/{}_val_best.pth'.format(opt.save_path, opt.post)
        model, opt.best_value, opt.epoch = load_pretrained_models(model, best_model_path, opt.phase)
        test_value = test(model, test_loader, opt)
        opt.printer.info('Test m-F1 of model on validation dataset: {: 6f}'.format(test_value))

        # load best model on test_dataset
        # opt.printer.info('\nLoading best model on test dataset')
        best_model_path = '{}/{}_test_best.pth'.format(opt.save_path, opt.post)
        model, opt.best_value, opt.epoch = load_pretrained_models(model, best_model_path, opt.phase)
        test_value = test(model, test_loader, opt)
        opt.printer.info('Test m-F1 of model on test dataset: {: 6f}'.format(test_value))

    else:
        test_value = test(model, test_loader, opt)
        opt.printer.info('Test m-F1: {: 6f}'.format(test_value))




