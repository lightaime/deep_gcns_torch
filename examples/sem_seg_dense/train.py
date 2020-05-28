import __init__
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


def main():
    opt = OptInit().get_args()
    logging.info('===> Creating dataloader ...')
    train_dataset = GeoData.S3DIS(opt.data_dir, 5, True, pre_transform=T.NormalizeScale())
    train_loader = DenseDataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
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
    # opt.test_metric = miou
    # opt.test_values = AverageMeter()
    opt.test_value = 0.

    logging.info('===> start training ...')
    for _ in range(opt.epoch, opt.total_epochs):
        opt.epoch += 1
        train(model, train_loader, optimizer, scheduler, criterion, opt)
        # test_value = test(model, test_loader, test_metric, opt)
        scheduler.step()
    logging.info('Saving the final model.Finish!')


def train(model, train_loader, optimizer, scheduler, criterion, opt):
    model.train()
    for i, data in enumerate(train_loader):
        opt.iter += 1
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
        # ------------------ show information
        if opt.iter % opt.print_freq == 0:
            logging.info('Epoch:{}\t Iter: {}\t [{}/{}]\t Loss: {Losses.avg: .4f}'.format(
                opt.epoch, opt.iter, i + 1, len(train_loader), Losses=opt.losses))
            opt.losses.reset()

        # ------------------ tensor board log
        info = {
            'loss': loss,
            'test_value': opt.test_value,
            'lr': scheduler.get_lr()[0]
        }
        for tag, value in info.items():
            opt.logger.scalar_summary(tag, value, opt.iter)

    # ------------------ save checkpoints
    # min or max. based on the metrics
    is_best = (opt.test_value < opt.best_value)
    opt.best_value = min(opt.test_value, opt.best_value)

    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    save_checkpoint({
        'epoch': opt.epoch,
        'state_dict': model_cpu,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_value': opt.best_value,
    }, is_best, opt.save_path, opt.post)


def test(model, test_loader, test_metric, opt):
    opt.test_values.reset()
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if not opt.multi_gpus:
                data = data.to(opt.device)
            inputs = torch.cat((data.pos.transpose(2, 1).unsqueeze(3), data.x.transpose(2, 1).unsqueeze(3)), 1)
            gt = data.y.to(opt.device)

            out = opt.model(inputs)
            test_value = test_metric(out.max(dim=1)[1], gt, opt.n_classes)
            opt.test_values.update(test_value, opt.batch_size)
        logging.info('Epoch: [{0}]\t Iter: [{1}]\t''TEST loss: {test_values.avg: .4f})\t'.format(
               opt.epoch, opt.iter, test_values=opt.test_values))

    opt.test_value = opt.test_values.avg


if __name__ == '__main__':
    main()


