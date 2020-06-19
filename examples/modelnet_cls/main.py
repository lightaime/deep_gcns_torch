import __init__
import numpy as np
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from config import OptInit
from architecture import DeepGCN

import sklearn.metrics as metrics
from utils.ckpt_util import load_pretrained_models, load_pretrained_optimizer
from utils.metrics import AverageMeter
from utils.loss import SmoothCrossEntropy
from data import ModelNet40


def train(model, train_loader, test_loader, opt):
    logging.info('===> Init the optimizer ...')
    criterion = SmoothCrossEntropy()
    if opt.use_sgd:
        logging.info("===> Use SGD")
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr * 100, momentum=0.9, weight_decay=opt.weight_decay)
    else:
        logging.info("===> Use Adam")
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min=opt.lr)
    optimizer, scheduler, opt.lr = load_pretrained_optimizer(opt.pretrained_model, optimizer, scheduler, opt.lr)

    logging.info('===> Init Metric ...')
    opt.train_losses = AverageMeter()
    opt.test_losses = AverageMeter()
    best_test_overall_acc = 0.
    avg_acc_when_best = 0.

    logging.info('===> start training ...')
    for _ in range(opt.epoch, opt.epochs):
        opt.epoch += 1
        # reset tracker
        opt.train_losses.reset()
        opt.test_losses.reset()

        train_overall_acc, train_class_acc, opt = train_step(model, train_loader, optimizer, criterion, opt)
        test_overall_acc, test_class_acc, opt = infer(model, test_loader, criterion, opt)

        scheduler.step()

        # ------------------  save ckpt
        if test_overall_acc > best_test_overall_acc:
            best_test_overall_acc = test_overall_acc
            avg_acc_when_best = test_class_acc
            logging.info("Got a new best model on Test with Overall ACC {:.4f}. "
                         "Its avg acc is {:.4f}".format(best_test_overall_acc, avg_acc_when_best))
            save_ckpt(model, optimizer, scheduler, opt, 'best')

        # ------------------ show information
        logging.info(
            "===> Epoch {}/{}, Train Loss {:.4f}, Test Overall Acc {:.4f}, Test Avg Acc {:4f}, "
            "Best Test Overall Acc {:.4f}, Its test avg acc {:.4f}.".format(
                opt.epoch, opt.epochs, opt.train_losses.avg, test_overall_acc,
                test_class_acc, best_test_overall_acc, avg_acc_when_best))

        info = {
            'train_loss': opt.train_losses.avg,
            'train_OA': train_overall_acc,
            'train_avg_acc': train_class_acc,
            'test_loss': opt.test_losses.avg,
            'test_OA': test_overall_acc,
            'test_avg_acc': test_class_acc,
            'lr': scheduler.get_lr()[0]
        }
        for tag, value in info.items():
            opt.logger.scalar_summary(tag, value, opt.step)

    save_ckpt(model, optimizer, scheduler, opt, 'last')
    logging.info(
        'Saving the final model.Finish! Best Test Overall Acc {:.4f}, Its test avg acc {:.4f}. '
        'Last Test Overall Acc {:.4f}, Its test avg acc {:.4f}.'.
            format(best_test_overall_acc, avg_acc_when_best,
                   test_overall_acc, test_class_acc))


def train_step(model, train_loader, optimizer, criterion, opt):
    model.train()

    train_pred = []
    train_true = []
    for data, label in train_loader:
        data, label = data.to(opt.device), label.to(opt.device).squeeze()
        data = data.permute(0, 2, 1).unsqueeze(-1)

        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        opt.train_losses.update(loss.item())

        preds = logits.max(dim=1)[1]
        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())

    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    overall_acc = metrics.accuracy_score(train_true, train_pred)
    class_acc = metrics.balanced_accuracy_score(train_true, train_pred)
    return overall_acc, class_acc, opt


def infer(model, test_loader, criterion, opt):
    model.eval()
    test_true = []
    test_pred = []
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data, label = data.to(opt.device), label.to(opt.device).squeeze()
            data = data.permute(0, 2, 1).unsqueeze(-1)

            logits = model(data)
            loss = criterion(logits, label.squeeze())

            pred = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(pred.detach().cpu().numpy())

            opt.test_losses.update(loss.item())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        overall_acc = metrics.accuracy_score(test_true, test_pred)
        class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    return overall_acc, class_acc, opt


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

    train_loader = DataLoader(ModelNet40(data_dir=opt.data_dir, partition='train', num_points=opt.num_points),
                              num_workers=8, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(data_dir=opt.data_dir, partition='test', num_points=opt.num_points),
                             num_workers=8, batch_size=opt.test_batch_size, shuffle=True, drop_last=False)

    opt.n_classes = train_loader.dataset.num_classes()

    logging.info('===> Loading ModelNet40 from {}. number of classes equal to {}'.format(opt.data_dir, opt.n_classes))

    logging.info('===> Loading the network ...')
    model = DeepGCN(opt).to(opt.device)
    if opt.multi_gpus:
        model = nn.DataParallel(model).to(opt.device)
    logging.info('===> loading pre-trained ...')
    model, opt.best_value, opt.epoch = load_pretrained_models(model, opt.pretrained_model, opt.phase)

    if opt.phase == 'train':
        train(model, train_loader, test_loader, opt)

    else:
        criterion = SmoothCrossEntropy()
        opt.test_losses = AverageMeter()
        test_overall_acc, test_class_acc, opt = infer(model, test_loader, criterion, opt)
        logging.info(
            'Test Overall Acc {:.4f}, Its test avg acc {:.4f}.'.
                format(test_overall_acc, test_class_acc))
