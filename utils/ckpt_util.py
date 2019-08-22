import os
import torch
import shutil


def load_pretrained_models(opt):
    if opt.pretrained_model:
        if os.path.isfile(opt.pretrained_model):
            print("===> loading checkpoint '{}'".format(opt.pretrained_model))
            checkpoint = torch.load(opt.pretrained_model)
            best_loss = checkpoint['best_loss']
            model_dict = opt.model.state_dict()
            model_dict.update(checkpoint['state_dict'])
            opt.model.load_state_dict(model_dict)
            print("The pretrained_model is at checkpoint {}.".format(checkpoint['epoch']))
            if best_loss != 0:
                print("Its best loss is {}.".format(best_loss))
            if opt.train:
                opt.epoch = checkpoint['epoch']
        else:
            print("===> no checkpoint found at '{}'".format(opt.pretrained_model))
            opt.best_loss = 0
    else:
        opt.best_loss = 0


def load_pretrained_optimizer(opt):
    if opt.pretrained_model:
        if os.path.isfile(opt.pretrained_model):
            checkpoint = torch.load(opt.pretrained_model)
            if 'optimizer_state_dict' in checkpoint.keys():
                opt.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for state in opt.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
            if 'scheduler_state_dict' in checkpoint.keys():
                opt.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                opt.lr = opt.scheduler.get_lr()[0]


def save_checkpoint(state, is_best, save_path, postname):
    filename = '{}/{}_ckpt_{}.pth'.format(save_path, postname, int(state['epoch']))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}/{}_model_best.pth'.format(save_path, postname))

