import os
import torch
import shutil


def load_pretrained_models(model, pretrained_model, phase):
    best_value = 10000000.
    epoch = -1

    if pretrained_model:
        if os.path.isfile(pretrained_model):
            print("===> loading checkpoint '{}'".format(pretrained_model))
            checkpoint = torch.load(pretrained_model)
            best_value = checkpoint['best_value']
            model_dict = model.state_dict()
            model_dict.update(checkpoint['state_dict'])
            model.load_state_dict(model_dict)
            print("The pretrained_model is at checkpoint {}.".format(checkpoint['epoch']))
            if best_value != 0:
                print("Its best valid value is {}.".format(best_value))
            if phase == 'train':
                epoch = checkpoint['epoch']
            else:
                epoch = -1
        else:
            print("===> no checkpoint found at '{}'".format(pretrained_model))

    return model, best_value, epoch


def load_pretrained_optimizer(pretrained_model, optimizer, scheduler, lr):
    if pretrained_model:
        if os.path.isfile(pretrained_model):
            checkpoint = torch.load(pretrained_model)
            if 'optimizer_state_dict' in checkpoint.keys():
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()
            if 'scheduler_state_dict' in checkpoint.keys():
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                lr = scheduler.get_lr()[0]

    return optimizer, scheduler, lr


def save_checkpoint(state, is_best, save_path, postname):
    filename = '{}/{}_ckpt_{}.pth'.format(save_path, postname, int(state['epoch']))
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}/{}_model_best.pth'.format(save_path, postname))


def change_ckpt_dict(model, optimizer, scheduler, opt):

    # TODO:
    for _ in range(opt.epoch):
        scheduler.step()
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

