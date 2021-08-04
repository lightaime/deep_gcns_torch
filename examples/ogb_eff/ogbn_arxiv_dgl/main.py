#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import math
import os
import random
import time
import logging
import uuid
import sys
import gc
from functools import reduce
import operator as op

import dgl
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from loss import loss_kd_only

from model_rev import RevGAT

epsilon = 1 - math.log(2)

device = None

dataset = "ogbn-arxiv"
n_node_feats, n_classes = 0, 0


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


def load_data(dataset):
    global n_node_feats, n_classes

    data = DglNodePropPredDataset(name=dataset)
    evaluator = Evaluator(name=dataset)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]

    n_node_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()

    return graph, labels, train_idx, val_idx, test_idx, evaluator


def preprocess(graph):
    global n_node_feats

    # make bidirected
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    graph.create_formats_()

    return graph


def gen_model(args):
    if args.use_labels:
        n_node_feats_ = n_node_feats + n_classes
    else:
        n_node_feats_ = n_node_feats

    if args.backbone == "rev":
        model = RevGAT(
                      n_node_feats_,
                      n_classes,
                      n_hidden=args.n_hidden,
                      n_layers=args.n_layers,
                      n_heads=args.n_heads,
                      activation=F.relu,
                      dropout=args.dropout,
                      input_drop=args.input_drop,
                      attn_drop=args.attn_drop,
                      edge_drop=args.edge_drop,
                      use_attn_dst=not args.no_attn_dst,
                      use_symmetric_norm=args.use_norm)
    else:
        raise Exception("Unknown backnone")

    return model


def custom_loss_function(x, labels):
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)


def add_labels(feat, labels, idx):
    onehot = torch.zeros([feat.shape[0], n_classes], device=device)
    onehot[idx, labels[idx, 0]] = 1
    return torch.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def train(args, model, graph, labels, train_idx, val_idx, test_idx, optimizer,
          evaluator, mode='teacher', teacher_output=None):
    model.train()
    if mode == 'student':
        assert teacher_output != None

    alpha = args.alpha
    temp = args.temp

    feat = graph.ndata["feat"]

    if args.use_labels:
        mask = torch.rand(train_idx.shape) < args.mask_rate

        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]

        feat = add_labels(feat, labels, train_labels_idx)
    else:
        mask = torch.rand(train_idx.shape) < args.mask_rate

        train_pred_idx = train_idx[mask]

    optimizer.zero_grad()

    if args.n_label_iters > 0:
        with torch.no_grad():
            pred = model(graph, feat)
    else:
        pred = model(graph, feat)

    if args.n_label_iters > 0:
        unlabel_idx = torch.cat([train_pred_idx, val_idx, test_idx])
        for _ in range(args.n_label_iters):
            pred = pred.detach()
            torch.cuda.empty_cache()
            feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx], dim=-1)
            pred = model(graph, feat)

    if mode == 'teacher':
        loss = custom_loss_function(pred[train_pred_idx],
                                    labels[train_pred_idx])
    elif mode == 'student':
        loss_gt = custom_loss_function(pred[train_pred_idx],
                                       labels[train_pred_idx])
        loss_kd = loss_kd_only(pred, teacher_output, temp)
        loss = loss_gt * (1 - alpha) + loss_kd * alpha
    else:
        raise Exception('unkown mode')

    loss.backward()
    optimizer.step()

    return evaluator(pred[train_idx], labels[train_idx]), loss.item()


@torch.no_grad()
def evaluate(args, model, graph, labels, train_idx, val_idx, test_idx, evaluator):
    model.eval()

    feat = graph.ndata["feat"]

    if args.use_labels:
        feat = add_labels(feat, labels, train_idx)

    pred = model(graph, feat)

    if args.n_label_iters > 0:
        unlabel_idx = torch.cat([val_idx, test_idx])
        for _ in range(args.n_label_iters):
            feat[unlabel_idx, -n_classes:] = F.softmax(pred[unlabel_idx], dim=-1)
            pred = model(graph, feat)

    train_loss = custom_loss_function(pred[train_idx], labels[train_idx])
    val_loss = custom_loss_function(pred[val_idx], labels[val_idx])
    test_loss = custom_loss_function(pred[test_idx], labels[test_idx])

    return (
        evaluator(pred[train_idx], labels[train_idx]),
        evaluator(pred[val_idx], labels[val_idx]),
        evaluator(pred[test_idx], labels[test_idx]),
        train_loss,
        val_loss,
        test_loss,
        pred,
    )


def save_pred(pred, run_num, kd_dir):
    if not os.path.exists(kd_dir):
        os.makedirs(kd_dir)
    fname = os.path.join(kd_dir, 'best_pred_run{}.pt'.format(run_num))
    torch.save(pred.cpu(), fname)


def run(args, graph, labels, train_idx, val_idx, test_idx,
        evaluator, n_running):
    evaluator_wrapper = lambda pred, labels: evaluator.eval(
        {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels}
    )["acc"]

    # kd mode
    mode = args.mode

    # define model and optimizer
    model = gen_model(args).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # training loop
    total_time = 0
    best_val_acc, final_test_acc, best_val_loss = 0, 0, float("inf")
    final_pred = None

    accs, train_accs, val_accs, test_accs = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []

    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()
        if mode == 'student':
            teacher_output = torch.load('./{}/best_pred_run{}.pt'.format(
              args.kd_dir,
              n_running)).cpu().cuda()
        else:
            teacher_output = None

        adjust_learning_rate(optimizer, args.lr, epoch)

        acc, loss = train(args, model, graph, labels, train_idx,
                          val_idx, test_idx, optimizer, evaluator_wrapper,
                          mode=mode, teacher_output=teacher_output)

        train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, pred = evaluate(
            args, model, graph, labels, train_idx, val_idx, test_idx, evaluator_wrapper
        )

        toc = time.time()
        total_time += toc - tic

        if epoch == 1:
            peak_memuse = torch.cuda.max_memory_allocated(device) / float(1024 ** 3)
            logging.info('Peak memuse {:.2f} G'.format(peak_memuse))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            final_test_acc = test_acc
            final_pred = pred
            if mode == 'teacher':
                save_pred(final_pred, n_running, args.kd_dir)

        if epoch == args.n_epochs or epoch % args.log_every == 0:
            logging.info(
                f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f}\n"
                f"Loss: {loss:.4f}, Acc: {acc:.4f}\n"
                f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                f"Train/Val/Test/Best val/Final test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{final_test_acc:.4f}"
            )

        for l, e in zip(
            [accs, train_accs, val_accs, test_accs, losses, train_losses, val_losses, test_losses],
            [acc, train_acc, val_acc, test_acc, loss, train_loss, val_loss, test_loss],
        ):
            l.append(e)

    logging.info("*" * 50)
    logging.info(f"Best val acc: {best_val_acc}, Final test acc: {final_test_acc}")
    logging.info("*" * 50)

    # plot learning curves
    if args.plot_curves:
        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.n_epochs, 100))
        ax.set_yticks(np.linspace(0, 1.0, 101))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip([accs, train_accs, val_accs, test_accs], ["acc", "train acc", "val acc", "test acc"]):
            plt.plot(range(args.n_epochs), y, label=label, linewidth=1)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.01))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"gat_acc_{n_running}.png")

        fig = plt.figure(figsize=(24, 24))
        ax = fig.gca()
        ax.set_xticks(np.arange(0, args.n_epochs, 100))
        ax.tick_params(labeltop=True, labelright=True)
        for y, label in zip(
            [losses, train_losses, val_losses, test_losses], ["loss", "train loss", "val loss", "test loss"]
        ):
            plt.plot(range(args.n_epochs), y, label=label, linewidth=1)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        plt.grid(which="major", color="red", linestyle="dotted")
        plt.grid(which="minor", color="orange", linestyle="dotted")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"gat_loss_{n_running}.png")

    if args.save_pred:
        os.makedirs("./output", exist_ok=True)
        torch.save(F.softmax(final_pred, dim=1), f"./output/{n_running}.pt")

    return best_val_acc, final_test_acc


def count_parameters(args):
    model = gen_model(args)
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def main():
    global device, n_node_feats, n_classes, epsilon

    argparser = argparse.ArgumentParser(
        "GAT implementation on ogbn-arxiv", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--seed", type=int, default=0, help="seed")
    argparser.add_argument("--n-runs", type=int, default=10, help="running times")
    argparser.add_argument("--n-epochs", type=int, default=2000, help="number of epochs")
    argparser.add_argument(
        "--use-labels", action="store_true", help="Use labels in the training set as input features."
    )
    argparser.add_argument("--n-label-iters", type=int, default=0, help="number of label iterations")
    argparser.add_argument("--mask-rate", type=float, default=0.5, help="mask rate")
    argparser.add_argument("--no-attn-dst", action="store_true", help="Don't use attn_dst.")
    argparser.add_argument("--use-norm", action="store_true", help="Use symmetrically normalized adjacency matrix.")
    argparser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    argparser.add_argument("--n-layers", type=int, default=3, help="number of layers")
    argparser.add_argument("--n-heads", type=int, default=3, help="number of heads")
    argparser.add_argument("--n-hidden", type=int, default=250, help="number of hidden units")
    argparser.add_argument("--dropout", type=float, default=0.75, help="dropout rate")
    argparser.add_argument("--input-drop", type=float, default=0.1, help="input drop rate")
    argparser.add_argument("--attn-drop", type=float, default=0.0, help="attention drop rate")
    argparser.add_argument("--edge-drop", type=float, default=0.0, help="edge drop rate")
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument("--log-every", type=int, default=20, help="log every LOG_EVERY epochs")
    argparser.add_argument("--plot-curves", action="store_true", help="plot learning curves")
    argparser.add_argument("--save-pred", action="store_true", help="save final predictions")
    argparser.add_argument("--save", type=str, default='exp', help="save exp")
    argparser.add_argument('--backbone', type=str, default='rev',
                           help='gcn backbone [deepergcn, wt, deq, rev, gr]')
    argparser.add_argument('--group', type=int, default=2,
                           help='num of groups for rev gnns')
    argparser.add_argument("--kd_dir", type=str, default='./kd', help="kd path for pred")
    argparser.add_argument("--mode", type=str, default='teacher', help="kd mode [teacher, student]")
    argparser.add_argument("--alpha",type=float, default=0.5, help="ratio of kd loss")
    argparser.add_argument("--temp",type=float, default=1.0, help="temperature of kd")
    args = argparser.parse_args()

    args.save = '{}-L{}-DP{}-H{}'.format(args.save, args.n_layers, args.dropout, args.n_hidden)
    args.save = 'log/{}-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"), str(uuid.uuid4()))
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        format=log_format,
                        datefmt='%m/%d %I:%M:%S %p')
    logging.getLogger().setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if not args.use_labels and args.n_label_iters > 0:
        raise ValueError("'--use-labels' must be enabled when n_label_iters > 0")

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    # load data & preprocess
    graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset)
    graph = preprocess(graph)

    graph, labels, train_idx, val_idx, test_idx = map(
        lambda x: x.to(device), (graph, labels, train_idx, val_idx, test_idx)
    )

    logging.info(args)
    logging.info(f"Number of params: {count_parameters(args)}")

    # run
    val_accs, test_accs = [], []

    for i in range(args.n_runs):
        seed(args.seed + i)
        val_acc, test_acc = run(args, graph, labels, train_idx, val_idx,
                                test_idx, evaluator, i + 1)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    logging.info(args)
    logging.info(f"Runned {args.n_runs} times")
    logging.info("Val Accs:")
    logging.info(val_accs)
    logging.info("Test Accs:")
    logging.info(test_accs)
    logging.info(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
    logging.info(f"Average test accuracy: {np.mean(test_accs)} ± {np.std(test_accs)}")
    logging.info(f"Number of params: {count_parameters(args)}")

if __name__ == "__main__":
    main()

#  ******************************Teacher******************************
#  Namespace(alpha=0.5, attn_drop=0.0, backbone='rev', cpu=False, dropout=0.75, edge_drop=0.3, gpu=0, group=2, input_drop=0.25, kd_dir='./kd', log_every=20, lr=0.002, mask_rate=0.5, mode='teacher', n_epochs=2000, n_heads=3, n_hidden=256, n_label_iters=1, n_layers=5, n_runs=10, no_attn_dst=True, plot_curves=False, save='log/kd-L5-DP0.75-H256-20210620-044728-d9034b17-88b2-45fb-bfd2-bdc7d6de3313', save_pred=False, seed=0, temp=1.0, use_labels=True, use_norm=True, wd=0)
#  Runned 10 times
#  Val Accs:
#  [0.7505620993993087, 0.7485150508406322, 0.7487835162253766, 0.7498909359374476, 0.7514681700728212, 0.7500922849760059, 0.7511661465149837, 0.7490184234370281, 0.7503943085338434, 0.7509312393033323]
#  Test Accs:
#  [0.7388844310022015, 0.736826944838796, 0.741291689813386, 0.7382877600148139, 0.7432463016686213, 0.7416003127378968, 0.7406950188259984, 0.7411270909203136, 0.7407978931341687, 0.7394811019895892]
#  Average val accuracy: 0.750082217524078 ± 0.000972382448137496
#  Average test accuracy: 0.7402238544945785 ± 0.0017655209523257862
#  Number of params: 2098256

#  ******************************Student******************************
# Namespace(alpha=0.95, attn_drop=0.0, backbone='rev', cpu=False, dropout=0.75, edge_drop=0.3, gpu=0, group=2, input_drop=0.25, kd_dir='./kd', log_every=20, lr=0.002, mask_rate=0.5, mode='student', n_epochs=2000, n_heads=3, n_hidden=256, n_label_iters=1, n_layers=5, n_runs=10, no_attn_dst=True, plot_curves=False, save='log/kd-L5-DP0.75-H256-20210621-001327-5cb604b1-36de-46ac-9d20-3614702ece43', save_pred=False, seed=0, temp=0.7, use_labels=True, use_norm=True, wd=0)
# Runned 10 times
# Val Accs:
# [0.7487499580522836, 0.748951307090842, 0.7502936340145643, 0.750159401322192, 0.7500251686298198, 0.7499916104567267, 0.7487835162253766, 0.7487835162253766, 0.750696332091681, 0.7508976811302392]
# Test Accs:
# [0.7400366232537087, 0.7421764088636504, 0.7433697508384256, 0.7431845770837191, 0.7456947102030739, 0.7432668765302554, 0.7399131740839043, 0.7410859411970454, 0.7438635475176429, 0.7438223977943749]
# Average val accuracy: 0.7497332125239102 ± 0.0007945637178043291
# Average test accuracy: 0.74264140073658 ± 0.0017404484511541998
# Number of params: 2098256
