import __init__
from ogb.nodeproppred import Evaluator
import torch
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from utils.data_util import intersection, random_partition_graph, generate_sub_graphs
from args import ArgsInit
from ogb.nodeproppred import PygNodePropPredDataset
from model import DeeperGCN
import numpy as np
from utils.ckpt_util import save_ckpt
import logging
import statistics
import time


@torch.no_grad()
def test(model, x, edge_index, y_true, split_idx, evaluator):
    # test on CPU
    model.eval()
    model.to('cpu')
    out = model(x, edge_index)

    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc


def train(data, model, x, y_true, train_idx, optimizer, device):
    loss_list = []
    model.train()

    sg_nodes, sg_edges = data
    train_y = y_true[train_idx].squeeze(1)

    idx_clusters = np.arange(len(sg_nodes))
    np.random.shuffle(idx_clusters)

    for idx in idx_clusters:

        x_ = x[sg_nodes[idx]].to(device)
        sg_edges_ = sg_edges[idx].to(device)
        mapper = {node: idx for idx, node in enumerate(sg_nodes[idx])}

        inter_idx = intersection(sg_nodes[idx], train_idx)
        training_idx = [mapper[t_idx] for t_idx in inter_idx]

        optimizer.zero_grad()

        pred = model(x_, sg_edges_)
        target = train_y[inter_idx].to(device)

        loss = F.nll_loss(pred[training_idx], target)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

    return statistics.mean(loss_list)


def main():

    args = ArgsInit().save_exp()

    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')

    dataset = PygNodePropPredDataset(name=args.dataset)
    graph = dataset[0]

    adj = SparseTensor(row=graph.edge_index[0],
                       col=graph.edge_index[1])

    if args.self_loop:
        adj = adj.set_diag()
        graph.edge_index = add_self_loops(edge_index=graph.edge_index,
                                          num_nodes=graph.num_nodes)[0]
    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"].tolist()

    evaluator = Evaluator(args.dataset)

    sub_dir = 'random-train_{}-full_batch_test'.format(args.cluster_number)
    logging.info(sub_dir)

    args.in_channels = graph.x.size(-1)
    args.num_tasks = dataset.num_classes

    logging.info('%s' % args)

    model = DeeperGCN(args).to(device)

    logging.info(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    results = {'highest_valid': 0,
               'final_train': 0,
               'final_test': 0,
               'highest_train': 0}

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # generate batches
        parts = random_partition_graph(graph.num_nodes,
                                       cluster_number=args.cluster_number)
        data = generate_sub_graphs(adj, parts, cluster_number=args.cluster_number)

        epoch_loss = train(data, model, graph.x, graph.y, train_idx, optimizer, device)
        logging.info('Epoch {}, training loss {:.4f}'.format(epoch, epoch_loss))
        model.print_params(epoch=epoch)

        if epoch == args.epochs:

            result = test(model, graph.x, graph.edge_index, graph.y, split_idx, evaluator)
            logging.info(result)

            train_accuracy, valid_accuracy, test_accuracy = result

            if train_accuracy > results['highest_train']:
                results['highest_train'] = train_accuracy

            if valid_accuracy > results['highest_valid']:
                results['highest_valid'] = valid_accuracy
                results['final_train'] = train_accuracy
                results['final_test'] = test_accuracy

                save_ckpt(model, optimizer,
                          round(epoch_loss, 4), epoch,
                          args.model_save_path,
                          sub_dir, name_post='valid_best')

    logging.info("%s" % results)

    end_time = time.time()
    total_time = end_time - start_time
    logging.info('Total time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time))))


if __name__ == "__main__":
    main()
