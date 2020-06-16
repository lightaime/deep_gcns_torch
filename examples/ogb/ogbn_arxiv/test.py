import torch
from torch_geometric.utils import to_undirected, add_self_loops
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.nodeproppred import Evaluator
from args import ArgsInit
from model import DeeperGCN


@torch.no_grad()
def test(model, x, edge_index, y_true, split_idx, evaluator):
    model.eval()
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


def main():

    args = ArgsInit().args

    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')

    dataset = PygNodePropPredDataset(name=args.dataset)
    data = dataset[0]
    split_idx = dataset.get_idx_split()

    evaluator = Evaluator(args.dataset)

    x = data.x.to(device)
    y_true = data.y.to(device)

    edge_index = data.edge_index.to(device)
    edge_index = to_undirected(edge_index, data.num_nodes)

    if args.self_loop:
        edge_index = add_self_loops(edge_index, num_nodes=data.num_nodes)[0]

    args.in_channels = data.x.size(-1)
    args.num_tasks = dataset.num_classes

    print(args)

    model = DeeperGCN(args)

    model.load_state_dict(torch.load(args.model_load_path)['model_state_dict'])
    model.to(device)

    result = test(model, x, edge_index, y_true, split_idx, evaluator)
    train_accuracy, valid_accuracy, test_accuracy = result

    print({'Train': train_accuracy,
           'Validation': valid_accuracy,
           'Test': test_accuracy})

    model.print_params(final=True)


if __name__ == "__main__":
    main()
