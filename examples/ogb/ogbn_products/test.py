import __init__
from ogb.nodeproppred import Evaluator
import torch
from torch_geometric.utils import add_self_loops
from args import ArgsInit
from ogb.nodeproppred import PygNodePropPredDataset
from model import DeeperGCN


@torch.no_grad()
def test(model, x, edge_index, y_true, split_idx, evaluator):
    # test on CPU
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

    dataset = PygNodePropPredDataset(name=args.dataset)
    graph = dataset[0]

    if args.self_loop:
        graph.edge_index = add_self_loops(edge_index=graph.edge_index,
                                          num_nodes=graph.num_nodes)[0]
    split_idx = dataset.get_idx_split()

    evaluator = Evaluator(args.dataset)

    args.in_channels = graph.x.size(-1)
    args.num_tasks = dataset.num_classes

    print(args)

    model = DeeperGCN(args)

    print(model)

    model.load_state_dict(torch.load(args.model_load_path)['model_state_dict'])
    result = test(model, graph.x, graph.edge_index, graph.y, split_idx, evaluator)
    print(result)
    model.print_params(final=True)


if __name__ == "__main__":
    main()
