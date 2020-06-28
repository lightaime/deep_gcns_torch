import torch
from torch_geometric.data import DataLoader
from model import DeeperGCN
from tqdm import tqdm
from args import ArgsInit
from utils.data_util import add_zeros, extract_node_feature
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from functools import partial


@torch.no_grad()
def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            pred = model(batch)
            y_true.append(batch.y.view(-1, 1).detach().cpu())
            y_pred.append(torch.argmax(pred.detach(), dim=1).view(-1, 1).cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)['acc']


def main():

    args = ArgsInit().args

    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')

    if args.not_extract_node_feature:
        dataset = PygGraphPropPredDataset(name=args.dataset,
                                          transform=add_zeros)
    else:
        extract_node_feature_func = partial(extract_node_feature, reduce=args.aggr)
        dataset = PygGraphPropPredDataset(name=args.dataset,
                                          transform=extract_node_feature_func)

    args.num_tasks = dataset.num_classes
    evaluator = Evaluator(args.dataset)

    split_idx = dataset.get_idx_split()

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    print(args)

    model = DeeperGCN(args)
    model.load_state_dict(torch.load(args.model_load_path)['model_state_dict'])
    model.to(device)

    train_accuracy = eval(model, device, train_loader, evaluator)
    valid_accuracy = eval(model, device, valid_loader, evaluator)
    test_accuracy = eval(model, device, test_loader, evaluator)

    print({'Train': train_accuracy,
           'Validation': valid_accuracy,
           'Test': test_accuracy})
    model.print_params(final=True)


if __name__ == "__main__":
    main()
