import torch
from torch_geometric.data import DataLoader
from model import DeeperGCN
from tqdm import tqdm
from args import ArgsInit
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator


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
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"y_true": y_true,
                  "y_pred": y_pred}

    return evaluator.eval(input_dict)


def main():

    args = ArgsInit().args

    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')

    dataset = PygGraphPropPredDataset(name=args.dataset)
    args.num_tasks = dataset.num_tasks
    print(args)

    if args.feature == 'full':
        pass
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:, :2]
        dataset.data.edge_attr = dataset.data.edge_attr[:, :2]


    split_idx = dataset.get_idx_split()

    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = DeeperGCN(args)

    model.load_state_dict(torch.load(args.model_load_path)['model_state_dict'])
    model.to(device)

    train_result = eval(model, device, train_loader, evaluator)[dataset.eval_metric]
    valid_result = eval(model, device, valid_loader, evaluator)[dataset.eval_metric]
    test_result = eval(model, device, test_loader, evaluator)[dataset.eval_metric]

    print({'Train': train_result,
           'Validation': valid_result,
           'Test': test_result})

    model.print_params(final=True)


if __name__ == "__main__":
    main()
