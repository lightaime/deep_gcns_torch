import torch
from torch_geometric.data import DataLoader
import torch.optim as optim
from model import DeeperGCN
from tqdm import tqdm
from args import ArgsInit
from utils.ckpt_util import save_ckpt
from utils.data_util import add_zeros, extract_node_feature
import logging
from functools import partial
import time
import statistics
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator


def train(model, device, loader, optimizer, criterion):
    loss_list = []
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()

            loss = criterion(pred.to(torch.float32), batch.y.view(-1, ))

            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
    return statistics.mean(loss_list)


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
    args = ArgsInit().save_exp()

    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')

    sub_dir = 'BS_{}'.format(args.batch_size)

    if args.not_extract_node_feature:
        dataset = PygGraphPropPredDataset(name=args.dataset,
                                          transform=add_zeros)
    else:
        extract_node_feature_func = partial(extract_node_feature, reduce=args.aggr)
        dataset = PygGraphPropPredDataset(name=args.dataset,
                                          transform=extract_node_feature_func)

        sub_dir = sub_dir + '-NF_{}'.format(args.aggr)

    args.num_tasks = dataset.num_classes
    evaluator = Evaluator(args.dataset)

    logging.info('%s' % args)

    split_idx = dataset.get_idx_split()

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    model = DeeperGCN(args).to(device)

    logging.info(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    results = {'highest_valid': 0,
               'final_train': 0,
               'final_test': 0,
               'highest_train': 0}

    start_time = time.time()

    evaluate = True

    for epoch in range(1, args.epochs + 1):
        logging.info("=====Epoch {}".format(epoch))
        logging.info('Training...')

        epoch_loss = train(model, device, train_loader, optimizer, criterion)

        if args.num_layers > args.num_layers_threshold:
            if epoch % args.eval_steps != 0:
                evaluate = False
            else:
                evaluate = True

        model.print_params(epoch=epoch)

        if evaluate:

            logging.info('Evaluating...')

            train_accuracy = eval(model, device, train_loader, evaluator)
            valid_accuracy = eval(model, device, valid_loader, evaluator)
            test_accuracy = eval(model, device, test_loader, evaluator)

            logging.info({'Train': train_accuracy,
                          'Validation': valid_accuracy,
                          'Test': test_accuracy})

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
