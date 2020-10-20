import __init__
from ogb.nodeproppred import Evaluator
import torch
from torch.utils.data import DataLoader
from args import ArgsInit
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from model import DeeperGCN, LinkPredictor
from utils.ckpt_util import save_ckpt
import logging
import time
from torch.utils.tensorboard import SummaryWriter


@torch.no_grad()
def test(model, predictor, x, edge_index, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(x, edge_index)

    pos_train_edge = split_edge['train']['edge'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


def train(model, predictor,
          x, edge_index, split_edge, optimizer, batch_size):

    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(x.device)

    total_loss = total_examples = 0

    for perm in DataLoader(range(pos_train_edge.size(0)),
                           batch_size,
                           shuffle=True):

        optimizer.zero_grad()
        h = model(x, edge_index)
        # positive edges
        edge = pos_train_edge[perm].t()
        pos_out = predictor(h[edge[0]], h[edge[1]])
        # add a extremely small value to avoid gradient explode
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        # negative edges
        edge = torch.randint(0, x.size(0),
                             edge.size(), dtype=torch.long,
                             device=h.device)
        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()
        # why need to do this? clip grad norm
        # https://towardsdatascience.com/what-is-gradient-clipping-b8e815cdfb48
        # ||g|| < c if not new_g = g / ||g||
        # tackle exploding gradients issue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


def main():

    args = ArgsInit().save_exp()

    if args.use_tensor_board:
        writer = SummaryWriter(log_dir=args.save)

    if args.use_gpu:
        device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device('cpu')

    dataset = PygLinkPropPredDataset(name=args.dataset)
    data = dataset[0]
    # Data(edge_index=[2, 2358104], edge_weight=[2358104, 1], edge_year=[2358104, 1], x=[235868, 128])
    split_edge = dataset.get_edge_split()
    evaluator = Evaluator(args.dataset)

    x = data.x.to(device)

    edge_index = data.edge_index.to(device)

    args.in_channels = data.x.size(-1)
    args.num_tasks = 1

    logging.info('%s' % args)

    model = DeeperGCN(args).to(device)
    predictor = LinkPredictor(args).to(device)

    logging.info(model)
    logging.info(predictor)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()),
                                 lr=args.lr)

    results = {}
    keys = ['highest_valid', 'final_train', 'final_test', 'highest_train']
    hits = ['Hits@10', 'Hits@50', 'Hits@100']

    for key in keys:
        results[key] = {k: 0 for k in hits}

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):

        epoch_loss = train(model, predictor,
                           x, edge_index,
                           split_edge,
                           optimizer, args.batch_size)
        logging.info('Epoch {}, training loss {:.4f}'.format(epoch, epoch_loss))
        model.print_params(epoch=epoch)

        result = test(model, predictor,
                      x, edge_index,
                      split_edge,
                      evaluator, args.batch_size)

        for k in hits:
            # return a tuple
            train_result, valid_result, test_result = result[k]

            if args.use_tensor_board and k == 'Hits@50':
                writer.add_scalar('stats/train_loss', epoch_loss, epoch)
                writer.add_scalar('stats/train_Hits@50', train_result, epoch)
                writer.add_scalar('stats/valid_Hits@50', valid_result, epoch)
                writer.add_scalar('stats/test_Hits@50', test_result, epoch)

            if train_result > results['highest_train'][k]:
                results['highest_train'][k] = train_result

            if valid_result > results['highest_valid'][k]:
                results['highest_valid'][k] = valid_result
                results['final_train'][k] = train_result
                results['final_test'][k] = test_result

                save_ckpt(model, optimizer,
                          round(epoch_loss, 4), epoch,
                          args.model_save_path,
                          k, name_post='valid_best')
                save_ckpt(predictor, optimizer,
                          round(epoch_loss, 4), epoch,
                          args.model_save_path,
                          k, name_post='valid_best_link_predictor')

        logging.info(result)

    logging.info("%s" % results)

    end_time = time.time()
    total_time = end_time - start_time
    time_used = 'Total time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time)))
    logging.info(time_used)


if __name__ == "__main__":
    main()
