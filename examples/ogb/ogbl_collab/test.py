import __init__
from ogb.nodeproppred import Evaluator
import torch
from torch.utils.data import DataLoader
from args import ArgsInit
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from model import DeeperGCN, LinkPredictor


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


def main():

    args = ArgsInit().args

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

    print(args)

    model = DeeperGCN(args).to(device)
    predictor = LinkPredictor(args).to(device)

    model.load_state_dict(torch.load(args.model_load_path)['model_state_dict'])
    model.to(device)

    predictor.load_state_dict(torch.load(args.predictor_load_path)['model_state_dict'])
    predictor.to(device)

    hits = ['Hits@10', 'Hits@50', 'Hits@100']

    result = test(model, predictor, x, edge_index, split_edge, evaluator, args.batch_size)

    for k in hits:
        train_result, valid_result, test_result = result[k]
        print('{}--Train: {}, Validation: {}, Test: {}'.format(k,
                                                               train_result,
                                                               valid_result,
                                                               test_result))


if __name__ == "__main__":
    main()
