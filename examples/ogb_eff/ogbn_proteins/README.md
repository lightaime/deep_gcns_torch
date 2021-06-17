# Implementation of [Training Graph Neural Networks with 1000 Layers, ICML'2021](https://arxiv.org/abs/2106.07476)

# ogbn-proteins

We simply apply a random partition to generate batches for both mini-batch training and test. We set the number of partitions to be 10 for training and 5 for test, and we set the batch size to 1 subgraph.  We initialize the features of nodes through aggregating the features of their connected edges by a Sum (Add) aggregation.
## Default 
	--use_gpu False 
    --cluster_number 10 
    --valid_cluster_number 5 
    --aggr add 	#options: [mean, max, add]
    --block plain 	#options: [plain, res, res+]
    --conv gen
    --gcn_aggr max 	#options: [max, mean, add, softmax_sg, softmax, power]
    --num_layers 3
    --conv_encode_edge False
	--mlp_layers 2
    --norm layer
    --hidden_channels 64
    --epochs 1000
    --lr 0.01
    --dropout 0.0
    --num_evals 1
    --backbone rev
    --group 2


## RevGNN-Wide (448 layers, 224 channels)

### Train the RevGNN-Wide (448 layers, 224 channels) model on one GPU
    python main.py --use_gpu --conv_encode_edge --use_one_hot_encoding --block res+ --gcn_aggr max --num_layers 448 --hidden_channels 224 --lr 0.001 --backbone rev --dropout 0.2 --group 2

### Test the RevGNN-Wide model by multiple view inference (e.g. 10 times with 3 parts) (use pre-trained model, [download](coming soon) from Google Drive)
    python test.py  --conv_encode_edge --use_one_hot_encoding --block res+ --gcn_aggr max --num_layers 448 --hidden_channels 224 --lr 0.001 --backbone rev --dropout 0.2 --group 2 --model_load_path revgnn_wide.pth  --valid_cluster_number 3 --use_gpu --num_evals 10

### Test the RevGNN-Wide model by single inference (e.g. 1 times with 5 parts) (use pre-trained model, [download](coming soon) from Google Drive)
    python test.py  --conv_encode_edge --use_one_hot_encoding --block res+ --gcn_aggr max --num_layers 448 --hidden_channels 224 --lr 0.001 --backbone rev --dropout 0.2 --group 2 --model_load_path revgnn_wide.pth  --valid_cluster_number 5 --use_gpu --num_evals 1

## RevGNN-Deep (1001 layers, 80 channels)

### Train the RevGNN-Deep (1001 layers, 80 channels) model on one GPU
    python main.py --use_gpu --conv_encode_edge --use_one_hot_encoding --block res+ --gcn_aggr max --num_layers 1001 --hidden_channels 80 --lr 0.001 --backbone rev --dropout 0.1 --group 2

### Test the RevGNN-Wide model by multiple view inference (e.g. 10 times with 3 parts) (use pre-trained model, [download](coming soon) from Google Drive)
    python test.py  --conv_encode_edge --use_one_hot_encoding --block res+ --gcn_aggr max --num_layers 1001 --hidden_channels 80 --lr 0.001 --backbone rev --dropout 0.1 --group 2 --model_load_path revgnn_deep.pth  --valid_cluster_number 3 --use_gpu --num_evals 10

### Test the RevGNN-Wide model by single inference (e.g. 1 times with 5 parts) (use pre-trained model, [download](coming soon) from Google Drive)
    python test.py  --conv_encode_edge --use_one_hot_encoding --block res+ --gcn_aggr max --num_layers 1001 --hidden_channels 80 --lr 0.001 --backbone rev --dropout 0.1 --group 2 --model_load_path revgnn_deep.pth  --valid_cluster_number 5 --use_gpu --num_evals 1


