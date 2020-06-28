# ogbn-products
We simply apply a random partition to generate batches for mini-batch training on GPU and full-batch test on CPU. We set the number of partitions to be 10 for training and the batch size is 1 subgraph.
## Default 
	--use_gpu False 
	--self_loop False
	--cluster_number 10
    --block res+ 	#options: [plain, res, res+]
    --conv gen
    --gcn_aggr max 	#options: [max, mean, add, softmax_sg, softmax, power]
    --num_layers 3
	--mlp_layers 1
    --norm batch
    --hidden_channels 128
    --epochs 500
    --lr 0.01
	--dropout 0.5
## ResGEN
### Train
	python main.py --use_gpu --self_loop --num_layers 14 --gcn_aggr softmax_sg --t 0.1

### Test (use pre-trained model, [download](https://drive.google.com/file/d/1OxyA2IZN-4BCfkWzUG8QBS-khxhHHnZB/view?usp=sharing) from Google Drive)
	python test.py --self_loop --num_layers 14 --gcn_aggr softmax_sg --t 0.1
