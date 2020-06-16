# ogbn-arxiv
## Default 
	--use_gpu False 
	--self_loop False
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
	python main.py --use_gpu --self_loop --num_layers 28 --block res+ --gcn_aggr softmax_sg --t 0.1

### Test (use pre-trained model, [download](https://drive.google.com/open?id=19DA0SzfInkb3Q2cdeazejJ_mYMAvRZyb) from Google Drive)
	python test.py --use_gpu --self_loop --num_layers 28 --block res+ --gcn_aggr softmax_sg --t 0.1
