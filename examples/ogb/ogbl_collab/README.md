# ogbn-arxiv
## Default 
	--use_gpu False 
    --block res+ 	#options: [plain, res, res+]
    --conv gen
    --gcn_aggr max 	#options: [max, mean, add, softmax_sg, softmax, power]
    --num_layers 3 #the number of layers of DeeperGCN model
    --lp_num_layers 3 #the number of layers of the link predictor model
	--mlp_layers 1
    --norm batch
    --lp_norm #the type of normalization layer for link predictor
    --hidden_channels 128
    --epochs 400
    --lr 0.001
	--dropout 0.0
## DyResGEN
### Train
SoftMax aggregator with learnable t (initialized as 1.0)
    
    python main.py --use_gpu --num_layers 7 --block res+ --gcn_aggr softmax --learn_t --t 1.0

### Test (use pre-trained model, [DyResGEN](https://drive.google.com/file/d/1aPzYzXiKBN7vnSVHFfO010zJwTYWazgM/view?usp=sharing) and [Link Predictor](https://drive.google.com/file/d/1Y-UZjIxXA6swX8qGLs041Cg_qSh7SFgx/view?usp=sharing) from Google Drive)
	python test.py --use_gpu --num_layers 7 --block res+ --gcn_aggr softmax --learn_t --t 1.0
