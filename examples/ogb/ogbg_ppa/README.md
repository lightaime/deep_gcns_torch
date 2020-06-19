# ogbg-ppa
We initialize the features of nodes of ogbg_ppa dataset through aggregating the features of their connected edges by a Sum (Add) aggregation, just like what we do for ogbn_proteins.

## Default 
	--use_gpu False 
	--batch_size 32
    --aggr add		#options: [mean, max, add]
    --block res+	#options: [plain, res, res+]
    --conv gen
    --gcn_aggr max 	#options: [max, mean, add, softmax_sg, softmax, power]
    --num_layers 3
    --conv_encode_edge False
	--mlp_layers 2
    --norm layer
    --hidden_channels 128
    --epochs 200
    --lr 0.01
	--dropout 0.5
	--graph_pooling mean  #options: [mean, max, add]
## ResGEN
### Train
	python main.py --use_gpu --conv_encode_edge --num_layers 28 --gcn_aggr softmax_sg --t 0.01


### Test (use pre-trained model, [download](https://drive.google.com/file/d/1vlmNPUgDes8QJ0SQoo-K5L_yFVeV1lkH/view?usp=sharing) from Google Drive)
	python test.py --use_gpu --conv_encode_edge --num_layers 28 --gcn_aggr softmax_sg --t 0.01


