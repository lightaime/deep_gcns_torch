# ogbg-molhiv
## Default 
	--use_gpu False 
	--batch_size 32
    --block res+	#options: [plain, res, res+]
    --conv gen
    --gcn_aggr max 	#options: [max, mean, add, softmax_sg, softmax, power]
    --num_layers 3
    --conv_encode_edge False
	--mlp_layers 1
    --norm batch
    --hidden_channels 256
    --epochs 300
    --lr 0.01
	--dropout 0.5
	--graph_pooling mean  #options: [mean, max, add]

## DyResGEN
### Train
	python main.py --use_gpu --conv_encode_edge --num_layers 7 --dataset ogbg-molhiv --block res+ --gcn_aggr softmax --t 1.0 --learn_t
### Test (use pre-trained model, [download](https://drive.google.com/open?id=1ja1xc2a4U4ps8AtZm5xo2CmffWA-C5Yl) from Google Drive)
	python test.py --use_gpu --conv_encode_edge --num_layers 7 --dataset ogbg-molhiv --block res+ --gcn_aggr softmax --t 1.0 --learn_t
