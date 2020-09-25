# ogbg_mol
The code is shared by two molecular datasets: ogbg_molhiv and ogbg_molpcba.
## Default 
	--use_gpu False 
	--dataset ogbg-molhiv
	--batch_size 32
    --block res+	#options: [plain, res, res+]
    --conv gen
    --gcn_aggr max 	#options: [max, mean, add, softmax_sg, softmax, power]
    --num_layers 3
    --conv_encode_edge False
    --add_virtual_node False
	--mlp_layers 1
    --norm batch
    --hidden_channels 256
    --epochs 300
    --lr 0.01
	--dropout 0.5
	--graph_pooling mean  #options: [mean, max, add]
## ogbg_molhiv: DyResGEN
### Train
	python main.py --use_gpu --conv_encode_edge --num_layers 7 --dataset ogbg-molhiv --block res+ --gcn_aggr softmax --t 1.0 --learn_t --dropout 0.2
### Test (use pre-trained model, [download](https://drive.google.com/file/d/1ja1xc2a4U4ps8AtZm5xo2CmffWA-C5Yl/view?usp=sharing) from Google Drive)
	python test.py --use_gpu --conv_encode_edge --num_layers 7 --dataset ogbg-molhiv --block res+ --gcn_aggr softmax --t 1.0 --learn_t

## ogbg_molpcba: ResGEN + virtual nodes
### Train
    python main.py --use_gpu --conv_encode_edge --add_virtual_node --mlp_layers 2 --num_layers 14 --dataset ogbg-molpcba --block res+ --gcn_aggr softmax_sg --t 0.1

### Test (use pre-trained model, [download](https://drive.google.com/file/d/1OYds41b7NNKGYBt52bro8lbxSCXALalx/view?usp=sharing) from Google Drive)

    python test.py --use_gpu --conv_encode_edge --add_virtual_node --mlp_layers 2 --num_layers 14 --dataset ogbg-molpcba --block res+ --gcn_aggr softmax_sg --t 0.1 --model_load_path ogbg_molpcba_pretrained_model.pth
