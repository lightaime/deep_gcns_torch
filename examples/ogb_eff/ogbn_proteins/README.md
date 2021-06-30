# [Training Graph Neural Networks with 1000 Layers (ICML'2021)](https://arxiv.org/abs/2106.07476)

# ogbn-proteins

Our models RevGNN-Deep (1001 layers with 80 channels each) and RevGNN-Wide (448 layers with 224 channels each) were both trained on a single commodity GPU and achieve an ROC-AUC of 87.74 ± 0.13 and 88.24 ± 0.15 on the ogbn-proteins dataset. To the best of our knowledge, RevGNN-Deep is the deepest GNN in the literature by one order of magnitude.

## Default 
 ``` 
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
 ``` 
    
## All models are trained with one NVIDIA Tesla V100 (32GB GPU)

## RevGNN-Wide (448 layers, 224 channels)

### Train the RevGNN-Wide (448 layers, 224 channels) model on one GPU
``` 
python main.py --use_gpu --conv_encode_edge --use_one_hot_encoding --block res+ --gcn_aggr max --num_layers 448 --hidden_channels 224 --lr 0.001 --backbone rev --dropout 0.2 --group 2
``` 

### Test the RevGNN-Wide model by multiple view inference (e.g. 10 times with 3 parts)
Pre-trained model: [download](https://drive.google.com/drive/folders/1Bw6S0OUy8qDIZIfwQOD5I5VBjPdmN9yB?usp=sharing) from Google Drive.
 
Expected test ROC-AUC: 88.24 ± 0.15. Need 48G GPU memory. NVIDIA RTX 6000 (48G) is recommented.
```
python test.py --use_gpu --conv_encode_edge --use_one_hot_encoding --block res+ --gcn_aggr max --num_layers 448 --hidden_channels 224 --lr 0.001 --backbone rev --dropout 0.2 --group 2 --model_load_path revgnn_wide.pth  --valid_cluster_number 3 --num_evals 10
```
### Test the RevGNN-Wide model by single inference (e.g. 1 time with 5 parts)
Pre-trained model, [download](https://drive.google.com/drive/folders/1Bw6S0OUy8qDIZIfwQOD5I5VBjPdmN9yB?usp=sharing) from Google Drive.
 
Expected test ROC-AUC: 87.62 ± 0.18. 32G GPU is enough. NVIDIA Tesla V100 (32GB GPU) is recommented.
```
python test.py --use_gpu --conv_encode_edge --use_one_hot_encoding --block res+ --gcn_aggr max --num_layers 448 --hidden_channels 224 --lr 0.001 --backbone rev --dropout 0.2 --group 2 --model_load_path revgnn_wide.pth  --valid_cluster_number 5 --num_evals 1
```    

## RevGNN-Deep (1001 layers, 80 channels)

### Train the RevGNN-Deep (1001 layers, 80 channels) model on one GPU
``` 
python main.py --use_gpu --conv_encode_edge --use_one_hot_encoding --block res+ --gcn_aggr max --num_layers 1001 --hidden_channels 80 --lr 0.001 --backbone rev --dropout 0.1 --group 2
``` 

### Test the RevGNN-Deep model by multiple view inference (e.g. 10 times with 3 parts)
Pre-trained model, [download](https://drive.google.com/drive/folders/1Bw6S0OUy8qDIZIfwQOD5I5VBjPdmN9yB?usp=sharing) from Google Drive.
 
Expected test ROC-AUC 87.74 ± 0.13. 32G GPU is enough. NVIDIA Tesla V100 (32GB GPU) is recommented.
```
python test.py --use_gpu --conv_encode_edge --use_one_hot_encoding --block res+ --gcn_aggr max --num_layers 1001 --hidden_channels 80 --lr 0.001 --backbone rev --dropout 0.1 --group 2 --model_load_path revgnn_deep.pth  --valid_cluster_number 3 --num_evals 10
```

### Test the RevGNN-Deep model by single inference (e.g. 1 time with 5 parts)
Pre-trained model, [download](https://drive.google.com/drive/folders/1Bw6S0OUy8qDIZIfwQOD5I5VBjPdmN9yB?usp=sharing) from Google Drive.

Expected test ROC-AUC 87.06 ± 0.20. 32G GPU is enough. NVIDIA Tesla V100 (32GB GPU) is recommented.
```
python test.py --use_gpu --conv_encode_edge --use_one_hot_encoding --block res+ --gcn_aggr max --num_layers 1001 --hidden_channels 80 --lr 0.001 --backbone rev --dropout 0.1 --group 2 --model_load_path revgnn_deep.pth  --valid_cluster_number 5 --num_evals 1
```

### Acknowledgements
The [reversible module](../../../eff_gcn_modules/rev/gcn_revop.py) is implemented based on [MemCNN](https://github.com/silvandeleemput/memcnn/blob/master/LICENSE.txt) under MIT license.

