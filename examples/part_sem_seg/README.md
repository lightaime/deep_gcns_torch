## [Part Segmentation on PartNet](https://arxiv.org/pdf/1910.06849.pdf)

<p align="center">
  <img src='https://github.com/lightaime/deep_gcns_torch/blob/master/misc/part_sem_seg.png' width=800>
</p>

### Preparing the Dataset
Make sure you request access to download the PartNet v0 dataset [here](https://cs.stanford.edu/~kaichun/partnet/). It's an official website of Partnet. 
Once the data is downloaded, extract the `sem_seg_h5` data and put them inside a new folder called 'raw'. 
For example, our data folder structure is like this: `/data/deepgcn/partnet/raw/sem_seg_h5/category-level`. `category` is the name of a category, eg. Bed. `level` is 1, 2, or 3. When we train and test, we set `--data_dir /data/deepgcn/partnet`.

### Train
We train each model on one tesla V100. 

For training the default ResEdgeConv-28 with 64 filters on the Bed category, run:
```
python main.py --phase train  --category 1 --data_dir /data/deepgcn/partnet
```
Note that, We only focus on fine-grained level of part segmentation in the experiment. 
For all the categories, we use the same training parameters as default (see `config.py` for details). 

If you want to train a model with other gcn layers (for example mrgcn), run
```
python main.py --phase train --category 1 --conv mr --data_dir /data/deepgcn/partnet
```
Other important parameters are:
```
--block         graph backbone block type {res, plain, dense}
--conv          graph conv layer {edge, mr, sage, gin, gcn, gat}
--n_filters     number of channels of deep features, default is 64
--n_blocks      number of basic blocks, default is 28
--category      NO. of category. default is 1 (Bed)
```
The category list is:
```
category_names = ['Bag', 'Bed', 'Bottle', 'Bowl', 'Chair', 'Clock', 'Dishwasher', 'Display', 'Door', 'Earphone',  # 0-9
        'Faucet', 'Hat', 'Keyboard', 'Knife', 'Lamp', 'Laptop', 'Microwave', 'Mug', 'Refrigerator', 'Scissors',  # 10-19
        'StorageFurniture', 'Table', 'TrashCan', 'Vase'] 
```
### Test
We test and report results on the testing dataset using the checkpoints which perform the best in the validation dataset.  
Our pretrained models can be found from [Google Cloud](https://drive.google.com/drive/folders/15Y7Ao4VBysHBHxyQwYvb2SU1iFi9ZZRK?usp=sharing).  

The Naming format of our pretrained model is: `task-category-segmentationLevel-conv-n_blocks-n_filters-otherParameters-val_best_model_best.pth`, eg. `PartnetSemanticSeg-Bed-L3-res-edge-n28-C64-k9-drop0.5-lr0.005_B6-val_best_model.pth`. 
val_best means the checkpoint is the best one on the validation dataset. 

Use the parameter `--pretrained_model` to set a specific pretrained model to load. For example, 
```
python -u main.py --phase test --category 1 --pretrained_model checkpoints/PartnetSemanticSeg-Bed-L3-res-edge-n28-C64-k9-drop0.5-lr0.005_B6-val_best_model.pth --data_dir /data/deepgcn/partnet  --test_batch_size 8
```
Please also specify the number of blocks and filters.   
Note: 
- the path of `--pretrained_model` is a relative path to `main.py`, so don't add `examples/part_sem_seg` in `--pretrained_model`. Or you can feed an absolute path of `--pretrained_model`.  
- if you do not have V100, you can set the `test_batch_size` to 1. It does not influence the test accuracy.  

#### Visualization
1. step1
Use the script `eval.py` to generate `.obj` files to be visualized:
```
python -u eval.py --phase test --category 1 --pretrained_model checkpoints/PartnetSemanticSeg-Bed-L3-res-edge-n28-C64-k9-drop0.5-lr0.005_B6-val_best_model.pth --data_dir /data/deepgcn/partnet
```
2. step2
To visualize the output of a trained model please use `visualize.py`.
Define the category's name and model number in the script and run below:
```
python -u visualize.py
```
