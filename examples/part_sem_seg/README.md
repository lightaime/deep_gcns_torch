## Part Segmentation on PartNet

### Preparing the Dataset
Make sure you request access to download the PartNet v0 dataset [here](https://cs.stanford.edu/~kaichun/partnet/). It's an official website of Partnet. 
Once the data is downloaded, extract the `sem_seg_h5` data and put them inside a new folder called 'raw'. 
For example, our data folder structure is like this: `/data/deepgcn/partnet/raw/sem_seg_h5/category-level`. `category` is the name of a category, eg. Bed. `level` is 1, 2, or 3. When we train and test, we set `--train_path /data/deepgcn/partnet`.

### Train
For training the default ResEdgeConv-28 with 64 filters on the Bed category, run:
```
python examples/part_sem_seg/main.py --phase train  --category 1 --train_path /data/deepgcn/part
```
If you want to train a model with other gcn layers (for example mrgcn), run
```
python examples/part_sem_seg/main.py --phase train --category 1 --conv mr --train_path /data/deepgcn/part
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
clss = ['Bag', 'Bed', 'Bottle', 'Bowl', 'Chair', 'Clock', 'Dishwasher', 'Display', 'Door', 'Earphone',  # 0-9
        'Faucet', 'Hat', 'Keyboard', 'Knife', 'Lamp', 'Laptop', 'Microwave', 'Mug', 'Refrigerator', 'Scissors',  # 10-19
        'StorageFurniture', 'Table', 'TrashCan', 'Vase'] 
```
### Test

#### Loading Pretrained Models
Our pretrained models can be found from [Google Cloud](https://drive.google.com/open?id=15v_zDUMgpB6pf2F2_YJsDizeyHwe-7Oc).

The Naming format of our pretrained model is: `task-connection-conv_type-n_blocks-n_filters_model_best.pth`, eg. `part_sem_seg-res-edge-28-64_model_best.pth`

Use the parameter `--pretrained_model` to set a specific pretrained model to load. For example, 
```
python -u examples/part_sem_seg/main.py --category 1 --pretrained_model checkpoints/part_sem_seg-res-edge-28-64_model_best.pth --train_path /data/deepgcn/part
```
Please also specify the number of blocks and filters. 
Note: the path of `--pretrained_model` is a relative path to `examples/part_sem_seg/main.py`, so don't add `examples/part_sem_seg` in `--pretrained_model`. Or you can feed an absolute path of `--pretrained_model`. 

#### Evaluation
Use the following command to test a model:
```
python -u examples/part_sem_seg/main.py --phase test --test_path /data/deepgcn/part --category 1
```
#### Visualization
1. step1
Use the script `eval.py` to generate `.obj` files to be visualized:
```
python -u examples/part_sem_seg/eval.py --phase test --test_path /data/deepgcn/part --category 1
```
2. step2
To visualize the output of a trained model please use `visualize.py`.
Define the category's name and model number in the script and run below:
```
python -u examples/part_sem_seg/visualize.py
```
