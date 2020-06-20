## [Semantic segmentation of indoor scenes](https://arxiv.org/pdf/1904.03751.pdf)

<p align="center">
  <img src='https://github.com/lightaime/deep_gcns_torch/blob/master/misc/sem_seg_s3dis.png' width=800>
</p>

Sem_seg_dense and sem_seg_sparse are both for the semantic segmentation task. The difference between them is that the data shape is different. 
As for sem_seg_sparse, data shape is N x feature_size and there is a batch variable indicating the batch of each node. 
But for sem_seg_dense, data shape is Batch_size x feature_size x nodes_num x 1. 

In gcn_lib, there are two folders: dense and sparse. They are used for different data shapes above.


### Note
We suggest to use sem_seg_dense. It is more efficient. 

### Train
We keep using 2 Tesla V100 GPUs for distributed training. Run:
```
CUDA_VISIBLE_DEVICES=0,1 python examples/sem_seg_sparse/train.py  --multi_gpus --phase train --train_path /data/deepgcn/S3DIS
```
Note on `--train_path`: Make sure you have the folder. Just need to set `--train_path path/to/data`, dataset will be downloaded automatically. 

If you want to train model with other gcn layers (for example mrgcn), run
```
python train.py --conv mr --multi_gpus --phase train  --train_path /data/deepgcn/S3DIS
```
Other parameters for changing the architecture are:
```
    --block         graph backbone block type {res, dense}
    --conv          graph conv layer {edge, mr, sage, gin, gcn, gat}
    --n_filters     number of channels of deep features, default is 64
    --n_blocks      number of basic blocks, default is 28
```

### Evaluation
Qucik test on area 5, run:

```
python test.py --pretrained_model checkpoints/densedeepgcn-res-edge-ckpt_50.pth  --batch_size 1  --test_path /data/deepgcn/S3DIS
```

#### Visualization
Coming soon!! 

