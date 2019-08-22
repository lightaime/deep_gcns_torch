## Semantic segmentation of indoor scenes


### Train

We use 6-fold training, such that 6 models are trained leaving 1 of 6 areas as the testing area for each model. We keep using 2 GPUs for distributed training. To train 6 models sequentially, run
```
cd DeepGCN
bash sem_seg/script/train.sh
```
If you want to train model with other gcn layers (for example mrgcn), run
```
cd DeepGCN
python sem_seg/train.py --conv mr --multi_gpus --train
```

### Evaluation
Qucik test on area 5, run:

```
cd DeepGCN
bash sem_seg/script/test.sh
```

#### Pretrained Models

use parameter $--pretrained_model$ to change the specific pretrained model you want. Have fun.
```
python -u sem_seg/test.py --pretrained_model sem_seg/checkpoints/1v_checkpoint_50.pth  --batch_size 1 --train_path data/3D/S3DIS 
```

#### Visualization
Coming soon!! 