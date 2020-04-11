## Point cloud classification on ModelNet


### Train
We train PlainGCN-28 and ResGCN-28 models on one Tesla V100.
For DenseGCN,  we use 4 Tesla V100s.

For training ResGCN-28, run:
```
python main.py --phase train --n_blocks 28 --block res --data /data/deepgcn/modelnet40
```
Just need to set `--data` into your data folder, dataset will be downloaded automatically.

### Test

Our pretrained models can be found from [Google Drive](https://drive.google.com/drive/folders/1LUWH0V3ZoHNQBylj0u0_36Mx0-UrDh1v?usp=sharing).

Use the parameter `--pretrained_model` to set a specific pretrained model to load. For example,

test the best large architecture  using cri2
```
python main.py --phase test --n_blocks 28 --block res  --pretrained_model 
```

