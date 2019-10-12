## Graph Learning on Biological Networks
### Train

For training the default ResEdgeConv-28 with 64 filters, run
```
cd deep_gcns_torch
python -u examples/ppi/main.py --phase train --train_path path/to/data
```
If you want to train model with other gcn layers (for example mrgcn), run
```
cd deep_gcns_torch
python -u examples/ppi/main.py --phase train --conv mr --train_path path/to/data
```

Just need to set `--train_path path/to/data`, dataset will be downloaded automatically.
Other parameters for changing the architecture are:
```
    --block         graph backbone block type {res, plain, dense}
    --conv          graph conv layer {edge, mr, sage, gin, gcn, gat}
    --n_filters     number of channels of deep features, default is 64
    --n_blocks      number of basic blocks, default is 28
```
### Test
#### Pretrained Models
Our pretrained models will be available soon.
<!--Our pretrained models can be found [here](https://drive.google.com/drive/u/0/folders/15v_zDUMgpB6pf2F2_YJsDizeyHwe-7Oc).-->
The Naming format of our pretrained model: `task-connection-conv_type-n_blocks-n_filters_model_best.pth`, eg. `ppi-res-edge-14-256_model_best.pth`.

Use parameter `--pretrained_model` to set the specific pretrained model you want. 
```
python -u examples/ppi/main.py --pretrained_model checkpoints/ppi-res-edge-14-256_model_best.pth --train_path data/ --n_filters 256 --n_blocks 14
```
Please also specify the number of blocks and filters according to the name of pretrained models.
