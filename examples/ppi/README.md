## [Graph Learning on Biological Networks](https://arxiv.org/pdf/1910.06849.pdf)

<p align="center">
  <img src='https://github.com/lightaime/deep_gcns_torch/blob/master/misc/ppi.png' width=500>
</p>

### Train
We train each model on one tesla V100.

For training the default ResMRConv-14 with 64 filters, run
```
python -u examples/ppi/main.py --phase train --data_dir /data/deepgcn/ppi
```
If you want to train model with other gcn layers (for example EdgeConv, 28 layers, 256 channels in the first layer, with dense connection), run
```
python -u examples/ppi/main.py --phase train --conv edge --data_dir /data/deepgcn/ppi  --block dense --n_filters 256 --n_blocks 28
```

Just need to set `--data_dir` into your data folder, dataset will be downloaded automatically.
Other parameters for changing the architecture are:
```
--block         graph backbone block type {res, plain, dense}
--conv          graph conv layer {edge, mr, sage, gin, gcn, gat}
--n_filters     number of channels of deep features, default is 64
--n_blocks      number of basic blocks, default is 28
```
### Test
#### Pretrained Models
Our pretrained models can be found from [Goolge Cloud](https://drive.google.com/drive/folders/1LoT1B9FDgylUffHY8K43FFfred-luZaz?usp=sharing).

The Naming format of our pretrained model: `task-connection-conv_type-n_blocks-n_filters_phase_best.pth`, eg. `ppi-res-mr-28-256_val_best.pth`, which means PPI node classification task, with residual connection, convolution is MRGCN, 28 layers, 256 channels, the best pretrained model found in validation dataset.

Use parameter `--pretrained_model` to set the specific pretrained model you want. 
```
python -u examples/ppi/main.py --phase test --pretrained_model checkpoints/ppi-res-mr-28-256_val_best.pth --data_dir /data/deepgcn/ppi --n_filters 256 --n_blocks 28 --conv mr --block res
```

```
python -u examples/ppi/main.py --phase test --pretrained_model checkpoints/ppi-dense-mr-14-256_val_best.pth --data_dir /data/deepgcn/ppi --n_filters 256 --n_blocks 14 --conv mr --block dense
```
Please also specify the number of blocks and filters according to the name of pretrained models.
