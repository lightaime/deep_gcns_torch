# [Training Graph Neural Networks with 1000 Layers (ICML'2021)](https://arxiv.org/abs/2106.07476)

# ogbn-arxiv dgl implementation

## All models are trained with one NVIDIA Tesla V100 (32GB GPU)

### Train the RevGAT teacher models (RevGAT+NormAdj+LabelReuse)
Expected results: Average test accuracy: 74.02 ± 0.18
```
python3 main.py --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 5 --dropout 0.75 --n-hidden 256 --save kd --backbone rev --group 2 --mode teacher
```
### Train the RevGAT student models after training the teacher models (RevGAT+N.Adj+LabelReuse+SelfKD)
Expected results: Average test accuracy: 74.26 ± 0.17
```
python3 main.py --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --n-layers 5 --dropout 0.75 --n-hidden 256 --save kd --backbone rev --group 2 --alpha 0.95 --temp 0.7 --mode student
```

### Acknowledgements

Our implementation is based on two previous submissions on OGB: [GAT+norm. adj.+label reuse](https://github.com/Espylapiza/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv)
and [GAT+label reuse+self KD](https://github.com/ShunliRen/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv)
