# DeepGCNs: Can GCNs Go as Deep as CNNs?
In this work, we present new ways to successfully train very deep GCNs. We borrow concepts from CNNs, mainly residual/dense connections and dilated convolutions, and adapt them to GCN architectures. Through extensive experiments, we show the positive effect of these deep GCN frameworks.

[[Project]](https://sites.google.com/view/deep-gcns) [[Paper]](https://arxiv.org/abs/1904.03751) [[Slides]](https://docs.google.com/presentation/d/1L82wWymMnHyYJk3xUKvteEWD5fX0jVRbCbI65Cxxku0/edit?usp=sharing) [[Tensorflow Code]](https://github.com/lightaime/deep_gcns) [[Pytorch Code]](https://github.com/lightaime/deep_gcns_torch)

<div style="text-align:center"><img src='./misc/intro.png' width=800>

## Overview
We do extensive experiments to show how different components (#Layers, #Filters, #Nearest Neighbors, Dilation, etc.) effect `DeepGCNs`. We also provide ablation studies on different type of Deep GCNs (MRGCN, EdgeConv, GraphSage and GIN).

<div style="text-align:center"><img src='./misc/pipeline.png' width=800>

Further information and details please contact [Guohao Li](https://ghli.org) and [Matthias Muller](https://matthias.pw/).

## Requirements
* [Pytorch 1.1](https://pytorch.org)
* [pytorch_geometric 1.3.0](https://pytorch-geometric.readthedocs.io/en/latest/)

```
bash deepgcn_env_install.sh
```

## Citation
Please cite our paper if you find anything helpful,

```
@InProceedings{li2019deepgcns,
    title={DeepGCNs: Can GCNs Go as Deep as CNNs?},
    author={Guohao Li and Matthias Müller and Ali Thabet and Bernard Ghanem},
    booktitle={The IEEE International Conference on Computer Vision (ICCV)},
    year={2019}
}
```

## License
MIT License

## Acknowledgement
Thanks for [Guocheng Qian](https://www.gcqian.com/) for the implementation of the Pytorch version.
