# DeepGCNs: Can GCNs Go as Deep as CNNs?
In this work, we present new ways to successfully train very deep GCNs. We borrow concepts from CNNs, mainly residual/dense connections and dilated convolutions, and adapt them to GCN architectures. Through extensive experiments, we show the positive effect of these deep GCN frameworks.

[[Project]](https://www.deepgcns.org/) [[Paper]](https://arxiv.org/abs/1904.03751) [[Slides]](https://docs.google.com/presentation/d/1L82wWymMnHyYJk3xUKvteEWD5fX0jVRbCbI65Cxxku0/edit?usp=sharing) [[Tensorflow Code]](https://github.com/lightaime/deep_gcns) [[Pytorch Code]](https://github.com/lightaime/deep_gcns_torch)
    
<p align="center">
  <img src='./misc/intro.png' width=800>
</p>

## Overview
We do extensive experiments to show how different components (#Layers, #Filters, #Nearest Neighbors, Dilation, etc.) effect `DeepGCNs`. We also provide ablation studies on different type of Deep GCNs (MRGCN, EdgeConv, GraphSage and GIN).

<p align="center">
  <img src='./misc/pipeline.png' width=800>
</p>

## Requirements
* [Pytorch>=1.4.0](https://pytorch.org)
* [pytorch_geometric>=1.3.0](https://pytorch-geometric.readthedocs.io/en/latest/)
* [tensorflow graphics](https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/g3doc/install.md) only used for tensorboard visualization

Install enviroment by runing:
```
source deepgcn_env_install.sh
```

## Code Architecture
    .
    ├── misc                    # Misc images
    ├── utils                   # Common useful modules
    ├── gcn_lib                 # gcn library
    │   ├── dense               # gcn library for dense data (B x C x N x 1)
    │   └── sparse              # gcn library for sparse data (N x C)
    ├── examples 
    │   ├── modelnet_cls        # code for point clouds classification on ModelNet40
    │   ├── sem_seg_dense       # code for point clouds semantic segmentation on S3DIS (data type: dense)
    │   ├── sem_seg_sparse      # code for point clouds semantic segmentation on S3DIS (data type: sparse)
    │   ├── part_sem_seg        # code for part segmentation on PartNet
    │   ├── ppi                 # code for node classification on PPI dataset
    │   └── ogb                 # code for node/graph property prediction on OGB datasets
    └── ...

## How to train, test and evaluate our models
Please look the details in `Readme.md` of each task inside `examples` folder.
All the information of code, data, and pretrained models can be found there.
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

```
@misc{li2019deepgcns_journal,
    title={DeepGCNs: Making GCNs Go as Deep as CNNs},
    author={Guohao Li and Matthias Müller and Guocheng Qian and Itzel C. Delgadillo and Abdulellah Abualshour and Ali Thabet and Bernard Ghanem},
    year={2019},
    eprint={1910.06849},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

```
@misc{li2020deepergcn,
    title={DeeperGCN: All You Need to Train Deeper GCNs},
    author={Guohao Li and Chenxin Xiong and Ali Thabet and Bernard Ghanem},
    year={2020},
    eprint={2006.07739},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## License
MIT License

## Contact
For more information please contact [Guohao Li](https://ghli.org), [Matthias Muller](https://matthias.pw/), [Guocheng Qian](https://www.gcqian.com/).
