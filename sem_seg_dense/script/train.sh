#!/bin/bash

conda activate deepgcn
python -u sem_seg_dense/train.py  --multi_gpus --train --train_path data/3D/S3DIS --batch_size 16 --task sem_seg_dense
echo "===> training loaded Done..."

