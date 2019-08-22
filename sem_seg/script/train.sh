#!/bin/bash

conda activate deepgcn
python -u sem_seg/train.py  --multi_gpus --train --train_path data/3D/S3DIS --batch_size 16
echo "===> training loaded Done..."

