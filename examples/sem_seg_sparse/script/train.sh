#!/bin/bash

conda activate deepgcn
python -u train.py  --multi_gpus --phase train --train_path /data/deepgcn/S3DIS --batch_size 16
echo "===> training loaded Done..."

