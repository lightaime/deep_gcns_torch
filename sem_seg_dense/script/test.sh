#!/bin/bash
conda activate deepgcn
# TODO:
python -u sem_seg_dense/test.py --pretrained_model sem_seg_dense/checkpoints/densedeepgcn-res-edge-ckpt_50.pth  --batch_size 1  --test_path /data/deepgcn/S3DIS --task sem_seg_dense

#--test_path data/3D/S3DIS
