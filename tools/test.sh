#! /bin/bash

config=configs/dinov2/rein_dinov2_mask2former_512x512_bs1x4.py
# checkpoint=checkpoints/gta.pth
checkpoint=work_dirs/rein_dinov2_mask2former_512x512_bs1x4/iter_8000.pth
backbone=checkpoints/dinov2_converted.pth
gpu_id=7

CUDA_VISIBLE_DEVICES=$gpu_id python3 test.py $config $checkpoint --backbone $backbone