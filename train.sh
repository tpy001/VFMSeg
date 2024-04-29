#! /bin/bash

config=configs/dinov2/uda_rein_dinov2_mask2former_512x512_bs1x4.py
gpu_id=7

CUDA_VISIBLE_DEVICES=$gpu_id python3 train.py $config
         