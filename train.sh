#! /bin/bash

config=configs/dinov2/uda_rein_dinov2_Segformer_512x512.py
gpu_id=6

CUDA_VISIBLE_DEVICES=$gpu_id python3 train.py $config
         