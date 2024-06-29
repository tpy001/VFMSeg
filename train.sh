#! /bin/bash

config=configs/dg/gta2citys/dg_lora_dinov2_linearhead.py
gpu_id=1

CUDA_VISIBLE_DEVICES=$gpu_id python3 train.py $config
         