#! /bin/bash

config=configs/dg/gta2citys/dg_lora_dinov2_ms_1024x1024.py
gpu_id=7

CUDA_VISIBLE_DEVICES=$gpu_id python3 train.py $config
         