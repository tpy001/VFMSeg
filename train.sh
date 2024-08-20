#! /bin/bash

# config=configs/dg/gta2citys/dg_rein_eva02_mask2former_512x512_bs1x4.py
config=configs/dg/gta2citys/dg_rein_eva02_mask2former_512x512_bs1x4.py
gpu_id=7


CUDA_VISIBLE_DEVICES=$gpu_id python3 train.py $config
         
