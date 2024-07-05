#! /bin/bash
backbone=checkpoints/dinov2_converted.pth

# config=configs/dg/rein_dinov2_hrda_1024x1024.py
config=configs/dg/gta2citys/dg_lora_dinov2_linearhead.py

# checkpoint=work_dirs/dg_gta2citys_rein_dinov2_Segformer_512x512_bs1x4/iter_40000.pth
checkpoint=work_dirs/dg_lora_dinov2_linearhead/iter_40000.pth

gpu_id=7

# CUDA_VISIBLE_DEVICES=$gpu_id python3 test.py $config $checkpoint --backbone $backbone 
CUDA_VISIBLE_DEVICES=$gpu_id python3 test.py $config $checkpoint
         