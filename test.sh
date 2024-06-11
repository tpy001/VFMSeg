#! /bin/bash
# backbone=checkpoints/dinov2_converted.pth

# config=configs/dg/rein_dinov2_hrda_1024x1024.py
config=configs/dinov2/dg_gta2citys_rein_dinov2_Segformer_512x512_bs1x4.py

# checkpoint=work_dirs/dg_gta2citys_rein_dinov2_Segformer_512x512_bs1x4/iter_40000.pth
checkpoint=results/DG/gta2citys/mIoU=69.76-ReinDINO+SegFormer-GN/iter_40000.pth

gpu_id=3

CUDA_VISIBLE_DEVICES=$gpu_id python3 test.py $config $checkpoint --backbone $backbone
         