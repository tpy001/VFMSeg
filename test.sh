#! /bin/bash

config=configs/dinov2/dinov2_hrda_1024x1024.py
# checkpoint=work_dirs/dinov2_hrda_1024x1024/FrozenDINO+HRDA+RareClassSampling/best_citys_mIoU_iter_26000.pth
checkpoint=work_dirs/dinov2_hrda_1024x1024/test/best_citys_mIoU_iter_26000.pth

gpu_id=7

CUDA_VISIBLE_DEVICES=$gpu_id python3 test.py $config $checkpoint
         