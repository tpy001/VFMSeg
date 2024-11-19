# Unleashing the Power of Visual Foundation Models for Generalizable Semantic Segmentation

This is the official implementation of the paper "Unleashing the Power of Visual Foundation Models for Generalizable Semantic Segmentation." In this paper, we propose a novel framework to leverage visual foundation models for domain generalizable semantic segmentation (DGSS). The core idea is to fine-tune the VFM with minimal modifications and enable inference on high-resolution images. We argue that this approach can maintain the pretrained knowledge of the VFM and unleash its power for DGSS.
We conduct experiments on various benchmarks and achieve an average mIoU of 70.3% on GTAV to {Cityscapes + BDD100K + Mapillary} and 71.62% on Cityscapes to {BDD100K + Mapillary}, outperforming the previous state-of-the-art approaches by 3.3% and 1.1% in average mIoU, respectively.

![figure2.png](res/figure2.png)
![figure3.png](res/figure3.png)

## Table of Contents

- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Preparing Visual Foundation Models](#preparing-visual-foundation-models)
- [Training](#training)
- [Evaluation](#evaluation)	
- [Overview of Important Files](#Overview-of-Important Files)

## Environment Setup

To set up the environment for this project, execute the following script:

```bash
chmod +x install.sh
./install.sh
```

This script will create a conda virtual environment named **DGVFM** and install all the required dependencies. To run the code, you should activate the virtual environment using the following command:

```bash
conda activate DGVFM
```

## Dataset Preparation

**1. Download the dataset**

* **GTA:**  Download all image and label packages from [here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract them to `data/gta`.
* **Cityscapes:** Download `leftImg8bit_trainvaltest.zip` and `gt_trainvaltest.zip` from [here](https://www.cityscapes-dataset.com/downloads/) and extract them to `data/cityscapes`.
* **BDD100K:** Download the 10K Images and Segmentation from [here](https://bdd-data.berkeley.edu/portal.html#download) and extract them to `datasets/bdd100k`.
* **Mapillary:** Download MAPILLARY v1.2 from [here](https://research.mapillary.com/) and extract it to `data/mapillary`.

The final folder structure should look like this:

```
DGVFM
├── ...
├── data
│   ├── gta
│   │   ├── images
│   │   ├── labels
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── bdd100k
│   │   ├── images
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── labels
│   │   │   ├── train
│   │   │   ├── val
│   ├── mapillary
│   │   ├── training
│   │   │   ├── images
│   │   │   ├── labels
│   │   ├── validation
│   │   │   ├── images
│   │   │   ├── val_label
├── ...
```

**2. Convert the dataset**
Prepare datasets with these commands:

```shell
cd DGVFM
python tools/convert_datasets/gta.py data/gta 
python tools/convert_datasets/cityscapes.py data/cityscapes
python tools/convert_datasets/mapillary2cityscape.py data/mapillary data/mapillary/cityscapes_trainIdLabel --train_id
# you do not need to convert BDD100K. It is already in the correct format.
```


## Preparing Visual Foundation Models

* **Download:** Download pre-trained weights of VFMs and place them in the `checkpoints` directory without changing the file name. You only need to download one of the following models depending on which one you want to run:

| Model  | Download Link                                                | filename                   | Size  |
| ------ | ------------------------------------------------------------ | -------------------------- | ----- |
| DINOv2 | [DINOv2-ViT-L/14](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth) | dinov2_vitl14_pretrain.pth | 1.2GB |
| EVA02  | [EVA02-ViT-L/14](https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/pt/eva02_L_pt_m38m_p14to16.pt) | eva02_L_pt_m38m_p14to16.pt | 613MB |
| CLIP   | [CLIP-ViT-L/14](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) | ViT-L-14.pt                | 890MB |
| SAM    | [SAM-ViT-H/14](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) | sam_vit_h_4b8939.pth       | 2.4GB |

* **Convert:** Convert pre-trained weights for training or evaluation.

  ```bash
  # convert DINOv2
  python tools/convert_models/convert_dinov2.py checkpoints/dinov2_vitl14_pretrain.pth checkpoints/dinov2_converted.pth
  # convert EVA02
  python tools/convert_models/convert_eva2_512x512.py checkpoints/eva02_L_pt_m38m_p14to16.pth checkpoints/eva02_L_converted.pth
  # convert CLIP
  python tools/convert_models/convert_clip.py checkpoints/ViT-L-14.pt checkpoints/CLIP-ViT-L_converted.pth
  # convert SAM
  python tools/convert_models/convert_sam.py checkpoints/sam_vit_h_4b8939.pth checkpoints/sam_vit_h_converted.pth
  ```

## Training

Start training on a single GPU:

```
python tools/train.py configs/dg/gta2citys/dg_lora_dinov2_ms_masked.py
```

You can also run the script: 

```
./train.sh
```

## Evaluation

  Run the evaluation:

  ```
python tools/test.py \
    configs/dg/gta2citys/dg_lora_dinov2_ms_masked.py \
    <path_to_your_checkpoint> \
    --backbone checkpoints/dinov2_converted.pth
  ```

You can also run the script: 

```
./test.sh
```

## Overview of Important Files

This section provides an overview of the code files related to the model architecture and design:

- **`core/models/backbones`**: This folder contains the implementation of encoder of VFMs, including `dino_v2.py`,`eva_02.py`,`sam_vit.py`,`clip.py`. `lora_backbone.py` implements the lora-based fine-tuning algorithm.

- **`core/models/heads`**: This folder contains the implementation of the head for our VFMNet and MGRNet. `Liner_head.py` implements the head for VFMNet. `VFMHead.py` implements the head for MGRNet.

- **`core/segmentors/Ms_VFM_encoder_decoder.py`**: This file implements our multi-scale training algorithm and the two-stage coarse-to-fine inference algorithm.

  