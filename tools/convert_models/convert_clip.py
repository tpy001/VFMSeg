import torch
import os.path as osp
from collections import OrderedDict
from torch import Tensor
import torch.nn.functional as F
import sys
import numpy as np
import argparse


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("pretrained", type=str)
    args.add_argument("converted", type=str)
    args.add_argument("--kernel", default=16, type=int)
    args.add_argument("--height", default=512, type=int)
    args.add_argument("--embed_dim", default=1024, type=int)
    return args.parse_args()


def convert(pretrained,converted_path,
            resolution,patch_size,embd_dim):
    
    spatial_size = resolution // patch_size
    ps_embed_size = ( (resolution // patch_size) ** 2 + 1,embd_dim )
    if isinstance(pretrained, str):
        checkpoint = (
            torch.jit.load(pretrained, map_location="cpu").float().state_dict()
        )
        print("Load from", pretrained)
        state_dict = {}
        for k in checkpoint.keys():
            if k.startswith("visual."):
                new_k = k.replace("visual.", "")
                state_dict[new_k] = checkpoint[k]

        if "positional_embedding" in state_dict.keys():
            print(
                f'Resize the pos_embed shape from {state_dict["positional_embedding"].shape} to {ps_embed_size}'
            )
            cls_pos = state_dict["positional_embedding"][0:1, :]
            leng = int(state_dict["positional_embedding"][1:,].shape[-2] ** 0.5)
            spatial_pos = F.interpolate(
                state_dict["positional_embedding"][1:,]
                .reshape(1, leng, leng, embd_dim)
                .permute(0, 3, 1, 2),
                size=(spatial_size, spatial_size),
                mode="bilinear",
            )
            spatial_pos = spatial_pos.reshape(
                embd_dim, spatial_size * spatial_size
            ).permute(1, 0)
            positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
            state_dict["positional_embedding"] = positional_embedding
            assert (
                positional_embedding.shape
                == state_dict["positional_embedding"].shape
            )
        conv1 = state_dict["conv1.weight"]
        C_o, C_in, H, W = conv1.shape
        conv1 = torch.nn.functional.interpolate(
            conv1.float(),
            size=(patch_size, patch_size),
            mode="bicubic",
            align_corners=False,
        )
        state_dict["conv1.weight"] = conv1

        torch.save(state_dict, converted_path)


def main():
    args = parse_args()
    pretrained_path = args.pretrained
    converted_path = args.converted
    kernel_conv = args.kernel
    height = args.height
    embed_dim = args.embed_dim

    convert(pretrained=pretrained_path,
            converted_path=converted_path,
            resolution=height,
            patch_size=kernel_conv,
            embd_dim=embed_dim)
    
    return args


# Check if the script is run directly (and not imported)
if __name__ == "__main__":
    main()
