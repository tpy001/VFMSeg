# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from mmseg.models.utils  import resize
from .Transformer import SpatialTransformer
from mmseg.models.decode_heads import SegformerHead
from copy import deepcopy
from mmseg.models.losses import accuracy


def crop(img, crop_bbox):
    """Crop from ``img``"""
    crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
    if img.dim() == 4:
        img = img[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
    elif img.dim() == 3:
        img = img[:, crop_y1:crop_y2, crop_x1:crop_x2]
    elif img.dim() == 2:
        img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    else:
        raise NotImplementedError(img.dim())
    return img

@MODELS.register_module()
class DINOhead(SegformerHead):
    def __init__(self, 
                 n_heads=8, d_head=64,
                 depth=1, dropout=0.1, context_dim=19,
                 **kwargs):
        super().__init__(**kwargs)

        self.transformer_blocks = SpatialTransformer(in_channels= self.channels,
                                                     n_heads=n_heads, 
                                                     d_head=d_head, 
                                                     depth=depth, 
                                                     dropout=dropout, 
                                                     context_dim=context_dim)
        
        #self.hr_crop_box = None

    #def set_crop_box(self,box):
        #self.hr_crop_box = box

    def loss(self, inputs, seg_label,context=None,return_logits=False) -> dict:
        # inputs: 64x64
        # seg_label: 512x512
        # context: 64x64
        seg_logits = self.forward(inputs,context)
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        losses = dict()
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in losses:
                losses[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                losses[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)

        losses['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        
        if return_logits:
            return losses,seg_logits
        else:
            return losses
    
    def forward(self, inputs, seg_logits = None):
        # inputs: 64x64
        # seg_label: 512x512
        # context: 64x64
        if seg_logits is None:
            return super().forward(inputs)
        else:
            inputs = self._transform_inputs(inputs)
            outs = []
            for idx in range(len(inputs)):
                x = inputs[idx]
                conv = self.convs[idx]
                outs.append(conv(x))
            out = self.fusion_conv(torch.cat(outs, dim=1))

            context = seg_logits # when testing
            '''crop_box = self.resize_box(ratio=32) # when training
            context = crop(seg_logits,crop_box)
            context = resize(
                            context,
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners)'''
            out = self.transformer_blocks(out,context)

            out = self.cls_seg(out)

            return out

        




