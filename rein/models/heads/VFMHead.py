# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from mmseg.models.utils  import resize
from mmseg.models.losses import accuracy


@MODELS.register_module()
class VFMHead(BaseDecodeHead):

    def __init__(self, transformer,interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)
        
        self._channels = self.in_channels[0]
        self.fusion_conv = ConvModule(
            in_channels=self._channels * num_inputs,
            out_channels=self._channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        
        self.transformer_decoder = MODELS.build(transformer)

        self.seg_logits_embed = ConvModule(
                in_channels=19,
                out_channels=transformer['query_dim'],
                kernel_size=1,
                norm_cfg=self.norm_cfg)
        
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer['query_dim'], transformer['query_dim']//2, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=32, num_channels=transformer['query_dim']//2),
            nn.GELU(),
        )

    def forward(self,inputs,seg_logits,query=None):
        inputs = self._transform_inputs(inputs)
        img_feats = self.fusion_conv(torch.cat(inputs, dim=1))

        seg_logits_embed = self.seg_logits_embed(seg_logits)
        out = self.transformer_decoder(img_feats,seg_logits_embed,query)

        out = self.output_upscaling(out)
        out = self.cls_seg(out)

        return out
    
    def loss(self,inputs,seg_logits_embed,
             seg_label,query=None,return_logits=False) -> dict:
        # inputs: 64x64
        # seg_label: 512x512
        seg_logits = self.forward(inputs,seg_logits_embed,query)
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


