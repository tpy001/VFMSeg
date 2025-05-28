# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from mmseg.models.utils  import resize
from mmseg.models.losses import accuracy



@MODELS.register_module()
class LinearHead(BaseDecodeHead):

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        '''self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))'''
        
        self._channels = self.in_channels[0]
        self.fusion_conv = ConvModule(
            in_channels=self._channels * num_inputs,
            out_channels=self._channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(self._channels, self._channels//2, kernel_size=2, stride=2),
            nn.SyncBatchNorm(self._channels//2),
            nn.GELU(),
            nn.ConvTranspose2d(self._channels//2, self._channels//4, kernel_size=2, stride=2),
            nn.GELU(),
        )

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        '''outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))'''

        out = self.fusion_conv(torch.cat(inputs, dim=1))

        out = self.output_upscaling(out)

        out = self.cls_seg(out)

        return out

    def loss(self, inputs, seg_label,return_logits=False) -> dict:
        # inputs: 64x64
        # seg_label: 512x512
        seg_logits = self.forward(inputs)
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
