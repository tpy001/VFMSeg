# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from mmseg.models.utils  import resize



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
