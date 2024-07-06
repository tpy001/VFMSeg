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

        transformer['img_feat_dim'] = self.channels
        self.query_dim = transformer['query_dim']

        self.activation = nn.GELU()

        self.fuse_conv = nn.Sequential(
            # nn.Conv2d(self.in_channels[0]* num_inputs, self.in_channels[0], kernel_size=1, stride=1),
            nn.Conv2d(self.in_channels[0]* num_inputs, self.channels, kernel_size=1, stride=1),
            nn.GroupNorm(num_groups=32,num_channels= self.channels),
            self.activation,
        )

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(self.query_dim , self.query_dim  // 2, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=32,num_channels=self.query_dim  // 2),
            self.activation,
            nn.ConvTranspose2d(self.query_dim  //2 , self.query_dim  // 4, kernel_size=2, stride=2),
            self.activation
        )
        
        # self.conv_seg = nn.Conv2d(self.query_dim  // 4, self.out_channels, kernel_size=1)
        
        self.seg_logits_embed = nn.Sequential(
            nn.Conv2d(19, self.channels // 4, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=32,num_channels=self.channels // 4),
            self.activation,
            
            nn.Conv2d(self.channels // 4, self.channels // 2, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups=32,num_channels=self.channels // 2),
            self.activation,

            nn.Conv2d(self.channels // 2, self.channels, kernel_size=1, stride=1),
            nn.GroupNorm(num_groups=32,num_channels=self.channels),
        )

        self.transformer_decoder = MODELS.build(transformer)

        '''batch_size = 2
        H,W = 32,32
        self.query = nn.Parameter(torch.randn(batch_size, self.query_dim, H,W))
        self.pos_enc = nn.Parameter(torch.randn(batch_size, self.query_dim, H,W))'''

        
        

    def forward(self,inputs,seg_logits,query=None):
        inputs = self._transform_inputs(inputs)
        seg_logits = resize(
                input=seg_logits,
                size=(inputs[0].shape[2] * 4, inputs[0].shape[3] * 4),
                mode='bilinear',
                align_corners=self.align_corners)    # 128 x128
        
        img_feats = self.fuse_conv(torch.cat(inputs, dim=1))

        seg_logits_embed = self.seg_logits_embed(seg_logits)    

         # img_feats = torch.cat((img_feats,seg_logits_embed),dim=1)

        # img_feats = img_feats + seg_logits_embed

        
        #query = self.query + self.pos_enc
        # out = self.transformer_decoder(query,img_feats)

        # out = self.transformer_decoder(seg_logits_embed,img_feats)
        out = self.transformer_decoder(img_feats,seg_logits_embed)



        # out = self.output_upscaling(out)

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


