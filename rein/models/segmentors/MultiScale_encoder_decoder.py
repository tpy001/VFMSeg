# Obtained from: https://github.com/lhoyer/HRDA
# Modifications:
# - Add return_logits flag
# - Add upscale_pred flag
# - Update debug_output system
# ---------------------------------------------------------------
# Copyright (c) 2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from mmseg.utils import SampleList
import numpy as np
import torch
from torch import Tensor
from typing import List

from mmseg.models.utils  import resize
from mmseg.registry import MODELS
from mmseg.models.segmentors import EncoderDecoder
import torch.nn.functional as F
from mmseg.utils import add_prefix
from copy import deepcopy
from mmseg.structures import SegDataSample
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ...utils import subplotimg

def denorm(img, mean, std):
    return img.mul(std).add(mean) / 255.0


def get_crop_bbox(img_h, img_w, crop_size, divisible=1):
    """Randomly get a crop bounding box."""
    assert crop_size[0] > 0 and crop_size[1] > 0
    if img_h == crop_size[-2] and img_w == crop_size[-1]:
        return (0, img_h, 0, img_w)
    margin_h = max(img_h - crop_size[-2], 0)
    margin_w = max(img_w - crop_size[-1], 0)
    offset_h = np.random.randint(0, (margin_h + 1) // divisible) * divisible
    offset_w = np.random.randint(0, (margin_w + 1) // divisible) * divisible
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

    return crop_y1, crop_y2, crop_x1, crop_x2

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
class MultiScaleEncoderDecoder(EncoderDecoder):

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 scales=[1],
                 hr_crop_size=None,
                 crop_coord_divisible=1,
                 feature_scale=1,
                 data_preprocessor=None,
                 debug=False,
                 debug_interval=100):
        self.local_iter = 0
        self.feature_scale_all_strs = ['all']
        if isinstance(feature_scale, str):
            assert feature_scale in self.feature_scale_all_strs
        scales = sorted(scales)
        super(MultiScaleEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.scales = scales
        self.feature_scale = feature_scale
        self.crop_size = hr_crop_size
        self.crop_coord_divisible = crop_coord_divisible
        self.debug = debug
        self.debug_interval = self.train_cfg.log_config.img_interval

        self.means = self.data_preprocessor.mean
        self.stds = self.data_preprocessor.std
        

    def extract_unscaled_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    '''def extract_feat(self, img):
        if self.feature_scale in self.feature_scale_all_strs:
            mres_feats = []
            for i, s in enumerate(self.scales):
                scaled_img = self.resize(img, s)
                if self.crop_size is not None and i >= 1:
                    scaled_img = crop(
                        scaled_img, MultiScaleEncoderDecoder.last_train_crop_box[i])
                mres_feats.append(self.extract_unscaled_feat(scaled_img))
            return mres_feats
        else:
            scaled_img = self.resize(img, self.feature_scale)
            return self.extract_unscaled_feat(scaled_img)'''


    
    '''def encode_decode(self, img, img_metas, upscale_pred=True):
        mres_feats = []
        self.decode_head.debug_output = {}
        for i, s in enumerate(self.scales):
            scaled_img = self.resize(img, s)
            mres_feats.append(self.extract_unscaled_feat(scaled_img))
            if self.decode_head.debug:
                self.decode_head.debug_output[f'Img {i} Scale {s}'] = \
                    scaled_img.detach()
        out = self.decode_head.forward_test(mres_feats)
        if upscale_pred:
            out = resize(
                input=out,
                size=img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        return out'''

    def _forward_train_features(self, img):
        mres_feats = []
        assert len(self.scales) <= 2, 'Only up to 2 scales are supported.'
        for i, s in enumerate(self.scales):
            scaled_img = resize(
                input=img,
                scale_factor=s,
                mode='bilinear',
                align_corners=self.align_corners)
            if self.crop_size is not None and i >= 1:
                crop_box = get_crop_bbox(*scaled_img.shape[-2:],
                                         self.crop_size,
                                         self.crop_coord_divisible)
                self.decode_head.set_crop_box(crop_box) # save the crop box
                scaled_img = crop(scaled_img, crop_box)
            mres_feats.append(self.extract_unscaled_feat(scaled_img))
        return mres_feats

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        losses = self.forward_train(inputs,data_samples)
        return losses

    def forward_train(self,
                      img,
                      data_samples,
                      seg_weight=None,
                      return_feat=False,
                      return_logits=False):
        losses = dict()

        mres_feats = self._forward_train_features(img)
        lr_feats = mres_feats[0]
        hr_feats = mres_feats[1]
        
        seg_label = self.decode_head._stack_batch_gt(data_samples)

        lr_gt_seg = resize(seg_label.float(),
                           scale_factor=0.5,
                           mode='nearest').long()
        hr_gt_seg = crop(seg_label,self.decode_head.hr_crop_box)

        loss_decode_lr,seg_logits = self.decode_head.loss(lr_feats, lr_gt_seg,return_logits=True)
        losses.update(add_prefix(loss_decode_lr, 'decode_lr'))

        crop_box2 = (item//2 for item in self.decode_head.hr_crop_box)
        context = seg_logits.detach()
        context = crop(context,crop_box2)
        context = resize(context,scale_factor=1/8, mode='bilinear', align_corners=self.align_corners)

        loss_decode_hr = self.decode_head.loss(hr_feats, hr_gt_seg,context)
        losses.update(add_prefix(loss_decode_hr, 'decode_hr'))

        if self.local_iter % self.debug_interval == 0:
            self.debug_output(img,data_samples)
        self.local_iter += 1
        return losses
    
    def debug_output(self,img,data_samples):
        self.means = self.means.to(img.device)
        self.stds = self.stds.to(img.device)

        gt_sem_seg = torch.stack([item.gt_sem_seg.data for item in data_samples ])
        crop_box = get_crop_bbox(*img.shape[-2:],self.crop_size,self.crop_coord_divisible)
        lr_img = resize(
                input=img,
                scale_factor=0.5,
                mode='bilinear',
                align_corners=self.align_corners)
        lr_seg = resize(gt_sem_seg.float(),
                        scale_factor=0.5,
                        mode='nearest').int()
        hr_img = crop(img, crop_box)
        hr_seg = crop(gt_sem_seg, crop_box)

        with torch.no_grad():
            lr_feat = self.extract_unscaled_feat(lr_img)
            lr_seg_logits =  self.decode_head(lr_feat)
            lr_pred = resize(lr_seg_logits,size = lr_seg.shape[-2:], mode='bilinear', align_corners=self.align_corners)
            lr_pred = lr_pred.argmax(dim=1, keepdim=True).squeeze()

            crop_box2 = [item// 32 for item in crop_box]
            lr_logits_upsample = crop(lr_seg_logits,crop_box2)
            lr_logits_upsample = resize(lr_logits_upsample,scale_factor=2, mode='bilinear', align_corners=self.align_corners)

            lr_pred_upsample = resize(lr_logits_upsample,size=hr_seg.shape[-2:], mode='bilinear', align_corners=self.align_corners).argmax(dim=1, keepdim=True).squeeze()

            hr_feat = self.extract_unscaled_feat(hr_img)
            hr_pred =  resize(self.decode_head(hr_feat),size=hr_seg.shape[-2:], mode='bilinear', align_corners=self.align_corners)
            hr_pred = hr_pred.argmax(dim=1, keepdim=True).squeeze()
            
            hr_pred_refine = resize(self.decode_head(hr_feat,lr_logits_upsample),size=hr_seg.shape[-2:], mode='bilinear', align_corners=self.align_corners)
            hr_pred_refine = hr_pred_refine.argmax(dim=1, keepdim=True).squeeze()

        out_dir = os.path.join(self.train_cfg.work_dir,'class_mix_debug')
        os.makedirs(out_dir, exist_ok=True)
        
        # 可视化图片
        vis_lr_img = torch.clamp(denorm(lr_img, self.means, self.stds), 0, 1)
        vis_hr_img = torch.clamp(denorm(hr_img, self.means, self.stds), 0, 1)

        batch_size = img.shape[0]

        for j in range(batch_size):
            plt.ioff()
            rows, cols = 2, 5
            fig, axs = plt.subplots(rows,cols,figsize=(3 * cols, 3 * rows),gridspec_kw={'hspace': 0.1,'wspace': 0,'top': 0.95,'bottom': 0,'right': 1,'left': 0},)
            
            subplotimg(axs[0][0], vis_lr_img[j], 'lr img')
            subplotimg(axs[0][1],lr_pred[j],'lr seg pred',cmap='cityscapes')
            subplotimg(axs[0][4],lr_seg[j],'lr seg gt',cmap='cityscapes')
            

            subplotimg(axs[1][0], vis_hr_img[j], 'hr img')
            subplotimg(axs[1][1],lr_pred_upsample[j],'lr seg upscale',cmap='cityscapes')
            subplotimg(axs[1][2], hr_pred[j], 'hr img pred',cmap='cityscapes')
            subplotimg(axs[1][3], hr_pred_refine[j], 'hr img refined',cmap='cityscapes')
            subplotimg(axs[1][4],hr_seg[j],'hr seg gt',cmap='cityscapes')


            '''# 显示每个pseudo的softmax的概率
            subplotimg(axs[3][0],pseudo_prob[j] ,'Pselabel SoftmaxPro',cmap='hot', interpolation='nearest')
            # 显示大于pseudo域值的像素点有哪些
            subplotimg(axs[3][1],ps_large_p[j].int() ,'Pselabel mask',cmap='gray')
            # 计算熵
            log_probabilities = torch.log2(ema_softmax[j])
            entropy_map = -torch.sum(ema_softmax[j] * log_probabilities, dim=0)
            subplotimg(axs[3][2],entropy_map ,'Pseudo label entropy',cmap='hot')'''


            # 显示伪标签的熵
            for ax in axs.flat:
                ax.axis('off')
            plt.savefig(
                os.path.join(out_dir,
                            f'{(self.local_iter + 1):06d}_{j}.png'))
            plt.close()


    def enc_dec(self, inputs,context=None):
        
        x = self.extract_feat(inputs)
        seg_logits = self.decode_head(x,context)
        return seg_logits
    
    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict],mode='ms_slide_inference') -> Tensor:
       
        assert mode in  ['lr_slide_inference','hr_slide_inference','ms_slide_inference']
        if mode == 'lr_slide_inference':
            inputs_lr = resize(inputs, scale_factor=0.5, mode='bilinear', align_corners=self.align_corners)
            lr_seg_logits = super(MultiScaleEncoderDecoder,self).slide_inference(inputs_lr,batch_img_metas)
            seg_logits = resize(lr_seg_logits, scale_factor=2, mode='bilinear', align_corners=self.align_corners)
        elif mode == 'hr_slide_inference':
            seg_logits = super(MultiScaleEncoderDecoder,self).slide_inference(inputs,batch_img_metas)
        elif mode == 'ms_slide_inference':
            inputs_lr = resize(inputs, scale_factor=0.5, mode='bilinear', align_corners=self.align_corners)  # 512,1024
            lr_seg_logits = super(MultiScaleEncoderDecoder,self).slide_inference(inputs_lr,batch_img_metas) # 512,1024
            lr_seg_logits = resize(lr_seg_logits, scale_factor=2, mode='bilinear', align_corners=self.align_corners) # 1024,2048

            h_stride, w_stride = self.test_cfg.stride
            h_crop, w_crop = self.test_cfg.crop_size
            batch_size, _, h_img, w_img = inputs.size()
            out_channels = self.out_channels
            h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
            w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
            preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
            count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
                
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_box = (y1,y2, x1,x2)  # 512,512
                    crop_img = crop(inputs,crop_box) # 512,512
                    context = crop(lr_seg_logits, crop_box) # 512,512
                    context = resize(context, scale_factor=1/16, mode='bilinear', align_corners=self.align_corners) # 32x32

                    self.decode_head.set_crop_box(crop_box)
                    
                    # with shape [N, C, H, W]
                    crop_seg_logit = self.enc_dec(crop_img, context)
                    crop_seg_logit = resize(crop_seg_logit, size=crop_img.shape[2:], mode='bilinear', align_corners=self.align_corners)
                    preds += F.pad(crop_seg_logit,
                                (int(x1), int(preds.shape[3] - x2), int(y1),
                                    int(preds.shape[2] - y2)))

                    count_mat[:, :, y1:y2, x1:x2] += 1
            assert (count_mat == 0).sum() == 0
            seg_logits = preds / count_mat

        return seg_logits



        
    