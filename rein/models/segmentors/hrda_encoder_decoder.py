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
from .frozen_encoder_decoder import detach_everything

from mmseg.models.utils  import resize
from mmseg.registry import MODELS
from mmseg.models.segmentors import EncoderDecoder
import torch.nn.functional as F
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
class HRDAEncoderDecoder(EncoderDecoder):
    last_train_crop_box = {}

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
                 hr_slide_inference=True,
                 hr_slide_overlapping=True,
                 crop_coord_divisible=1,
                 blur_hr_crop=False,
                 feature_scale=1,
                 data_preprocessor=None):
        self.debug = True
        self.debug_output = dict()
        self.local_iter = 0
        self.feature_scale_all_strs = ['all']
        if isinstance(feature_scale, str):
            assert feature_scale in self.feature_scale_all_strs
        scales = sorted(scales)
        decode_head['scales'] = scales
        decode_head['enable_hr_crop'] = hr_crop_size is not None
        decode_head['hr_slide_inference'] = hr_slide_inference
        super(HRDAEncoderDecoder, self).__init__(
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
        self.hr_slide_inference = hr_slide_inference
        self.hr_slide_overlapping = hr_slide_overlapping
        self.crop_coord_divisible = crop_coord_divisible
        self.blur_hr_crop = blur_hr_crop

        self.orginal_slide_inference = self.test_cfg.get('orginal_slide_inference', False)

        self.log_interval = 250

    def extract_unscaled_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_slide_feat(self, img):
        if self.hr_slide_overlapping:
            h_stride, w_stride = [e // 2 for e in self.crop_size]
        else:
            h_stride, w_stride = self.crop_size
        h_crop, w_crop = self.crop_size
        bs, _, h_img, w_img = img.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        crop_imgs, crop_feats, crop_boxes = [], [], []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_imgs.append(img[:, :, y1:y2, x1:x2])
                crop_boxes.append([y1, y2, x1, x2])
        crop_imgs = torch.cat(crop_imgs, dim=0)
        crop_feats = self.extract_unscaled_feat(crop_imgs)
        # shape: feature levels, crops * batch size x c x h x w

        return {'features': crop_feats, 'boxes': crop_boxes}

    def blur_downup(self, img, s=0.5):
        img = resize(
            input=img,
            scale_factor=s,
            mode='bilinear',
            align_corners=self.align_corners)
        img = resize(
            input=img,
            scale_factor=1 / s,
            mode='bilinear',
            align_corners=self.align_corners)
        return img

    def resize(self, img, s):
        if s == 1:
            return img
        else:
            with torch.no_grad():
                return resize(
                    input=img,
                    scale_factor=s,
                    mode='bilinear',
                    align_corners=self.align_corners)

    def extract_feat(self, img):
        if self.feature_scale in self.feature_scale_all_strs:
            mres_feats = []
            for i, s in enumerate(self.scales):
                if s == 1 and self.blur_hr_crop:
                    scaled_img = self.blur_downup(img)
                else:
                    scaled_img = self.resize(img, s)
                if self.crop_size is not None and i >= 1:
                    scaled_img = crop(
                        scaled_img, HRDAEncoderDecoder.last_train_crop_box[i])
                mres_feats.append(self.extract_unscaled_feat(scaled_img))
            return mres_feats
        else:
            scaled_img = self.resize(img, self.feature_scale)
            return self.extract_unscaled_feat(scaled_img)

    def slide_inference(self, img: Tensor,img_meta: List[dict]) -> Tensor:
        batched_slide = self.test_cfg.get('batched_slide', False)
        if not batched_slide:
            return super().slide_inference(img, img_meta)
        else:
            h_stride, w_stride = self.test_cfg.stride
            h_crop, w_crop = self.test_cfg.crop_size
            batch_size, _, h_img, w_img = img.size()
            out_channels = self.out_channels
            h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
            w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
            preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
            count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
            crop_imgs, crops = [], []
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    crop_img = img[:, :, y1:y2, x1:x2]
                    crop_imgs.append(crop_img)
                    crops.append((y1, y2, x1, x2))
            crop_imgs = torch.cat(crop_imgs, dim=0)
            # img_meta[0]['img_shape'] = crop_img.shape[2:]
            crop_seg_logits = self.encode_decode(crop_imgs, img_meta) # img_meta没用到
            for i in range(len(crops)):
                y1, y2, x1, x2 = crops[i]
                crop_seg_logit = \
                    crop_seg_logits[i * batch_size:(i + 1) * batch_size]
                preds += F.pad(crop_seg_logit,
                            (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
            assert (count_mat == 0).sum() == 0
            if torch.onnx.is_in_onnx_export():
                # cast count_mat to constant while exporting to ONNX
                count_mat = torch.from_numpy(
                    count_mat.cpu().detach().numpy()).to(device=img.device)
            preds = preds / count_mat
            return preds 
    
    def encode_decode(self, img, img_metas, upscale_pred=True):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        if self.orginal_slide_inference:
            feat = self.extract_unscaled_feat(img)
            out =  self.decode_head.head(feat)
        else:
            mres_feats = []
            self.decode_head.debug_output = {}
            for i, s in enumerate(self.scales):
                if s == 1 and self.blur_hr_crop:
                    scaled_img = self.blur_downup(img)
                else:
                    scaled_img = self.resize(img, s)
                if i >= 1 and self.hr_slide_inference:
                    mres_feats.append(self.extract_slide_feat(scaled_img))
                else:
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
        return out

    def _forward_train_features(self, img):
        mres_feats = []
        self.decode_head.debug_output = {}
        assert len(self.scales) <= 2, 'Only up to 2 scales are supported.'
        prob_vis = None
        for i, s in enumerate(self.scales):
            if s == 1 and self.blur_hr_crop:
                scaled_img = self.blur_downup(img)
            else:
                scaled_img = resize(
                    input=img,
                    scale_factor=s,
                    mode='bilinear',
                    align_corners=self.align_corners)
            if self.crop_size is not None and i >= 1:
                crop_box = get_crop_bbox(*scaled_img.shape[-2:],
                                         self.crop_size,
                                         self.crop_coord_divisible)
                if self.feature_scale in self.feature_scale_all_strs:
                    HRDAEncoderDecoder.last_train_crop_box[i] = crop_box
                self.decode_head.set_hr_crop_box(crop_box)
                scaled_img = crop(scaled_img, crop_box)
            if self.decode_head.debug:
                self.decode_head.debug_output[f'Img {i} Scale {s}'] = \
                    scaled_img.detach()
            mres_feats.append(self.extract_unscaled_feat(scaled_img))
        return mres_feats, prob_vis

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        losses = self.forward_train(inputs,data_samples)
        return losses
    
    def forward_train(self,
                      img,
                      data_samples,
                      seg_weight=None,
                      return_feat=False,
                      return_logits=False):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()

        mres_feats, prob_vis = self._forward_train_features(img)
        for i, s in enumerate(self.scales):
            if return_feat and self.feature_scale in \
                    self.feature_scale_all_strs:
                if 'features' not in losses:
                    losses['features'] = []
                losses['features'].append(mres_feats[i])
            if return_feat and s == self.feature_scale:
                losses['features'] = mres_feats[i]
                break

        loss_decode = self._decode_head_forward_train(mres_feats, data_samples)

        losses.update(loss_decode)

        if self.decode_head.debug and prob_vis is not None:
            self.decode_head.debug_output['Crop Prob.'] = prob_vis

        if self.with_auxiliary_head:
            raise NotImplementedError

        if self.debug:
            self.debug_output.update(self.decode_head.debug_output)
            seg_debug = {}
            seg_debug['Source'] = self.debug_output
            if self.local_iter % self.log_interval == 0:
                batchsize = img.shape[0]
                means = self.data_preprocessor.mean.to(img.device)
                stds = self.data_preprocessor.std.to(img.device)
                self.draw_img(seg_debug,batchsize,means,stds)

        self.local_iter += 1
        return losses

    def forward_with_aux(self, img, img_metas):
        assert not self.with_auxiliary_head
        mres_feats, _ = self._forward_train_features(img)
        out = self.decode_head.forward(mres_feats)
        # out = resize(
        #     input=out,
        #     size=img.shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        return {'main': out}

    def draw_img(self, debug_output,batch_size,means,stds):
        out_dir = os.path.join(self.train_cfg['work_dir'],
                                   'class_mix_debug')
        os.makedirs(out_dir, exist_ok=True)

        for j in range(batch_size):
            # rows, cols = 3, len(debug_output)
            rows, cols = 1, 9
            fig, axs = plt.subplots(
                rows,
                cols,
                figsize=(3 * cols, 3 * rows),
                gridspec_kw={
                    'hspace': 0.1,
                    'wspace': 0,
                    'top': 0.85,
                    'bottom': 0,
                    'right': 1,
                    'left': 0
                },
            )
            for k1, (n1, outs) in enumerate(debug_output.items()):
                for k2, (n2, out) in enumerate(outs.items()):
                    if out.shape[1] == 3:
                        vis = torch.clamp(
                            denorm(out, means, stds), 0, 1)
                        # subplotimg(axs[k1][k2], vis[j], f'{n1} {n2}')
                        subplotimg(axs[k2], vis[j], f'{n1} {n2}')

                    else:
                        if out.ndim == 3:
                            args = dict(cmap='cityscapes')
                        else:
                            args = dict(cmap='gray', vmin=0, vmax=1)
                        #subplotimg(axs[k1][k2], out[j], f'{n1} {n2}',
                               #      **args)
                        subplotimg(axs[k2], out[j], f'{n1} {n2}',**args)
            for ax in axs.flat:
                ax.axis('off')
            plt.savefig(
                os.path.join(out_dir,
                                f'{(self.local_iter + 1):06d}_{j}_s.png'))
            plt.close()


@MODELS.register_module()
class FrozenHRDAEncoderDecoder(HRDAEncoderDecoder):
    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    def extract_unscaled_feat(self, img):
        with torch.no_grad():
            x = self.backbone(img)
            x = detach_everything(x)
        if self.with_neck:
            x = self.neck(x)
        return x
    