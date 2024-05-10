# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
from mmseg.registry import MODELS



import torch
from mmseg.models import EncoderDecoder
from copy import deepcopy
from mmseg.models import  build_head
from .dacs_transforms import (denorm, get_class_masks,strong_transform)
import numpy as np
import random
import os
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ...utils import subplotimg
from mmseg.models.utils  import resize
from mmseg.models.losses import accuracy
from torch import Tensor
from typing import List
from typing import Iterable
from torch.nn.modules.dropout import _DropoutNd
from timm.models.layers import DropPath

def detach_everything(everything):
    if isinstance(everything, Tensor):
        return everything.detach()
    elif isinstance(everything, Iterable):
        return [detach_everything(x) for x in everything]
    else:
        return everything
    
@MODELS.register_module()
class DACS(EncoderDecoder):

    """ def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False """

    ''' def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        with torch.no_grad():
            x = self.backbone(inputs)
            x = detach_everything(x)
        if self.with_neck:
            x = self.neck(x)
        return x '''

    def __init__(self, **cfg):
        super().__init__(
            backbone=cfg['backbone'],
            decode_head=cfg['decode_head'],
            train_cfg=cfg['train_cfg'],
            test_cfg=cfg['test_cfg'],
            data_preprocessor=cfg['data_preprocessor']
        )
    
        self.local_iter = 0
        
        self.alpha=cfg['alpha']
        self.pseudo_threshold=cfg['pseudo_threshold']
        self.psweight_ignore_top=cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom=cfg['pseudo_weight_ignore_bottom']
        self.mix=cfg['mix']
        self.blur=cfg['blur']
        self.color_jitter_s=cfg['color_jitter_strength']
        self.color_jitter_p=cfg['color_jitter_probability']
        self.debug_img_interval=cfg['debug_img_interval']
        self.print_grad_magnitude=cfg['print_grad_magnitude']

        self.mean = None
        self.std = None
        self.work_dir = cfg['work_dir']

        self.ema_head = build_head(deepcopy(cfg['decode_head']))

    def init_log_vars(self,source=True,target=None):
        log_vars = OrderedDict()
        log_vars['total_loss'] = torch.tensor(0.0)

        if source is not None:
            log_vars['decode.loss_src'] = torch.tensor(0.0)
          

        if target is not None:
            log_vars['decode.loss_tgt'] = torch.tensor(0.0)
          

        return log_vars
    
    def get_ema_model(self):
        return self.ema_head
    
    def get_model(self):
        return self
    
    def get_trainable_param(self,model):
        param = list(model.decode_head.parameters())
        if model.with_auxiliary_head:
            param += list(model.auxiliary_head.parameters())
        if model.with_neck:
            param += list(model.neck.parameters())
        return param

    def _init_ema_weights(self):
        self.get_ema_model().requires_grad_(False)
        for param in self.get_ema_model().parameters():
            param.detach_()

        mp = self.get_trainable_param(self.get_model())
        mcp = list( self.get_ema_model().parameters() )

        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        mp = self.get_trainable_param(self.get_model())
        mcp = list( self.get_ema_model().parameters() )
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(mcp,mp):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]
                
    # 获取模型预测的label
    def get_pred_label(self,img):
        with torch.no_grad():
            seg_data_samples = self.forward(img,mode="predict")
        return [data_sample.pred_sem_seg.data for data_sample in seg_data_samples]
    
    def train_and_update(self,optim_wrapper, log_vars,img,img_data_samples,loss_weight=1,suffix=""):
        losses = self.forward(img,img_data_samples,mode='loss')
        parsed_losses, _ = self.parse_losses(losses)  # type: ignore
        parsed_losses = parsed_losses * loss_weight
        
        log_vars[f'decode.loss_{suffix}'] = parsed_losses.item()
        log_vars[f'total_loss'] += parsed_losses.item()

        final_loss = optim_wrapper.scale_loss(parsed_losses)
        optim_wrapper.backward(final_loss)
    

    def train_step(self, data,optim_wrapper):
        # Enable automatic mixed precision training context.
        self.mean = self.data_preprocessor.mean
        self.std  = self.data_preprocessor.std

        log_vars = {}
        """with optim_wrapper.optim_context(self):
            img = data['img']
            target_img = data['target_img']
            img = self.data_preprocessor(img, True)
            target_img = self.data_preprocessor(target_img, True)
            losses = self._run_forward(img, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars"""


        with optim_wrapper.optim_context(self):
            img = data['img']
            target_img = data['target_img']
            img = self.data_preprocessor(img, True)
            target_img = self.data_preprocessor(target_img, True)
            log_vars = self.forward_train(img,target_img,optim_wrapper) 
            
            optim_wrapper.step()
            optim_wrapper.zero_grad() # type: ignore
        return log_vars
        
    
    def forward_train(self, img_dict, target_img_dict, optim_wrapper=None):
        img = img_dict['inputs']
        target_img = target_img_dict['inputs']
        img_data_samples = img_dict['data_samples']
        tgt_data_samples = target_img_dict['data_samples']
        
        batch_size,_,W,H = img.shape
        gt_semantic_seg = [ img_dict['data_samples'][i].gt_sem_seg.data for i in range(batch_size)]
        gt_semantic_seg = torch.stack(gt_semantic_seg)


        dev = img.device

        log_vars = self.init_log_vars(target=True)
       

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()

        if self.local_iter > 0:
            self._update_ema(self.local_iter)

        means, stds = self.mean,self.std

        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }



        # 训练 source image
        self.train_and_update(optim_wrapper,log_vars,img,img_data_samples,suffix="src") 
        
        with torch.no_grad():
            '''for m in self.get_ema_model().modules():
                if isinstance(m, _DropoutNd):
                    m.training = False
                if isinstance(m, DropPath):
                    m.training = False'''

            batch_img_metas = [
                data_sample.metainfo for data_sample in tgt_data_samples
            ]

            x = self.extract_feat(target_img)
            ema_logits = self.get_ema_model().predict(x, batch_img_metas,self.test_cfg)
           
            ema_softmax = torch.softmax(ema_logits, dim=1)
            pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
            ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
            ps_size = np.size(np.array(pseudo_label.cpu()))
            ps_large_p_ratio = torch.sum(ps_large_p).item() / ps_size
            pseudo_weight = ps_large_p_ratio 
            # gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)


        """if self.psweight_ignore_top > 0:
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0""" 
        

        # gt_semantic_seg:   [batchsize,1,512,512]
        # pseudo_label:  [batchsize,512,512]
        # Apply mixing
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)
        # mix_weight = torch.ones_like(pseudo_weight)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])),
                mix="cut_mix")
            """_, mix_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))"""
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)

        mix_data_samples = deepcopy(tgt_data_samples)
        for i in range(batch_size):
            mix_data_samples[i].gt_sem_seg.data = mixed_lbl[i].unsqueeze(0)

        # train mixed image
        # self.train_and_update(optim_wrapper,log_vars,mixed_img,mix_data_samples,loss_weight=pseudo_weight,suffix="tgt") 
        self.train_and_update(optim_wrapper,log_vars,mixed_img,mix_data_samples,suffix="tgt") 


       
        
        # 绘制用于debug的图片
        with torch.no_grad():
            if self.local_iter % self.debug_img_interval == 0:
                out_dir = os.path.join(self.work_dir,'class_mix_debug')
                os.makedirs(out_dir, exist_ok=True)
                
                # 可视化图片
                vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
                vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
                vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)

                # 可视化模型预测的结果
                vis_src_pred = self.get_pred_label(img)
                vis_mixed_pred = self.get_pred_label(mixed_img)


                for j in range(batch_size):
                    plt.ioff()
                    rows, cols = 4, 5
                    fig, axs = plt.subplots(rows,cols,figsize=(3 * cols, 3 * rows),gridspec_kw={'hspace': 0.1,'wspace': 0,'top': 0.95,'bottom': 0,'right': 1,'left': 0},)
                    
                    subplotimg(axs[0][0], vis_img[j], 'Source Image')
                    subplotimg(axs[0][1],gt_semantic_seg[j],'Source Seg GT',cmap='cityscapes')


                    # subplotimg(axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                    subplotimg(axs[1][4],vis_src_pred[j],'Source Img Pred',cmap='cityscapes')
                    

                    subplotimg(axs[2][0], vis_trg_img[j], 'Target Image')
                    subplotimg(axs[2][1],pseudo_label[j],'Target Seg (Pseudo) GT',cmap='cityscapes')
                    subplotimg(axs[2][2], vis_mixed_img[j], 'Mixed Image')
                    subplotimg(axs[2][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
                    subplotimg(axs[2][4],vis_mixed_pred[j],'Mixed Img Pred',cmap='cityscapes')
                   


                    # 显示每个pseudo的softmax的概率
                    subplotimg(axs[3][0],pseudo_prob[j] ,'Pselabel SoftmaxPro',cmap='hot', interpolation='nearest')
                    # 显示大于pseudo域值的像素点有哪些
                    subplotimg(axs[3][1],ps_large_p[j].int() ,'Pselabel mask',cmap='gray')
                    # 计算熵
                    log_probabilities = torch.log2(ema_softmax[j])
                    entropy_map = -torch.sum(ema_softmax[j] * log_probabilities, dim=0)
                    subplotimg(axs[3][2],entropy_map ,'Pseudo label entropy',cmap='hot')


                    # 显示伪标签的熵
                    for ax in axs.flat:
                        ax.axis('off')
                    plt.savefig(
                        os.path.join(out_dir,
                                    f'{(self.local_iter + 1):06d}_{j}.png'))
                    plt.close()
        self.local_iter += 1

        return log_vars 
    

