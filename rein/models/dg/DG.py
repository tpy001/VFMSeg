from ..Wrapper import SegmentWrapper
from mmseg.utils import (ForwardResults, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList)
from typing import List, Tuple
from torch import Tensor
from mmseg.registry import MODELS
import torch
from ..utils import strong_transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ...utils import subplotimg
import os
from ..uda import denorm
import random
from ..utils.masking_transforms import BlockMaskGenerator

@MODELS.register_module()
class DomainGeneral(SegmentWrapper):
    def __init__(self,model_cfg,train_cfg,**kwargs):
        super().__init__(model_cfg,train_cfg)
        
        self.color_jitter_s = kwargs['color_jitter_strength']
        self.color_jitter_p = kwargs['color_jitter_probability']
        self.blur = kwargs['blur']    
    
    def source_loss(self,inputs: Tensor, data_samples: SampleList) -> dict:
        losses = super().loss(inputs, data_samples)
        parsed_losses, log_vars = self.parse_losses(losses)  
        parsed_losses.backward()
        return log_vars
    
    def mask_loss(self,inputs: Tensor, data_samples: SampleList,acc,return_masked_img=False) -> dict:
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': self.means[0].unsqueeze(0),  # assume same normalization
            'std': self.stds[0].unsqueeze(0)
        }
        inputs_aug,_ = strong_transform(strong_parameters,data=inputs.clone())
        MaskGenerator = BlockMaskGenerator(0.7,64)
        inputs_masked =   MaskGenerator.mask_image(inputs_aug)

        losses = super(DomainGeneral, self).loss(inputs_masked, data_samples)
        lamda_mask = 0.5
        for key,value in losses.items():
            losses[key] = value * acc * lamda_mask
        parsed_losses, log_vars = self.parse_losses(losses) 
        parsed_losses.backward()
        if not return_masked_img:
            return log_vars
        else:
            return log_vars,inputs_masked
    
    def get_pred_label(self,img):
            with torch.no_grad():
                seg_data_samples = self.model.forward(img,mode="predict")
            return [data_sample.pred_sem_seg.data for data_sample in seg_data_samples]
    
    def debug(self,inputs,data_samples,masked_img=None):
        
        gt_semantic_seg = self.get_label(data_samples)
        batch_size = inputs.shape[0]
        means = self.means.unsqueeze(1).unsqueeze(1).to(inputs.device)
        stds = self.stds.unsqueeze(1).unsqueeze(1).to(inputs.device)
        with torch.no_grad():
          
            out_dir = os.path.join(self.work_dir,'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            
            vis_img = torch.clamp(denorm(inputs, means, stds), 0, 1)
            vis_masked_img = torch.clamp(denorm(masked_img, means, stds), 0, 1) if masked_img is not None else None
            vis_src_pred = self.get_pred_label(inputs)
            vis_masked_pred = self.get_pred_label(masked_img) if masked_img is not None else None


            for j in range(batch_size):
                plt.ioff()
                rows, cols = 2, 4
                fig, axs = plt.subplots(rows,cols,figsize=(3 * cols, 3 * rows),gridspec_kw={'hspace': 0.1,'wspace': 0,'top': 0.95,'bottom': 0,'right': 1,'left': 0},)
                
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[0][1],gt_semantic_seg[j],'Source Seg GT',cmap='cityscapes')
                subplotimg(axs[0][2],vis_src_pred[j],'Source Img Pred',cmap='cityscapes')

                subplotimg(axs[1][0], vis_masked_img[j], 'Masked Image')
                subplotimg(axs[1][1],gt_semantic_seg[j],'Masked Seg GT',cmap='cityscapes')
                subplotimg(axs[1][2],vis_masked_pred[j],'Masked Img Pred',cmap='cityscapes')

                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                f'{(self.log_iter + 1):06d}_{j}.png'))
                plt.close()

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        self.means = self.means.to(inputs.device)
        self.stds = self.stds.to(inputs.device)

        enable_mask_loss = True
        return_masked_img = True
        src_log_var = self.source_loss(inputs, data_samples)
        if enable_mask_loss:
            acc = src_log_var['decode.acc_seg'] / 100.0
            if not return_masked_img:
                mask_log_var = self.mask_loss(inputs, data_samples,acc)
            else:
                mask_log_var,masked_img = self.mask_loss(inputs, data_samples,acc,return_masked_img=True)
            for key,value in mask_log_var.items():
                src_log_var['mask_'+key] = value

        if self.log_iter % self.debug_interval == 0:
            if enable_mask_loss and return_masked_img:
                self.debug(inputs, data_samples,masked_img)
            else:
                self.debug(inputs, data_samples)
        self.log_iter += 1
        return src_log_var
    
        # mask_loss = self.mask_loss(inputs, data_samples)
        # loss = src_loss + mask_loss

    def train_step(self, data,optim_wrapper):
        log_vars = {}
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            log_vars = self._run_forward(data, mode='loss')  # type: ignore
            optim_wrapper.step()
            optim_wrapper.zero_grad() # type: ignore
        return log_vars
    