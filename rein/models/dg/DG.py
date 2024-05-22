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

@MODELS.register_module()
class DomainGeneral(SegmentWrapper):
    def __init__(self,model_cfg,train_cfg,**kwargs):
        super().__init__(model_cfg,train_cfg)
        
        pass
    
    def source_loss(self,inputs: Tensor, data_samples: SampleList) -> dict:
        return super().loss(inputs, data_samples)
    
    def mask_loss(self,inputs: Tensor, data_samples: SampleList) -> dict:
        image = torch.stack(inputs)
        label = self.get_label(data_samples)
    
    def get_pred_label(self,img):
            with torch.no_grad():
                seg_data_samples = self.model.forward(img,mode="predict")
            return [data_sample.pred_sem_seg.data for data_sample in seg_data_samples]
    
    def debug(self,inputs,data_samples):
       
        
        gt_semantic_seg = self.get_label(data_samples)
        batch_size = inputs.shape[0]
        means = self.means.unsqueeze(1).unsqueeze(1).to(inputs.device)
        stds = self.stds.unsqueeze(1).unsqueeze(1).to(inputs.device)
        with torch.no_grad():
          
            out_dir = os.path.join(self.work_dir,'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            
            vis_img = torch.clamp(denorm(inputs, means, stds), 0, 1)
            vis_src_pred = self.get_pred_label(inputs)


            for j in range(batch_size):
                plt.ioff()
                rows, cols = 2, 4
                fig, axs = plt.subplots(rows,cols,figsize=(3 * cols, 3 * rows),gridspec_kw={'hspace': 0.1,'wspace': 0,'top': 0.95,'bottom': 0,'right': 1,'left': 0},)
                
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[0][1],gt_semantic_seg[j],'Source Seg GT',cmap='cityscapes')
                subplotimg(axs[0][2],vis_src_pred[j],'Source Img Pred',cmap='cityscapes')

                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                f'{(self.log_iter + 1):06d}_{j}.png'))
                plt.close()

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        src_loss = self.source_loss(inputs, data_samples)
        if self.log_iter % self.debug_interval == 0:
            self.debug(inputs, data_samples)
        self.log_iter += 1
        return src_loss
    
        # mask_loss = self.mask_loss(inputs, data_samples)
        # loss = src_loss + mask_loss