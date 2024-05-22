from ..Wrapper import SegmentWrapper
from mmseg.utils import (ForwardResults, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList)
from typing import List, Tuple
from torch import Tensor
from mmseg.registry import MODELS
import torch
from ..utils import strong_transform

@MODELS.register_module()
class DomainGeneral(SegmentWrapper):
    def __init__(self,model_cfg,train_cfg,**kwargs):
        super().__init__(model_cfg,train_cfg)
        
        pass
    
    def source_loss(self,inputs: Tensor, data_samples: SampleList) -> dict:
        return super().loss(inputs, data_samples)
    
    def mask_loss(self,inputs: Tensor, data_samples: SampleList) -> dict:
        image = torch.stack(inputs)
        label = get_label(data_samples)
        

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        src_loss = self.source_loss(inputs, data_samples)
        return src_loss
        # mask_loss = self.mask_loss(inputs, data_samples)
        # loss = src_loss + mask_loss