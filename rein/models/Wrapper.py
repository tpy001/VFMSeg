
from mmseg.models.segmentors import BaseSegmentor
from mmseg.utils import (ForwardResults, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList)
from mmseg.registry import MODELS
from typing import List, Tuple
from torch import Tensor
import torch

@MODELS.register_module()
class SegmentWrapper(BaseSegmentor):
    def __init__(self,model_cfg,train_cfg,**kwargs):
        super().__init__(data_preprocessor=model_cfg['data_preprocessor'])
        
        self.work_dir = train_cfg['work_dir']
        self.log_config = train_cfg['log_config']
  
        self.model = MODELS.build(model_cfg)

    def get_label(data_samples: SampleList):
        label = [ data_samples[i].gt_sem_seg.data for i in range(len(data_samples))]
        return torch.stack(label)

    def extract_feat(self, inputs: Tensor) -> bool:
        """Placeholder for extract features from images."""
        return self.model.extract_feat(inputs)
    
    def encode_decode(self, inputs: Tensor, batch_data_samples: SampleList):
        """Placeholder for encode images with backbone and decode into a
        semantic segmentation map of the same size as input."""
        return self.model.encode_decode(inputs, batch_data_samples)

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        return self.model.loss(inputs, data_samples)

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing."""
        return self.model.predict(inputs, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        return self.model._forward(inputs, data_samples)
    
