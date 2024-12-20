
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
        self.work_dir = train_cfg['work_dir']
        self.debug_interval = self.log_config['img_interval']

        self.means = torch.tensor(model_cfg['data_preprocessor'].mean)
        self.stds = torch.tensor(model_cfg['data_preprocessor'].std)
        self.log_iter = 0

    def get_label(self,data_samples: SampleList):
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
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):

        if any(key.startswith("model.") for key in state_dict.keys()):
            new_dict = {}
            for key, value in state_dict.items():
                if key.startswith("model."):
                    new_key = key.replace("model.", "")
                else:
                    new_key = key
                new_dict[new_key] = value

            dino_dict = {}
            dino_path = "/data3/tangpeiyuan/code/Rein-train/checkpoints/dinov2_converted.pth"
            dino_state_dict = torch.load(dino_path, map_location='cpu')
            for key, value in dino_state_dict.items():
                new_key = "backbone." + key
                dino_dict[new_key] = value

            new_dict.update(dino_dict)
        
            self.model.load_state_dict(new_dict, strict=strict)
        else:
            return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
    