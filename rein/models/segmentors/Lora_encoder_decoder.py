from typing import List
import torch
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.models.segmentors import EncoderDecoder
from typing import Iterable
from peft import LoraConfig,get_peft_model
from ..backbones.utils import set_requires_grad, set_train
from ..backbones.utils import get_trainable_params

@MODELS.register_module()
class LoraBackboneEncoderDecoder(EncoderDecoder):
    def __init__(self,checkpoint,Lora_config,**kwargs):
        super().__init__(**kwargs)

        self.Lora_config = LoraConfig(
                r=Lora_config['r'],
                lora_alpha=Lora_config['lora_alpha'],
                target_modules=Lora_config['target_modules'],
                lora_dropout=Lora_config['lora_dropout'],
                bias='none',
        )
        self.backbone = get_peft_model(self.backbone,self.Lora_config)          # add lora layer to backbone
        self.load_pretrained_backbone(checkpoint,Lora_config['target_modules']) # load pretrained weights for backbone
       
    
    def load_pretrained_backbone(self,checkpoint,target_modules):
        original_weight = torch.load(checkpoint,map_location='cpu')
        new_weight = {}
        for name,weight in original_weight.items():
            for target_name in target_modules:
                if target_name in name:
                    name = name.replace(target_name,target_name+'.base_layer')
                new_weight[name] = weight
        self.backbone.base_model.model.load_state_dict(new_weight,strict=False)

    def train(self, mode: bool = True):
        self.decode_head.train(mode)
        if not mode:
            return super().train(mode)
        set_requires_grad(self.backbone, ["lora"])
        set_train(self.backbone, ["lora"])
        # get_trainable_params(self)
   
