import torch

from mmseg.registry import MODELS
from typing import Iterable
from peft import LoraConfig,get_peft_model
from .utils import set_requires_grad, set_train
from mmengine.model import BaseModule


@MODELS.register_module()
class LoRABackbone(BaseModule):
    def __init__(self,model,checkpoint,Lora_config,**kwargs):
        super().__init__(**kwargs)

        backbone_model = MODELS.build(model)
        self.Lora_config = LoraConfig(
                r=Lora_config['r'],
                lora_alpha=Lora_config['lora_alpha'],
                target_modules=Lora_config['target_modules'],
                lora_dropout=Lora_config['lora_dropout'],
                bias='none',
        )
        self.model = get_peft_model(backbone_model,self.Lora_config)          # add lora layer to backbone
        self.load_pretrained_backbone(checkpoint,Lora_config['target_modules']) # load pretrained weights for backbone
       
    
    def load_pretrained_backbone(self,checkpoint,target_modules):
        original_weight = torch.load(checkpoint,map_location='cpu')
        new_weight = {}
        for name,weight in original_weight.items():
            for target_name in target_modules:
                if target_name in name:
                    name = name.replace(target_name,target_name+'.base_layer')
                new_weight[name] = weight
        self.model.base_model.model.load_state_dict(new_weight,strict=False)

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self.model, ["lora"])
        set_train(self.model, ["lora"])

    def forward(self, x):
        return self.model(x)
   
