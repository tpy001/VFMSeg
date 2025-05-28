import torch
import torch.nn as nn

def print_trainable_parameters(model):
    """
    打印模型的可训练参数信息，包括参数名称、形状和每层的参数数量，并计算总的可训练参数数量。
    
    Args:
        model (torch.nn.Module): 需要统计参数的 PyTorch 模型。
    """
    print("Trainable parameters:")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            print(f"Name: {name} \t Shape: {param.shape} \t Number of parameters: {param_count}")
    
    print(f"Total trainable parameters: {total_params}")

