''' 
    This script is used to remove frozen weights of VFM from a given .pth file and only save the lora weights.
    This operation significantly reduces the size of the .pth file.
    How to use this script:
        python tools/convert_legacy_weight.py --input <path_to_input_file> --output <path_to_output_file>
'''
import torch
import argparse
import os

def filter_base_weights(input_path, output_path):
    """
    读取 .pth 文件，删除 LoRA 对应的 base weights，保留 LoRA 权重和其他非 base weights（如 head 的权重），
    并保存到新文件。
    
    Args:
        input_path (str): 输入的 .pth 文件路径
        output_path (str): 输出的新 .pth 文件路径
    """
    # 加载原始权重
    state_dict = torch.load(input_path, map_location='cpu')
    print(f"Loaded weights from {input_path}")
    print(f"Original number of parameters: {len(state_dict)}")

    # 过滤权重
    filtered_state_dict = {}
    base_weight_count = 0
    
    for name, param in state_dict.items():
        # 检查是否是 LoRA 对应的 base weight（通常包含 'base_layer' 或类似的标识）
        if 'base_layer' in name:  # 假设 LoRA 的 base weight 包含 'base_layer'
            base_weight_count += 1
            print(f"Removing base weight: {name} (shape: {tuple(param.shape)})")
            continue
        # 保留其他权重（包括 LoRA 权重和 head 权重）
        else:
            filtered_state_dict[name] = param
            print(f"Keeping: {name} (shape: {tuple(param.shape)})")

    print(f"Number of base weights removed: {base_weight_count}")
    print(f"Number of parameters kept: {len(filtered_state_dict)}")

    # 如果没有找到任何 base weights，给出警告
    if base_weight_count == 0:
        print("Warning: No base weights (e.g., containing 'base_layer') found in the .pth file!")

    # 保存新的权重文件
    torch.save(filtered_state_dict, output_path)
    print(f"Saved filtered weights to {output_path}")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="Remove LoRA base weights from a .pth file.")
    parser.add_argument('--input', type=str, required=True, default="checkpoints/iter_40000.pth",help="Path to the input .pth file")
    parser.add_argument('--output', type=str, required=True, default="test.pth",help="Path to save the output .pth file")
    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file {args.input} does not exist!")

    # 如果输出目录不存在，创建它
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # 执行过滤
    filter_base_weights(args.input, args.output)

if __name__ == "__main__":
    main()