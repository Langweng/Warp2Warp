#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用于生成 GoogleMap 数据集的 JSON 清单文件。

默认读取 data/raw/GoogleMap，并输出 data/manifests/googlemap.json。
"""
import os
import json
import argparse
from tqdm import tqdm


def process_dataset_subset(data_dir, subset_name, max_samples=None):
    """
    处理数据集的一个子集（train或val）
    
    Args:
        data_dir (str): 数据集根目录
        subset_name (str): 子集名称，'train'或'val'
        max_samples (int, optional): 最大样本数量
        
    Returns:
        list: 包含该子集所有图像对的列表
    """
    # 定义fixA和fixB的目录路径
    fixA_dir = os.path.join(data_dir, f'{subset_name}_fixA')
    fixB_dir = os.path.join(data_dir, f'{subset_name}_fixB')
    
    # 检查目录是否存在
    if not os.path.exists(fixA_dir):
        raise FileNotFoundError(f"{subset_name}_fixA目录不存在: {fixA_dir}")
    if not os.path.exists(fixB_dir):
        raise FileNotFoundError(f"{subset_name}_fixB目录不存在: {fixB_dir}")
    
    # 获取fixA目录中的所有图像文件
    fixA_files = [f for f in os.listdir(fixA_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # 对文件进行排序，确保生成的pair_id是连续的
    fixA_files.sort()
    
    # 限制样本数量（如果指定了max_samples）
    if max_samples is not None:
        fixA_files = fixA_files[:max_samples]
    
    # 初始化子集数据
    subset_data = []
    
    # 遍历所有fixA文件，检查对应的fixB文件是否存在
    for i, filename in enumerate(fixA_files):
        # 每处理100个文件输出一次进度
        if (i + 1) % 100 == 0:
            print(f"处理{subset_name}数据集: 已处理{min(i + 1, max_samples or len(fixA_files))}个文件")
        
        # 检查fixB目录中是否存在同名文件
        if os.path.exists(os.path.join(fixB_dir, filename)):
            # 生成pair_id，格式为4位数字，不足前面补0
            pair_id = f"{i+1:04d}"
            
            # 添加到子集数据中
            subset_data.append({
                'fixA_path': os.path.join(f'{subset_name}_fixA', filename),
                'fixB_path': os.path.join(f'{subset_name}_fixB', filename),
                'pair_id': pair_id
            })
        else:
            print(f"警告: {subset_name}_fixB目录中找不到对应的文件: {filename}")
    
    return subset_data


def generate_manifest(data_dir, output_file, max_samples_train=None, max_samples_val=None):
    """
    生成GoogleMap数据集的JSON清单文件，包含train和val两个键
    
    Args:
        data_dir (str): GoogleMap数据集根目录
        output_file (str): 输出的JSON清单文件路径
        max_samples_train (int, optional): train数据集的最大样本数量
        max_samples_val (int, optional): val数据集的最大样本数量
    """
    print("开始生成JSON清单...")
    
    # 处理train数据集
    print("处理train数据集...")
    train_data = process_dataset_subset(data_dir, 'train', max_samples_train)
    
    # 处理val数据集
    print("处理val数据集...")
    val_data = process_dataset_subset(data_dir, 'val', max_samples_val)
    
    # 构建包含train和val键的完整数据结构
    manifest_data = {
        'train': train_data,
        'val': val_data
    }
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 写入JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)
    
    print(f"JSON清单文件已生成: {output_file}")
    print(f"train数据集包含{len(train_data)}对图像数据")
    print(f"val数据集包含{len(val_data)}对图像数据")


def main():
    """
    主函数，解析命令行参数并调用generate_manifest函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成GoogleMap数据集的JSON清单文件')
    parser.add_argument('--data_dir', type=str, 
                        default='data/raw/GoogleMap',
                        help='GoogleMap数据集根目录')
    parser.add_argument('--output_file', type=str, 
                        default='data/manifests/googlemap.json',
                        help='输出的JSON清单文件路径')
    parser.add_argument('--max_samples_train', type=int, default=None,
                        help='train数据集的最大样本数量，None表示使用所有样本')
    parser.add_argument('--max_samples_val', type=int, default=None,
                        help='val数据集的最大样本数量，None表示使用所有样本')
    
    args = parser.parse_args()
    
    # 调用生成函数
    generate_manifest(args.data_dir, args.output_file, args.max_samples_train, args.max_samples_val)


if __name__ == '__main__':
    main()
