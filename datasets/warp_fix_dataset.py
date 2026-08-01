from __future__ import annotations

import torch
import numpy as np
import cv2
import json
import os
import matplotlib.pyplot as plt
import kornia.geometry.transform as tgm  # 用于单应性矩阵计算

from torch.utils.data import Dataset
from utils.homography_utils import (
    delta_to_homography_matrices,
    get_homography_with_adaptation,
    resolve_homography_adaptation,
    resolve_reflect_padding,
    safe_inverse_homography,
    warp_perspective_with_reflect_padding,
)
from utils.fixed_test_set import load_fixed_homography_labels


_INTERPOLATION_MODES = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4,
}


class WarpFixDataset(Dataset):
    """
    标准有监督单应性估计的数据集加载器
    
    负责：
    1. 加载FixA和FixB模态的原始图像
    2. 对FixB模态图像应用一次随机单应性变换得到WarpB
    3. 返回裁剪后的模态A原图、裁剪后的模态B扭曲图、真实四角点位移、全局与局部单应性矩阵
    """
    def __init__(
        self,
        manifest_path: str,
        image_root: str,
        config: dict,
        dataset_type='train',
        fixed_labels_path: str | None = None,
    ):
        """
        初始化WarpFixDataset
        
        Args:
            manifest_path (str): 包含图像路径的JSON文件路径
            image_root (str): 图像文件的根目录
            config (dict): 包含数据处理相关配置的字典
            dataset_type (str): 数据集类型，'train', 'val'
        """
        # 加载JSON清单文件
        with open(manifest_path, 'r') as f:
            manifest_data = json.load(f)
            
        # 检查JSON格式
        if isinstance(manifest_data, list):
            # 简单格式：直接是样本数组
            self.manifest = manifest_data
        elif isinstance(manifest_data, dict) and dataset_type in manifest_data:
            # 复杂格式：按 split 读取对应样本，支持 train / val / test
            self.manifest = manifest_data.get(dataset_type, [])
        else:
            raise ValueError(f"Invalid manifest format for split {dataset_type!r}: {manifest_path}")
        
        self.image_root = image_root
        self.config = config
        
        # 几何参数
        self.marginal = config['data']['marginal']
        self.H, self.W = config['experiment']['image_size'] # 从experiment节点获取图像尺寸
        self.crop_H, self.crop_W = self.H - 2 * self.marginal, self.W - 2 * self.marginal
        self.perturb_range = config['data']['perturb_range'] # 扰动范围
        self.reflect_pad = resolve_reflect_padding(config, self.crop_H, self.crop_W)
        self.homography_adaptation = resolve_homography_adaptation(config)

        resize_cfg = config.get('data', {}).get('input_resize', {})
        self.resize_enabled = bool(resize_cfg.get('enabled', False))
        resize_size = resize_cfg.get('size')
        self.resize_size = (int(resize_size[0]), int(resize_size[1])) if resize_size is not None else None
        resize_apply_to = resize_cfg.get('apply_to', ['fixA', 'fixB'])
        if isinstance(resize_apply_to, str):
            resize_apply_to = [resize_apply_to]
        self.resize_apply_to = set(resize_apply_to)
        interpolation_name = str(resize_cfg.get('interpolation', 'bicubic')).lower()
        if interpolation_name not in _INTERPOLATION_MODES:
            raise ValueError(
                f"Unsupported data.input_resize.interpolation={interpolation_name!r}. "
                f"Expected one of {sorted(_INTERPOLATION_MODES)}."
            )
        self.resize_interpolation = _INTERPOLATION_MODES[interpolation_name]
        self.fixed_labels = None
        if fixed_labels_path is not None:
            self.fixed_labels = load_fixed_homography_labels(
                fixed_labels_path,
                self.manifest,
                dataset_type,
                config,
            )
        
        print(f"Loaded {len(self.manifest)} {dataset_type} samples from {manifest_path}")
    
    def __len__(self):
        return len(self.manifest)
        
    
    def _maybe_resize(self, img_np: np.ndarray, role: str) -> np.ndarray:
        if not self.resize_enabled or role not in self.resize_apply_to:
            return img_np

        target_h, target_w = self.resize_size or (self.H, self.W)
        if img_np.shape[0] == target_h and img_np.shape[1] == target_w:
            return img_np

        return cv2.resize(img_np, (target_w, target_h), interpolation=self.resize_interpolation)

    def _validate_image_size(self, img_np: np.ndarray, full_path: str, role: str):
        if img_np.shape[0] != self.H or img_np.shape[1] != self.W:
            raise ValueError(
                f"Expected {role} image {full_path} to have shape ({self.H}, {self.W}) after preprocessing, "
                f"but got ({img_np.shape[0]}, {img_np.shape[1]}). "
                "Adjust experiment.image_size or enable data.input_resize for this modality."
            )

    def load_and_preprocess(self, path: str, role: str):
        """加载图像，归一化[0,1]，转为float32 (H, W, C) 并裁剪ROI"""
        full_path = os.path.join(self.image_root, path)
        img_np = plt.imread(full_path)
        if img_np is None:
            raise FileNotFoundError(f"Image not found at {full_path}")
        
        # 对于.jpg格式，归一化到 [0, 1]
        if img_np.dtype == np.uint8:
            img_np = img_np.astype(np.float32) / 255.0
            
        # 确保图像是RGB格式 (有些图像可能是RGBA或灰度图)
        if len(img_np.shape) == 2:
            # 灰度图转RGB
            img_np = np.stack([img_np, img_np, img_np], axis=-1)
        elif img_np.shape[2] == 4:
            # RGBA转RGB
            img_np = img_np[:, :, :3]

        img_np = self._maybe_resize(img_np, role)
        self._validate_image_size(img_np, full_path, role)
        
        # 裁剪ROI区域
        img_crop = img_np[self.marginal:self.H-self.marginal, self.marginal:self.W-self.marginal]
        
        return img_np, img_crop # numpy array (H, W, C), (crop_H, crop_W, C)
    
    
    def __getitem__(self, idx):
        """
        获取指定索引的样本数据
        
        Args:
            idx (int): 样本索引
            
        Returns:
            dict: 包含以下键的字典
                - I_fixA_crop: 裁剪后的模态A原始图像
                - I_warpB_crop: 裁剪后的模态B扭曲图像
                - d: 四角点位移向量
                - H_local: 局部单应性矩阵
                - H_global: 全局单应性矩阵
        """
        item = self.manifest[idx]
        fixed_record = self.fixed_labels[idx] if self.fixed_labels is not None else None
        
        # 1. 加载图像并裁剪 (FixA和FixB都是未扭曲的)
        I_fixA, I_fixA_crop = self.load_and_preprocess(item['fixA_path'], role='fixA') # Modality A: Fixed image
        I_fixB, I_fixB_crop = self.load_and_preprocess(item['fixB_path'], role='fixB') # Modality B: Source image

        # 2. 在线生成H_true：FixB -> WarpB
        # H_local: 局部单应性矩阵 (crop_H x crop_W 坐标系)
        # H_global: 全局单应性矩阵 (H x W 坐标系)
        # d: 四角点位移
        if fixed_record is None:
            H_local_true, H_global_true, d_true = get_homography_with_adaptation(
                self.H,
                self.W,
                self.marginal,
                self.perturb_range,
                adaptation=self.homography_adaptation,
            )
            d_eval = np.zeros((4, 2), dtype=np.float32)
            sample_key = str(item.get('sample_id', item.get('id', idx)))
        else:
            d_true = fixed_record['d_true']
            d_eval = fixed_record['d_eval']
            H_local_true, H_global_true = delta_to_homography_matrices(d_true, self.H, self.W, self.marginal)
            sample_key = fixed_record['sample_key']
        
        # 应用单应性变换生成扭曲图像
        I_warpB = warp_perspective_with_reflect_padding(
            I_fixB,
            safe_inverse_homography(H_global_true, context="H_global_true"),
            (self.H, self.W),
            reflect_pad=self.reflect_pad,
        )
        I_warpB_crop = I_warpB[self.marginal:self.H-self.marginal, self.marginal:self.W-self.marginal]

        # # Debug only
        # from utils.visualize import visualize_homo
        # fpt_src = np.array([
        #     [self.marginal, self.marginal], [self.W - 1 - self.marginal, self.marginal], 
        #     [self.marginal, self.H - 1 - self.marginal], [self.W - 1 - self.marginal, self.H - 1 - self.marginal]
        # ], dtype=np.float32)
        # fpt_tar = fpt_src + d_true
        # visualize_homo(I_fixB, I_warpB, I_warpB_crop, [fpt_src, fpt_tar])
        
        # 返回处理后的样本数据
        return {
            'I_fixA': torch.from_numpy(I_fixA).permute(2, 0, 1).float(),              # 模态A原始图像
            'I_fixA_crop': torch.from_numpy(I_fixA_crop).permute(2, 0, 1).float(),    # 裁剪后的模态A原始图像
            'I_fixB': torch.from_numpy(I_fixB).permute(2, 0, 1).float(),              # 模态B原始图像
            'I_fixB_crop': torch.from_numpy(I_fixB_crop).permute(2, 0, 1).float(),    # 裁剪后的模态B原始图像
            'I_warpB': torch.from_numpy(I_warpB).permute(2, 0, 1).float(),            # 模态B扭曲图像
            'I_warpB_crop': torch.from_numpy(I_warpB_crop).permute(2, 0, 1).float(),  # 裁剪后的模态B扭曲图像
            'd_gt': torch.from_numpy(d_true).float(),            # 四角点位移向量
            'd_true': torch.from_numpy(d_true).float(),          # 四角点位移向量, d_gt = d_true
            'H_local_true': torch.from_numpy(H_local_true).float(),   # 局部单应性矩阵
            'H_global_true': torch.from_numpy(H_global_true).float(),  # 全局单应性矩阵
            'd_eval': torch.from_numpy(d_eval).float(),
            'sample_index': idx,
            'sample_key': sample_key,
            'fixA_path': item['fixA_path'],
            'fixB_path': item['fixB_path'],
        }
    
