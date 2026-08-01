# datasets/warp2_fix_online_dataset.py
from __future__ import annotations

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import logging
import matplotlib.pyplot as plt
import cv2
import kornia.geometry.transform as tgm # 用于单应性矩阵计算

from utils.homography_utils import (
    delta_to_homography_matrices,
    get_homography_with_adaptation,
    corners_to_H,
    resolve_homography_adaptation,
    resolve_reflect_padding,
    safe_inverse_homography,
    warp_perspective_with_reflect_padding,
)
from utils.fixed_test_set import load_fixed_homography_labels
from models.ihn_utils import reshape_delta_222_to_42
from models.ihn_net import IHN
from utils.io import load_config # 用于加载配置中的超参数

logger = logging.getLogger(__name__)


_INTERPOLATION_MODES = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4,
}


class Warp2FixOnlineDataset(Dataset):
    """
    IDR 训练阶段使用的在线数据集，负责：
    1. 在线生成 H_true 扭曲（FixB -> Warp）。
    2. 根据当前 Epoch (t)，决定是做初始扰动 (H1) 还是 IDR 细化 (H_hat + Ht)。
    3. 生成训练样本 (I_warp2, I_fixA) 和伪标签 d_gt (d_t)。
    """
    def __init__(
        self,
        manifest_path: str,
        image_root: str,
        config: dict,
        dataset_type='train',
        fixed_labels_path: str | None = None,
    ):

        with open(manifest_path, 'r') as f:
            manifest_data = json.load(f)
            
        if isinstance(manifest_data, list):
            self.manifest = manifest_data
        elif isinstance(manifest_data, dict) and dataset_type in manifest_data:
            self.manifest = manifest_data.get(dataset_type, [])
        else:
            raise ValueError(f"Invalid manifest format for split {dataset_type!r}: {manifest_path}")
        
        self.image_root = image_root
        self.config = config
        
        # 几何参数
        self.marginal = config['data']['marginal']
        self.H, self.W = config['experiment']['image_size'] # 从experiment节点获取图像尺寸
        self.crop_H, self.crop_W = self.H - 2 * self.marginal, self.W - 2 * self.marginal # 例如 128, 128
        self.perturb_range_H1 = config['data']['perturb_range']    # warp 扰动范围
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
        
        print(f"Loaded {len(self.manifest)} {dataset_type} samples from {manifest_path}")

        # IDR 状态变量 (由 train_loop.py 设定)
        # self.refiner_net: IHN = None
        self.current_epoch: int = 0
        self.epoch_list = config['training']['epoch_list'] # 每个迭代阶段训练多少个 epoch
        self.cumulative_epochs = np.cumsum([0] + self.epoch_list).tolist()
        self.perturb_range_H2 = config['training']['warp2'] # 固定的 warp2 角点扰动范围 U[-r, r]
        self.fixed_labels = None
        if fixed_labels_path is not None:
            self.fixed_labels = load_fixed_homography_labels(
                fixed_labels_path,
                self.manifest,
                dataset_type,
                config,
            )


    def __len__(self):
        return len(self.manifest)
    
    def get_current_stage_params(self):
        """根据当前 epoch 确定是否使用细化网络"""
        current_stage = len(self.epoch_list) - 1    # 默认值：如果 epoch 超过所有阶段，使用最后一个阶段
        # self.cumulative_epochs[1:] 是每个阶段的结束 epoch 边界 (例如：1, 2, 3, 4, 5)
        for i, end_epoch in enumerate(self.cumulative_epochs[1:]):
            if self.current_epoch <= end_epoch:
                current_stage = i
                break
        
        # 使用固定的 warp2 值
        perturb_range_Hk = self.perturb_range_H2
        is_refine_net = current_stage > 0
        
        return perturb_range_Hk, is_refine_net

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
        """加载图像，归一化[0,1]，转为 float32 (H, W, C) 并裁剪 ROI"""
        full_path = os.path.join(self.image_root, path)
        img_np = plt.imread(full_path)
        if img_np is None:
            raise FileNotFoundError(f"Image not found at {full_path}")
        # 归一化到 [0, 1]
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
        
        img_crop = img_np[self.marginal:self.H-self.marginal, self.marginal:self.W-self.marginal]
        
        return img_np, img_crop # ndarray (H, W, C), (crop_H, crop_W, C)


    def __getitem__(self, idx):
        item = self.manifest[idx]
        fixed_record = self.fixed_labels[idx] if self.fixed_labels is not None else None
        
        # 1. 加载图像并裁剪 (FixA 和 FixB 都是未扭曲的)
        I_fixA, I_fixA_crop = self.load_and_preprocess(item['fixA_path'], role='fixA') # Modality A: Fixed image y (H_crop, W_crop, C)
        I_fixB, I_fixB_crop = self.load_and_preprocess(item['fixB_path'], role='fixB') # Modality B: Source image x (H_crop, W_crop, C)

        # 2. 在线生成 H_true：FixB -> WarpB
        if fixed_record is None:
            H_local_true, H_global_true, d_true = get_homography_with_adaptation(
                self.H,
                self.W,
                self.marginal,
                self.perturb_range_H1,
                adaptation=self.homography_adaptation,
            )
            d_eval = np.zeros((4, 2), dtype=np.float32)
            sample_key = str(item.get('sample_id', item.get('id', idx)))
        else:
            d_true = fixed_record['d_true']
            d_eval = fixed_record['d_eval']
            H_local_true, H_global_true = delta_to_homography_matrices(d_true, self.H, self.W, self.marginal)
            sample_key = fixed_record['sample_key']
        
        I_warpB = warp_perspective_with_reflect_padding(
            I_fixB,
            safe_inverse_homography(H_global_true, context="H_global_true"),
            (self.H, self.W),
            reflect_pad=self.reflect_pad,
        )
        I_warpB_crop = I_warpB[self.marginal:self.H-self.marginal, self.marginal:self.W-self.marginal]

        # Debug only
        # from utils.visualize import save_debug_image
        # save_debug_image(I_fixB, 'fixB')
        # save_debug_image(I_warpB, 'warpB')
        # save_debug_image(I_warpB_crop, 'warpB_crop')

        return {
            'I_fixA': torch.from_numpy(I_fixA).permute(2, 0, 1).float(),                      # 模态A原始图像
            'I_fixA_crop': torch.from_numpy(I_fixA_crop).permute(2, 0, 1).float(),            # 裁剪后的模态A原始图像
            'I_fixB': torch.from_numpy(I_fixB).permute(2, 0, 1).float(),                      # 模态B原始图像
            'I_fixB_crop': torch.from_numpy(I_fixB_crop).permute(2, 0, 1).float(),            # 裁剪后的模态B原始图像
            'I_warpB': torch.from_numpy(I_warpB).permute(2, 0, 1).float(),                    # 模态B扭曲图像
            'I_warpB_crop': torch.from_numpy(I_warpB_crop).permute(2, 0, 1).float(),          # 裁剪后的模态B扭曲图像
            'd_true': torch.from_numpy(d_true).float(),               # 4 x 2 (真实局部单应性)
            'H_local_true': torch.from_numpy(H_local_true).float(),   # 3 x 3 (真实局部单应性)
            'H_global_true': torch.from_numpy(H_global_true).float(), # 3 x 3 (真实全局单应性)

            # 返回空张量占位，将在 Trainer 生成
            'I_refined': torch.empty(0),                             # 模态B细化图像                                        
            'I_warp2B': torch.empty(0),                              # 模态B二次扭曲图像
            'I_warp2B_crop': torch.empty(0),                         # 裁剪后的模态B二次扭曲图像
            'd_gt': torch.empty(0),                                  # 4 x 2 (伪标签)
            'H_local_gt': torch.empty(0),                            # 3 x 3 (伪标签局部单应性)
            'H_global_gt': torch.empty(0),                           # 3 x 3 (伪标签全局单应性)
            'd_eval': torch.from_numpy(d_eval).float(),
            'sample_index': idx,
            'sample_key': sample_key,
            'fixA_path': item['fixA_path'],
            'fixB_path': item['fixB_path'],
        }
