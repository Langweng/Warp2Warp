# engine/trainer.py
import torch
import torch.nn as nn
import logging
import cv2
import numpy as np
from models.ihn_utils import compute_weighted_loss, reshape_delta_42_to_222, reshape_delta_222_to_42
from utils.homography_utils import (
    get_homography_with_adaptation,
    corners_to_H,
    is_homography_well_conditioned,
    resolve_homography_adaptation,
    resolve_reflect_padding,
    safe_inverse_homography,
    warp_perspective_with_reflect_padding,
)

logger = logging.getLogger(__name__)

class Trainer:
    """
    封装单步优化过程，包括损失计算、反向传播和参数更新
    """
    def __init__(self, model, optimizer, device, config, net_copy=None, current_epoch=1):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config

        # IDR相关
        self.net_copy = net_copy
        self.current_epoch = current_epoch
        self.epoch_list = config['training']['epoch_list'] if 'epoch_list' in config['training'] else [0]
        self.warp2 = config['training']['warp2'] if 'warp2' in config['training'] else 0
        self.cumulative_epochs = np.cumsum([0] + self.epoch_list).tolist()
        self.current_stage = self._get_stage_index(current_epoch)
        self.net_copy_stage = self.current_stage if current_epoch > 1 else -1
        self.H, self.W = config['experiment']['image_size']
        self.marginal = config['data']['marginal']
        self.crop_H, self.crop_W = self.H - 2 * self.marginal, self.W - 2 * self.marginal
        self.reflect_pad = resolve_reflect_padding(config, self.crop_H, self.crop_W)
        self.homography_adaptation = resolve_homography_adaptation(config)
        
        # Basemodel iteration counts (IHN)
        model_name = config.get('model', {}).get('name', model.__class__.__name__)
        model_cfg = config.get('model', {}).get(model_name, {})
        self.iters_lev0 = int(model_cfg.get('iters_lev0', getattr(model, 'iters_lev0', 6)))
        self.iters_lev1 = int(model_cfg.get('iters_lev1', getattr(model, 'iters_lev1', 0)))
        loss_cfg = config.get('training', {}).get('loss', {})
        self.loss_type = str(loss_cfg.get('type', config.get('training', {}).get('loss_type', 'l1'))).lower()
        self.loss_alpha = float(loss_cfg.get('alpha', config.get('training', {}).get('loss_alpha', 0.85)))
        self.loss_epsilon = float(loss_cfg.get('epsilon', config.get('training', {}).get('loss_epsilon', 0.1)))
        self.loss_speed_threshold = float(
            loss_cfg.get('speed_threshold', config.get('training', {}).get('loss_speed_threshold', 1.0))
        )
        self.max_refine_displacement = float(
            config.get('training', {}).get(
                'max_refine_displacement',
                max(self.crop_H, self.crop_W) / 2,
            )
        )
    
    def set_epoch(self, epoch):
        """由 train_loop 调用，更新当前 epoch"""
        self.current_epoch = epoch
        self.current_stage = self._get_stage_index(epoch)

    def _get_stage_index(self, epoch: int) -> int:
        current_stage = len(self.epoch_list) - 1
        for i, end_epoch in enumerate(self.cumulative_epochs[1:]):
            if epoch <= end_epoch:
                current_stage = i
                break
        return current_stage
    
    def _sync_net_copy(self, target_stage: int):
        """仅在新的 refine stage 开始时同步主模型参数到 net_copy"""
        if self.net_copy is None:
            return
            
        if target_stage > 0 and self.net_copy_stage < target_stage:
            self.net_copy_stage = target_stage
            self.net_copy.load_state_dict(self.model.state_dict(), strict=False)
            self.net_copy.eval()
            for param in self.net_copy.parameters():
                param.requires_grad = False
            self.net_copy.to(self.device)
            logger.info("net_copy synchronized and frozen for stage %d.", target_stage)
    
    def _get_current_stage_params(self):
        """根据当前 epoch 确定是否使用细化网络"""
        perturb_range_Hk = self.warp2
        is_refine_stage = self.current_stage > 0
        
        return perturb_range_Hk, is_refine_stage, self.current_stage

    def _prediction_delta_to_global_homography(self, d_hat: np.ndarray, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Convert net_copy prediction to a stable homography, falling back to identity if it degenerates."""
        d_hat = np.asarray(d_hat, dtype=np.float32)
        d_hat = np.nan_to_num(d_hat, nan=0.0, posinf=0.0, neginf=0.0)
        d_hat = np.clip(d_hat, -self.max_refine_displacement, self.max_refine_displacement)
        d_hat = np.rint(d_hat).astype(np.float32)

        try:
            H_hat = corners_to_H(d_hat, self.marginal, self.H, self.W).astype(np.float32)
            if not is_homography_well_conditioned(H_hat):
                raise ValueError(f"ill-conditioned predicted H: {H_hat.tolist()}")
            return H_hat, d_hat
        except Exception as exc:
            logger.warning(
                "Using identity refinement homography at epoch %d, batch sample %d because net_copy prediction is invalid: %s",
                self.current_epoch,
                batch_index,
                exc,
            )
            return np.eye(3, dtype=np.float32), np.zeros((4, 2), dtype=np.float32)
    
    def preprocess(self, data):
        """
        实现 IDR 动态数据生成逻辑: 
        I_fixA, I_warpB, H_global_true -> [net_copy] -> I_warp2B, d_gt
        Args:
            data (dict): 来自 DataLoader 的基础数据
        Returns:
            dict (dict): 包含 warp2B_crop 和 d_gt 的字典
        """
        
        # 0. 获取 IDR 状态
        perturb_range_Hk, is_refine_stage, current_stage = self._get_current_stage_params()
        
        # 1. 数据准备 (将基础数据从 CPU 转移到 GPU)
        I_fixA_crop = data['I_fixA_crop'].to(self.device)       # B x 3 x H_c x W_c (GPU)
        I_fixB = data['I_fixB'].to(self.device)                 # B x 3 x H x W (GPU)
        I_warpB_crop = data['I_warpB_crop'].to(self.device)     # B x 3 x H_c x W_c (GPU)
        H_global_true = data['H_global_true'].to(self.device)   # B x 3 x 3 (GPU)
        
        if not is_refine_stage or self.net_copy is None:
            # ============ 初始阶段：对 I_warpB 施加随机扰动得到 I_warp2B =============
            B = I_fixB.shape[0]
            
            I_warp2B = torch.zeros_like(I_fixB)
            I_warp2B_crop = torch.zeros((B, 3, self.H - 2*self.marginal, self.W - 2*self.marginal), device=self.device)
            d_gt = torch.zeros((B, 4, 2), device=self.device)
            H_global_gt = torch.zeros((B, 3, 3), device=self.device)
            H_local_gt = torch.zeros((B, 3, 3), device=self.device)
            
            for b in range(B):
                H_local_gt_np, H_global_gt_np, d_gt_np = get_homography_with_adaptation(
                    self.H,
                    self.W,
                    self.marginal,
                    perturb_range_Hk,
                    adaptation=self.homography_adaptation,
                )
                d_gt[b] = torch.from_numpy(d_gt_np).to(self.device)
                H_global_gt[b] = torch.from_numpy(H_global_gt_np).to(self.device)
                H_local_gt[b] = torch.from_numpy(H_local_gt_np).to(self.device)
                
                # 对单张图像应用透视变换
                # I_warp2B[b] = Hk^-1 * H_true^-1 * I_fixB[b]
                I_fixB_np = I_fixB[b].permute(1, 2, 0).cpu().numpy()  # 转为numpy格式 (H,W,C)
                H_combined = (
                    safe_inverse_homography(H_global_gt[b].cpu().numpy(), context="H_global_gt")
                    @ safe_inverse_homography(H_global_true[b].cpu().numpy(), context="H_global_true")
                )
                I_warp2B_np = warp_perspective_with_reflect_padding(
                    I_fixB_np,
                    H_combined,
                    (self.H, self.W),
                    reflect_pad=self.reflect_pad,
                )

                I_warp2B[b] = torch.from_numpy(I_warp2B_np).permute(2, 0, 1).to(self.device)
                I_warp2B_crop[b] = I_warp2B[b, :, self.marginal:self.H-self.marginal, self.marginal:self.W-self.marginal]

                # Debug only
                # from utils.visualize import save_debug_image, visualize_homo
                # fpt_src = np.array([
                # [self.marginal, self.marginal], [self.W - 1 - self.marginal, self.marginal], 
                # [self.marginal, self.H - 1 - self.marginal], [self.W - 1 - self.marginal, self.H - 1 - self.marginal]
                # ], dtype=np.float32)
                # fpt_tar_1 = fpt_src + data['d_true'][b].cpu().numpy()
                # fpt_tar_2 = fpt_src + d_gt_np
                # visualize_homo(I_fixB_np, data['I_warpB'][b].permute(1, 2, 0).cpu().numpy(), data['I_warpB_crop'][b].permute(1, 2, 0).cpu().numpy(), [fpt_src, fpt_tar_1])
                # visualize_homo(data['I_warpB'][b].permute(1, 2, 0).cpu().numpy(), I_warp2B_np, I_warp2B_crop[b].permute(1, 2, 0).cpu().numpy(), [fpt_src, fpt_tar_2])

                I_refined = torch.zeros_like(I_fixB)    # I_refined 为空
            
        else:
            # ================= 细化阶段：I_warp -> I_refined -> I_warp2 =================
            
            # --- 1. 同步 net_copy ---
            self._sync_net_copy(current_stage)

            # --- 2. 预测细化单应性矩阵 (使用冻结 net_copy) ---
            with torch.no_grad():
                input_pair = torch.cat([I_warpB_crop, I_fixA_crop], dim=1) # B x 6 x h x w
                d_hat = self.net_copy(input_pair, test_mode=True)          # B x 2 x 2 x 2 (取 lev0 预测)
                d_hat = reshape_delta_222_to_42(d_hat).detach().cpu().numpy()

                # 计算用来细化的全局单应性矩阵 H_hat
                B = I_fixB.shape[0]
                H_hat_global = torch.zeros((B, 3, 3), device=self.device)
                for b in range(B):
                    H_hat_global_np, d_hat_np = self._prediction_delta_to_global_homography(d_hat[b], b)
                    d_hat[b] = d_hat_np
                    H_hat_global[b] = torch.from_numpy(H_hat_global_np)
                
            # --- 3. 数据细化 (I_refined = H_hat * I_warpB = H_hat * H_true^-1 * I_fixB) ---
            # 初始化结果张量
            B = I_fixB.shape[0]
            I_refined = torch.zeros_like(I_fixB)
            
            for b in range(B):
                I_fixB_np = I_fixB[b].permute(1, 2, 0).cpu().numpy()
                H_combined = (
                    H_hat_global[b].cpu().numpy()
                    @ safe_inverse_homography(H_global_true[b].cpu().numpy(), context="H_global_true")
                )
                I_refined_np = warp_perspective_with_reflect_padding(
                    I_fixB_np,
                    H_combined,
                    (self.H, self.W),
                    reflect_pad=self.reflect_pad,
                )
                I_refined[b] = torch.from_numpy(I_refined_np).permute(2, 0, 1).to(self.device)
            
            # --- 4. 对 I_refined (即 H_hat * I_warpB) 施加随机扰动 H_k =============
            # 初始化结果张量
            I_warp2B = torch.zeros_like(I_fixB)
            I_warp2B_crop = torch.zeros((B, 3, self.H - 2*self.marginal, self.W - 2*self.marginal), device=self.device)
            d_gt = torch.zeros((B, 4, 2), device=self.device)
            H_global_gt = torch.zeros((B, 3, 3), device=self.device)
            H_local_gt = torch.zeros((B, 3, 3), device=self.device)

            for b in range(B):
                # 生成随机扰动 Hk (对每个批次单独生成)
                H_local_gt_np, H_global_gt_np, d_gt_np = get_homography_with_adaptation(
                    self.H,
                    self.W,
                    self.marginal,
                    perturb_range_Hk,
                    adaptation=self.homography_adaptation,
                )
                d_gt[b] = torch.from_numpy(d_gt_np).to(self.device)
                H_global_gt[b] = torch.from_numpy(H_global_gt_np).to(self.device)
                H_local_gt[b] = torch.from_numpy(H_local_gt_np).to(self.device)
                
                # H_warp2_global = Hk^-1 * H_refined_global 
                #                = Hk^-1 * H_hat * H_true^-1
                I_fixB_np = I_fixB[b].permute(1, 2, 0).cpu().numpy()
                H_combined = (
                    safe_inverse_homography(H_global_gt[b].cpu().numpy(), context="H_global_gt")
                    @ H_hat_global[b].cpu().numpy()
                    @ safe_inverse_homography(H_global_true[b].cpu().numpy(), context="H_global_true")
                )
                I_warp2B_np = warp_perspective_with_reflect_padding(
                    I_fixB_np,
                    H_combined,
                    (self.H, self.W),
                    reflect_pad=self.reflect_pad,
                )
                I_warp2B[b] = torch.from_numpy(I_warp2B_np).permute(2, 0, 1).to(self.device)
                I_warp2B_crop[b] = I_warp2B[b, :, self.marginal:self.H-self.marginal, self.marginal:self.W-self.marginal]

                # Debug only
                # from utils.visualize import save_debug_image, visualize_homo
                # fpt_src = np.array([
                # [self.marginal, self.marginal], [self.W - 1 - self.marginal, self.marginal], 
                # [self.marginal, self.H - 1 - self.marginal], [self.W - 1 - self.marginal, self.H - 1 - self.marginal]
                # ], dtype=np.float32)
                # fpt_tar_1 = fpt_src + data['d_true'][b].cpu().numpy()
                # fpt_tar_2 = fpt_src + d_gt_np
                # visualize_homo(I_fixB_np, data['I_warpB'][b].permute(1, 2, 0).cpu().numpy(), data['I_warpB_crop'][b].permute(1, 2, 0).cpu().numpy(), [fpt_src, fpt_tar_1])
                # visualize_homo(I_refined[b].permute(1, 2, 0).cpu().numpy(), I_warp2B_np, I_warp2B_crop[b].permute(1, 2, 0).cpu().numpy(), [fpt_src, fpt_tar_2])

        # 5. 组合最终数据
        data['I_refined'] = I_refined
        data['I_warp2B'] = I_warp2B
        data['I_warp2B_crop'] = I_warp2B_crop
        data['d_gt'] = d_gt
        data['H_local_gt'] = H_local_gt
        data['H_global_gt'] = H_global_gt

        return data
        
    def train_step(self, data):
        """
        执行单个 batch 的训练步骤
        支持 IHN basemodel
        """
        model_name = self.config.get('model', {}).get('name', self.model.__class__.__name__)
        if model_name == 'IHN':

            self.model.train()
            
            # 1. 数据准备, 支持 warp2-fix 训练
            if self.config['training']['strategy'] == 'self-supervised':
                data = self.preprocess(data)

            I_fix = data['I_fixA_crop'].to(self.device)
            if 'I_warp2B_crop' in data and data['I_warp2B_crop'] is not None:
                I_warp = data['I_warp2B_crop'].to(self.device)
            else:
                I_warp = data['I_warpB_crop'].to(self.device)
            d_gt = data['d_gt'].to(self.device)     # 伪标签 Bx4x2
            d_true = data['d_true'].to(self.device) # 真实标签 Bx4x2

            # Debug only
            # from utils.visualize import save_debug_image, visualize_homo
            # import numpy as np
            # save_debug_image(I_fix[0].permute(1, 2, 0).cpu().numpy(), 'I_fixA.png')
            # save_debug_image(data['I_warpB_crop'][0].permute(1, 2, 0).cpu().numpy(), 'I_warpB.png')
            # save_debug_image(I_warp[0].permute(1, 2, 0).cpu().numpy(), 'I_warp.png')
            # marginal = self.config['data']['marginal']
            # H, W = data['I_fixB'][0].shape[1:]
            # fpt_src = np.array([
            # [marginal, marginal], [W - 1 - marginal, marginal], 
            # [marginal, H - 1 - marginal], [W - 1 - marginal, H - 1 - marginal]
            # ], dtype=np.float32)
            # visualize_homo(data['I_fixB'][0].permute(1, 2, 0).cpu().numpy(), 
            #                data['I_warpB'][0].permute(1, 2, 0).cpu().numpy(), 
            #                data['I_warpB_crop'][0].permute(1, 2, 0).cpu().numpy(),
            #                [fpt_src, fpt_src + d_true[0].cpu().numpy()]
            #                )
            # visualize_homo(data['I_warpB'][0].permute(1, 2, 0).cpu().numpy(), 
            #                data['I_warp2B'][0].permute(1, 2, 0).cpu().numpy(), 
            #                data['I_warp2B_crop'][0].permute(1, 2, 0).cpu().numpy(),
            #                [fpt_src, fpt_src + d_gt[0].cpu().numpy()]
            #                )


            # 2. 构造模型输入 (I_warp 和 I_fix 拼接)
            input_pair = torch.cat([I_warp, I_fix], dim=1)  # Bx6xHxW
            
            # 3. 前向传播
            self.optimizer.zero_grad()

            preds_lev0, preds_lev1 = self.model(input_pair)

            # 4. 损失计算
            # d_gt 需要从 Bx4x2 转换为 Bx2x2x2
            d_gt = reshape_delta_42_to_222(d_gt)
            loss_lev0 = compute_weighted_loss(
                preds_lev0,
                d_gt,
                alpha=self.loss_alpha,
                loss_type=self.loss_type,
                epsilon=self.loss_epsilon,
                speed_threshold=self.loss_speed_threshold,
            )
            loss_lev1 = compute_weighted_loss(
                preds_lev1,
                d_gt,
                alpha=self.loss_alpha,
                loss_type=self.loss_type,
                epsilon=self.loss_epsilon,
                speed_threshold=self.loss_speed_threshold,
            ) if self.iters_lev1 > 0 else 0.0
            loss = loss_lev0 + loss_lev1

            # 5. 反向传播与优化
            loss.backward()
            
            # 可选：梯度裁剪
            if self.config['training'].get('grad_clip', None):
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
                
            self.optimizer.step()

            return loss

        raise NotImplementedError(f"Unsupported basemodel {model_name!r} in engine.Trainer")
        
