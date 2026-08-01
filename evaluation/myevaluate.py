from __future__ import annotations

import csv
import os
import torch
import time
import numpy as np
import cv2
import logging
from models.ihn_utils import reshape_delta_42_to_222, reshape_delta_222_to_42
from utils.visualize import visualize_compare, visualize_warp_refine
from utils.homography_utils import (
    delta_to_homography_matrices,
    get_homography_with_adaptation,
    corners_to_H,
    is_homography_well_conditioned,
    resolve_homography_adaptation,
    resolve_reflect_padding,
    safe_inverse_homography,
    warp_perspective_with_reflect_padding,
)

logger = logging.getLogger(__name__)

class Evaluator:
    """
    模型评估器类，负责评估模型性能
    """
    def __init__(self, model, device, config, net_copy=None, current_epoch=0, report_dir=None, report_prefix=None):
        self.model = model
        self.device = device
        self.config = config
        self.net_copy = net_copy
        self.report_dir = report_dir
        self.report_prefix = report_prefix or "eval"
        self.H, self.W = config['experiment']['image_size']
        self.marginal = config['data']['marginal']
        self.crop_H, self.crop_W = self.H - 2 * self.marginal, self.W - 2 * self.marginal
        self.reflect_pad = resolve_reflect_padding(config, self.crop_H, self.crop_W)
        self.homography_adaptation = resolve_homography_adaptation(config)
        self.metric_scale_x, self.metric_scale_y = self._resolve_metric_scale()
        self.last_detailed_report = None
        
        # IDR相关
        self.current_epoch = current_epoch
        self.epoch_list = config['training']['epoch_list'] if 'epoch_list' in config['training'] else [0]
        self.warp2 = config['training']['warp2'] if 'warp2' in config['training'] else 0
        self.cumulative_epochs = np.cumsum([0] + self.epoch_list).tolist()
        self.max_refine_displacement = float(
            config.get('training', {}).get(
                'max_refine_displacement',
                max(self.crop_H, self.crop_W) / 2,
            )
        )

        if self.net_copy is not None:
            self.net_copy.eval()

    def _prediction_delta_to_global_homography(self, d_hat: np.ndarray, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
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
                "Using identity refinement homography during evaluation, sample %d, because net_copy prediction is invalid: %s",
                batch_index,
                exc,
            )
            return np.eye(3, dtype=np.float32), np.zeros((4, 2), dtype=np.float32)

    def _resolve_metric_scale(self):
        """
        返回用于汇报误差的坐标缩放。

        推理始终在当前 crop 分辨率上进行；如果 validation.report_resolution 被设置，
        则只在计算 MACE 时将角点位移换算到目标分辨率。
        """
        validation_cfg = self.config.get('validation', {})
        report_resolution = validation_cfg.get('report_resolution')
        if report_resolution is None:
            return 1.0, 1.0

        if len(report_resolution) != 2:
            raise ValueError(
                f"validation.report_resolution must be [H, W], got {report_resolution!r}."
            )

        report_h = int(report_resolution[0])
        report_w = int(report_resolution[1])
        if report_h <= 0 or report_w <= 0:
            raise ValueError(
                f"validation.report_resolution must be positive, got {report_resolution!r}."
            )

        report_scaling = validation_cfg.get('report_scaling')
        if report_scaling is None and self.homography_adaptation is not None:
            report_scaling = self.homography_adaptation['scale_mode']
        if report_scaling is None:
            report_scaling = 'size'

        if report_scaling == 'align_corners':
            if report_h <= 1 or report_w <= 1 or self.crop_H <= 1 or self.crop_W <= 1:
                raise ValueError(
                    "validation.report_scaling='align_corners' requires current and report resolutions > 1."
                )
            scale_y = float(report_h - 1) / float(self.crop_H - 1)
            scale_x = float(report_w - 1) / float(self.crop_W - 1)
        elif report_scaling == 'size':
            scale_y = report_h / float(self.crop_H)
            scale_x = report_w / float(self.crop_W)
        else:
            raise ValueError(
                f"Unsupported validation.report_scaling={report_scaling!r}. "
                "Expected 'align_corners' or 'size'."
            )
        logger.info(
            "Reporting MACE at resolution (%d, %d); crop resolution is (%d, %d), "
            "report scaling is %s and metric scales are (x=%.4f, y=%.4f).",
            report_h,
            report_w,
            self.crop_H,
            self.crop_W,
            report_scaling,
            scale_x,
            scale_y,
        )
        return scale_x, scale_y

    def _scale_delta_for_metric(self, delta_222: torch.Tensor) -> torch.Tensor:
        delta_metric = delta_222.detach().cpu()
        if self.metric_scale_x == 1.0 and self.metric_scale_y == 1.0:
            return delta_metric

        delta_metric = delta_metric.clone()
        delta_metric[:, 0, :, :] *= self.metric_scale_x
        delta_metric[:, 1, :, :] *= self.metric_scale_y
        return delta_metric

    def _compute_mace(self, d_true_222: torch.Tensor, d_pred_222: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        d_true_metric = self._scale_delta_for_metric(d_true_222)
        d_pred_metric = self._scale_delta_for_metric(d_pred_222)
        mace = (d_true_metric - d_pred_metric) ** 2
        mace = (mace[:, 0, :, :] + mace[:, 1, :, :]) ** 0.5
        mace_vec = torch.mean(torch.mean(mace, dim=1), dim=1)
        return mace, mace_vec

    def _compute_average_distortion_pixels(self, d_true_222: torch.Tensor) -> torch.Tensor:
        d_true_metric = self._scale_delta_for_metric(d_true_222)
        distortion = (d_true_metric[:, 0, :, :] ** 2 + d_true_metric[:, 1, :, :] ** 2) ** 0.5
        return torch.mean(torch.mean(distortion, dim=1), dim=1)

    def _batch_to_list(self, value, batch_size: int):
        if value is None:
            return [None] * batch_size
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value] * batch_size

    def _build_sample_rows(
        self,
        data_blob: dict,
        d_true_222: torch.Tensor,
        primary_metric_key: str,
        primary_metric_values: torch.Tensor,
        extra_metrics: dict[str, torch.Tensor] | None = None,
    ) -> list[dict]:
        batch_size = int(primary_metric_values.shape[0])
        sample_indexes = self._batch_to_list(data_blob.get('sample_index'), batch_size)
        sample_keys = self._batch_to_list(data_blob.get('sample_key'), batch_size)
        fixA_paths = self._batch_to_list(data_blob.get('fixA_path'), batch_size)
        fixB_paths = self._batch_to_list(data_blob.get('fixB_path'), batch_size)
        distortions = self._compute_average_distortion_pixels(d_true_222).detach().cpu().tolist()
        primary_values = primary_metric_values.detach().cpu().tolist()
        extra_metric_lists = {}
        for metric_name, metric_values in (extra_metrics or {}).items():
            extra_metric_lists[metric_name] = metric_values.detach().cpu().tolist()

        rows = []
        for index in range(batch_size):
            distortion_pixels = float(distortions[index])
            row = {
                'sample_index': int(sample_indexes[index]) if sample_indexes[index] is not None else index,
                'sample_key': '' if sample_keys[index] is None else str(sample_keys[index]),
                'fixA_path': '' if fixA_paths[index] is None else str(fixA_paths[index]),
                'fixB_path': '' if fixB_paths[index] is None else str(fixB_paths[index]),
                'average_distortion_pixels': distortion_pixels,
                'distortion_percentile': 0.0,
                'difficulty': '',
                primary_metric_key: float(primary_values[index]),
            }
            if primary_metric_key != 'mace':
                row['mace'] = float(primary_values[index])
            for metric_name, metric_values in extra_metric_lists.items():
                row[metric_name] = float(metric_values[index])
            rows.append(row)
        return rows

    def _assign_difficulty_groups(self, rows: list[dict]) -> list[dict]:
        if not rows:
            return rows

        sorted_indices = sorted(
            range(len(rows)),
            key=lambda idx: (rows[idx]['average_distortion_pixels'], rows[idx]['sample_index']),
        )
        total = len(sorted_indices)
        for rank, row_index in enumerate(sorted_indices):
            percentile = 100.0 * rank / total
            if percentile < 30.0:
                difficulty = 'easy'
            elif percentile < 60.0:
                difficulty = 'moderate'
            else:
                difficulty = 'hard'
            rows[row_index]['distortion_percentile'] = percentile
            rows[row_index]['difficulty'] = difficulty
        return rows

    def _summarize_group_metrics(self, rows: list[dict], metric_key: str) -> dict[str, dict]:
        group_ranges = {
            'easy': '0-30%',
            'moderate': '30-60%',
            'hard': '60-100%',
        }
        grouped_values = {group_name: [] for group_name in group_ranges}
        for row in rows:
            grouped_values[row['difficulty']].append(float(row[metric_key]))

        summary = {}
        for group_name, values in grouped_values.items():
            summary[group_name] = {
                'range': group_ranges[group_name],
                'count': len(values),
                'mace': float(np.mean(values)) if values else None,
            }
        return summary

    def _write_detailed_csv(self, rows: list[dict]) -> str | None:
        if self.report_dir is None or not rows:
            return None

        os.makedirs(self.report_dir, exist_ok=True)
        csv_path = os.path.join(self.report_dir, f"{self.report_prefix}_per_sample_metrics.csv")
        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return csv_path

    def _finalize_detailed_report(self, rows: list[dict], metric_key: str):
        rows = self._assign_difficulty_groups(rows)
        self.last_detailed_report = {
            'metric_key': metric_key,
            'group_metrics': self._summarize_group_metrics(rows, metric_key),
            'csv_path': self._write_detailed_csv(rows),
            'num_samples': len(rows),
        }

    def set_epoch(self, epoch):
        """由 train_loop 调用，更新当前 epoch"""
        self.current_epoch = epoch

    def _get_current_stage_params(self):
        """确定使用的 warp2 扰动范围 Hk"""
        perturb_range_Hk = self.warp2
        # 确定当前阶段
        current_stage = max(0, len(self.cumulative_epochs) - 2)
        for i, end_epoch in enumerate(self.cumulative_epochs[1:]):
            if self.current_epoch <= end_epoch:
                current_stage = i
                break
        
        is_refine_stage = current_stage > 0
        
        return perturb_range_Hk, is_refine_stage
        
    def _generate_dynamic_data(self, data):
        """
        实现 IDR 动态数据生成逻辑: 
        I_fixA, I_warpB, H_global_true -> [net_copy] -> I_refined -> I_warp2B, d_gt
        
        Args:
            data (dict): 来自 DataLoader 的基础数据
            
        Returns:
            dict: 包含生成的动态数据
        """
        # 0. 获取 IDR 状态
        perturb_range_Hk, is_refine_stage = self._get_current_stage_params()
        
        # 1. 数据准备 (将基础数据从 CPU 转移到 GPU)
        I_fixA_crop = data['I_fixA_crop'].to(self.device)   # B x 3 x H_c x W_c
        I_fixB = data['I_fixB'].to(self.device)             # B x 3 x H x W
        I_warpB_crop = data['I_warpB_crop'].to(self.device) # B x 3 x H_c x W_c
        H_global_true = data['H_global_true'].to(self.device) # B x 3 x 3
        d_eval = data.get('d_eval')
        if isinstance(d_eval, torch.Tensor) and d_eval.numel() > 0:
            d_eval = d_eval.to(self.device)
        else:
            d_eval = None
        
        B = I_fixB.shape[0]
        
        if not is_refine_stage:
            # ============ 初始阶段：对 I_warpB 施加随机扰动得到 I_warp2B =============
            I_warp2B = torch.zeros_like(I_fixB)
            I_warp2B_crop = torch.zeros((B, 3, self.H - 2*self.marginal, self.W - 2*self.marginal), device=self.device)
            d_gt = torch.zeros((B, 4, 2), device=self.device)
            H_local_gt = torch.zeros((B, 3, 3), device=self.device)
            H_global_gt = torch.zeros((B, 3, 3), device=self.device)
            
            for b in range(B):
                # 生成随机扰动 Hk (对每个批次单独生成)
                if d_eval is None:
                    H_local_gt_np, H_global_gt_np, d_gt_np = get_homography_with_adaptation(
                        self.H,
                        self.W,
                        self.marginal,
                        perturb_range_Hk,
                        adaptation=self.homography_adaptation,
                    )
                else:
                    d_gt_np = d_eval[b].detach().cpu().numpy().astype(np.float32)
                    H_local_gt_np, H_global_gt_np = delta_to_homography_matrices(
                        d_gt_np,
                        self.H,
                        self.W,
                        self.marginal,
                    )
                d_gt[b] = torch.from_numpy(d_gt_np).to(self.device)
                H_local_gt[b] = torch.from_numpy(H_local_gt_np).to(self.device)
                H_global_gt[b] = torch.from_numpy(H_global_gt_np).to(self.device)
                
                # 对单张图像应用透视变换
                # I_warp2B[b] = Hk^-1 * H_true^-1 * I_fixB[b]
                I_fixB_np = I_fixB[b].permute(1, 2, 0).cpu().numpy()  # 转为numpy格式 (H,W,C)
                H_combined = (
                    safe_inverse_homography(H_global_gt_np, context="H_global_gt")
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
            
            # 初始阶段 d_hat, I_refined 为空
            d_hat = torch.zeros((B, 4, 2), device=self.device)
            I_refined = torch.zeros_like(I_fixB)
            
        else:
            # ================= 细化阶段：I_warp -> I_refined -> I_warp2 =================
            
            # --- 1. 使用 net_copy 预测细化单应性矩阵 ---
            with torch.no_grad():
                input_pair = torch.cat([I_warpB_crop, I_fixA_crop], dim=1)  # B x 6 x h x w
                d_hat = self.net_copy(input_pair, test_mode=True)           # B x 2 x 2 x 2 (取 lev0 预测)
                d_hat = reshape_delta_222_to_42(d_hat)
                d_hat_np = d_hat.cpu().numpy()

                # 计算用来细化的全局单应性矩阵 H_hat
                B = I_fixB.shape[0]
                H_hat_global = torch.zeros((B, 3, 3), device=self.device)
                for b in range(B):
                    H_hat_global_np, d_hat_np[b] = self._prediction_delta_to_global_homography(d_hat_np[b], b)
                    H_hat_global[b] = torch.from_numpy(H_hat_global_np)
            
            # --- 2. 数据细化 (I_refined = H_hat * I_warpB = H_hat * H_true^-1 * I_fixB) ---
            # 初始化结果张量
            B = I_fixB.shape[0]
            I_refined = torch.zeros_like(I_fixB)
            
            # 对每个批次图像单独处理
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
            
            # --- 3. 对 I_refined 施加随机扰动 H_k ---
            # 初始化结果张量
            I_warp2B = torch.zeros_like(I_fixB)
            I_warp2B_crop = torch.zeros((B, 3, self.H - 2*self.marginal, self.W - 2*self.marginal), device=self.device)
            d_gt = torch.zeros((B, 4, 2), device=self.device)
            H_global_gt = torch.zeros((B, 3, 3), device=self.device)
            H_local_gt = torch.zeros((B, 3, 3), device=self.device)
            
            # 对每个批次图像单独处理
            for b in range(B):
                # 生成随机扰动 Hk (对每个批次单独生成)
                if d_eval is None:
                    H_local_gt_np, H_global_gt_np, d_gt_np = get_homography_with_adaptation(
                        self.H,
                        self.W,
                        self.marginal,
                        perturb_range_Hk,
                        adaptation=self.homography_adaptation,
                    )
                else:
                    d_gt_np = d_eval[b].detach().cpu().numpy().astype(np.float32)
                    H_local_gt_np, H_global_gt_np = delta_to_homography_matrices(
                        d_gt_np,
                        self.H,
                        self.W,
                        self.marginal,
                    )
                d_gt[b] = torch.from_numpy(d_gt_np).to(self.device)
                H_global_gt[b] = torch.from_numpy(H_global_gt_np).to(self.device)
                H_local_gt[b] = torch.from_numpy(H_local_gt_np).to(self.device)
                
                # 对单张图像应用透视变换
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
        
        # 4. 返回生成的动态数据
        return {
            'd_hat': d_hat,
            'I_refined': I_refined,
            'I_warp2B': I_warp2B,
            'I_warp2B_crop': I_warp2B_crop,
            'd_gt': d_gt,
            'H_local_gt': H_local_gt,
            'H_global_gt': H_global_gt
        }
    
    def evaluate(self, val_loader):
        """
        评估模型性能
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            tuple: 包含评估结果的元组
        """
        assert self.config['validation']['batch_size'] > 0, "batchsize > 0"
        self.last_detailed_report = None
        
        total_mace_0 = torch.empty(0)  # refine MACE
        total_mace_1 = torch.empty(0)  # Warp2-Fix MACE
        total_mace_2 = torch.empty(0)  # Warp-Fix MACE
        timeall = []
        detailed_rows = [] if self.report_dir is not None else None
        
        # 正常有监督训练
        if self.config['training']['strategy'] == 'supervised':
            compare_figs = []
            for i_batch, data_blob in enumerate(val_loader):
                img1 = data_blob['I_warpB_crop'].to(self.device)
                img2 = data_blob['I_fixA_crop'].to(self.device)
                d_true = data_blob['d_true'].to(self.device)       # (B, 4, 2)
                
                time_start = time.time()
                d_pred = self.model(torch.cat([img1, img2], dim=1), test_mode=True)   # (B, 2, 2, 2)
                time_end = time.time()
                timeall.append(time_end - time_start)
                
                d_true = reshape_delta_42_to_222(d_true)  # (B, 2, 2, 2)
                
                mace_, mace_vec = self._compute_mace(d_true, d_pred)
                total_mace_2 = torch.cat([total_mace_2, mace_vec], dim=0)   # 所有图片的MACE误差数组
                final_mace = torch.mean(total_mace_2).item()                # 截至目前所有图片的平均MACE误差
                if detailed_rows is not None:
                    detailed_rows.extend(
                        self._build_sample_rows(
                            data_blob=data_blob,
                            d_true_222=d_true,
                            primary_metric_key='mace',
                            primary_metric_values=mace_vec,
                        )
                    )
                
                # 每 vis_interval 控制台输出 (推理时间和MACE误差) 并可视化配准结果
                vis_num = self.config['validation']['vis_num']
                vis_interval = max(1, len(val_loader) // max(1, vis_num))
                if i_batch % vis_interval == 0:
                    print(f"Inference Time of image {i_batch}: ", time_end - time_start)    # 这张图片的推理时间
                    print(f"MACE of image {i_batch}: ", mace_.mean())       # 这张图片的平均MACE误差
                    print("Mean MACE so far: ", final_mace)                 # 截至目前所有图片的平均MACE误差
                    
                    compare_fig = visualize_compare(
                        d_pred[0],
                        d_true[0],
                        data_blob['I_fixA'][0],
                        data_blob['I_warpB'][0],
                        self.config['data']['marginal']
                    )
                    compare_figs.append(compare_fig)
            
            # 输出最终的平均MACE误差
            print(f"Mean MACE final: {final_mace}")
            print(f"Mean inference time: {np.mean(np.array(timeall[1:-1]))}")
            if detailed_rows is not None:
                self._finalize_detailed_report(detailed_rows, metric_key='mace')
            
            # 返回验证集的平均MACE误差
            return final_mace, compare_figs
        
        # IDR 自监督训练
        elif self.config['training']['strategy'] == 'self-supervised':
            figs_warp_refine = []
            compare_figs_warp2_refine = []
            compare_figs_warp_fix = []
            
            for i_batch, data_blob in enumerate(val_loader):
                # 1. 加载基础数据
                I_fixA_crop = data_blob['I_fixA_crop'].to(self.device)
                I_warpB_crop = data_blob['I_warpB_crop'].to(self.device)
                d_true = data_blob['d_true'].to(self.device)
                
                # 2. 生成动态数据 (I_refined, I_warp2B, I_warp2B_crop, d_gt)
                dynamic_data = self._generate_dynamic_data(data_blob)
                I_warp2B_crop = dynamic_data['I_warp2B_crop']
                d_gt = dynamic_data['d_gt']
                d_hat = dynamic_data['d_hat']
                
                # 3. 转换 d_true 和 d_gt 为 222 格式
                d_true = reshape_delta_42_to_222(d_true)  # (B, 2, 2, 2)
                d_gt = reshape_delta_42_to_222(d_gt)      # (B, 2, 2, 2)
                d_hat = reshape_delta_42_to_222(d_hat)    # (B, 2, 2, 2)
                
                # 4. refine 预测误差
                mace_0, mace_0vec = self._compute_mace(d_true, d_hat)
                total_mace_0 = torch.cat([total_mace_0, mace_0vec], dim=0)
                final_mace_0 = torch.mean(total_mace_0).item()

                # 5. Warp2-Fix 预测结果 -- 伪监督训练
                d_pred_warp2_fix = self.model(torch.cat([I_warp2B_crop, I_fixA_crop], dim=1), test_mode=True)  # (B, 2, 2, 2)
                mace_1, mace_1vec = self._compute_mace(d_gt, d_pred_warp2_fix)
                total_mace_1 = torch.cat([total_mace_1, mace_1vec], dim=0)   # 所有图片的MACE误差数组
                final_mace_1 = torch.mean(total_mace_1).item()               # 截至目前所有图片的平均MACE误差
                
                # 6. Warp-Fix 预测结果 -- 最终目标
                d_pred_warp_fix = self.model(torch.cat([I_warpB_crop, I_fixA_crop], dim=1), test_mode=True)  # (B, 2, 2, 2)
                mace_2, mace_2vec = self._compute_mace(d_true, d_pred_warp_fix)
                total_mace_2 = torch.cat([total_mace_2, mace_2vec], dim=0)
                final_mace_2 = torch.mean(total_mace_2).item()
                if detailed_rows is not None:
                    detailed_rows.extend(
                        self._build_sample_rows(
                            data_blob=data_blob,
                            d_true_222=d_true,
                            primary_metric_key='warp_fix_mace',
                            primary_metric_values=mace_2vec,
                            extra_metrics={
                                'refine_mace': mace_0vec,
                                'warp2_fix_mace': mace_1vec,
                            },
                        )
                    )
                
                # 7. 每 vis_interval 控制台输出 MACE误差 与 配准结果可视化
                vis_num = self.config['validation']['vis_num']
                vis_interval = max(1, len(val_loader) // max(1, vis_num))
                if i_batch % vis_interval == 0:
                    print(f"Refine MACE of image {i_batch}: ", mace_0.mean())
                    print("Refine Mean MACE so far: ", final_mace_0)
                    print(f"Warp2-Fix MACE of image {i_batch}: ", mace_1.mean())       # 这张图片的平均MACE误差
                    print("Warp2-Fix Mean MACE so far: ", final_mace_1)                # 截至目前所有图片的平均MACE误差
                    print(f"Warp-Fix MACE of image {i_batch}: ", mace_2.mean())
                    print("Warp-Fix Mean MACE so far: ", final_mace_2)
                    
                    fig_warp_refine = visualize_warp_refine(
                        I_warp = data_blob['I_warpB'][0],
                        I_refined = dynamic_data['I_refined'][0],
                        mace = mace_0vec[0],
                    )
                    compare_fig_warp2_warp = visualize_compare(
                        d_pred = d_pred_warp2_fix[0],
                        d_true = d_gt[0],
                        I_fix = dynamic_data['I_refined'][0],
                        I_warp = dynamic_data['I_warp2B'][0],
                        marginal = self.config['data']['marginal']
                    )
                    compare_fig_warp_fix = visualize_compare(
                        d_pred = d_pred_warp_fix[0],
                        d_true = d_true[0],
                        I_fix = data_blob['I_fixA'][0],
                        I_warp = data_blob['I_warpB'][0],
                        marginal = self.config['data']['marginal']
                    )
                    figs_warp_refine.append(fig_warp_refine)
                    compare_figs_warp2_refine.append(compare_fig_warp2_warp)
                    compare_figs_warp_fix.append(compare_fig_warp_fix)
            
            # 输出最终的平均MACE误差
            print(f"Warp2-Fix Mean MACE final: {final_mace_1}")
            print(f"Warp-Fix Mean MACE final: {final_mace_2}")
            if detailed_rows is not None:
                self._finalize_detailed_report(detailed_rows, metric_key='warp_fix_mace')
            
            # 返回验证集的平均MACE误差和可视化图
            return final_mace_0, final_mace_1, final_mace_2, figs_warp_refine, compare_figs_warp2_refine, compare_figs_warp_fix
