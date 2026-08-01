import torch
import torch.nn as nn
import logging
from torch.utils.tensorboard import SummaryWriter
import os
import shutil
import numpy as np

from engine.trainer import Trainer # 假设 Trainer 封装了单步优化逻辑
from models.base_model_factory import build_base_model
from training.checkpoints import save_checkpoint

logger = logging.getLogger(__name__)

def sync_net_copy(net_main, net_copy, device):
    """
    将主模型参数复制到冻结的副本模型 (net_copy)，用于下一轮伪标签生成。
    使用strict=False处理无lev1的情况
    """
    logger.info("Synchronizing net_copy model parameters...")
    net_copy.load_state_dict(net_main.state_dict(), strict=False)
    net_copy.eval()
    for param in net_copy.parameters():
        param.requires_grad = False
    net_copy.to(device)
    logger.info("net_copy synchronized and frozen.")


def train_loop_idr(net, train_loader, val_loader, optimizer, scheduler, config, device, log_dir, start_epoch=1):
    """
    核心训练循环，实现 IDR 风格的迭代伪监督训练机制。
    
    Args:
        net: 主模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        config: 配置字典
        device: 训练设备
        log_dir: 日志目录
        start_epoch: 开始训练的epoch编号（用于resume）
    """

    epoch_list = config['training']['epoch_list']
    epochs = sum(epoch_list)
    best_mace = float('inf')
    writer = SummaryWriter(log_dir+'/tensorboard')
    
    # 1. 初始化 net_copy（用于IDR迭代细化）
    net_copy = build_base_model(config)
    net_copy.to(device) # 移到 GPU 上
    
    # 如果是恢复训练，确保 net_copy 与主模型同步
    if start_epoch > 1:
        logger.info(f"Resuming training from epoch {start_epoch}, synchronizing net_copy with main model...")
        sync_net_copy(net, net_copy, device)
    
    # 2. 初始化 Trainer
    trainer = Trainer(net, optimizer, device, config, net_copy=net_copy, current_epoch=start_epoch)

    # 3. 主训练循环
    for epoch in range(start_epoch, epochs+1):
        # A. IDR 同步机制：更新 net_copy
        cumulative_epochs = np.cumsum(config['training']['epoch_list']).tolist()
        is_stage_start = False
        if epoch > epoch_list[0]:
            # 完成第一个阶段后，才开始迭代
            for cumulative_end in cumulative_epochs[:-1]:
                if epoch == cumulative_end + 1:
                    is_stage_start = True
                    break
        
        # 更新 Trainer 中的 current_epoch
        trainer.set_epoch(epoch)

        logger.info(f"--- Epoch {epoch}/{epochs} ---")
        net.train()

        if is_stage_start:
            # 使用 net_copy，且更新 net_copy
            logger.info(f"IDR Refinement active. net_copy SYNCHRONIZING at epoch {epoch}.")
        elif epoch > epoch_list[0]:
            # 使用 net_copy， 暂不更新
            logger.info(f"IDR Refinement active. Using previous net_copy. Current epoch {epoch}.")
        else:
            # 第一个阶段，尚不使用 net_copy 进行细化
            logger.info(f"IDR first stage. Current epoch {epoch}.")
        
        total_loss = 0
        num_batches = len(train_loader)
        
        # B. 迭代训练
        for batch_idx, data in enumerate(train_loader):
            # data 包含 I_warp2, I_fix, d_gt (伪标签)
            loss = trainer.train_step(data)
            total_loss += loss.item()
            
            if batch_idx % config['logging']['log_interval'] == 0:
                avg_loss = total_loss / (batch_idx + 1)
                logger.info(f"Epoch {epoch} [{batch_idx}/{num_batches}] "
                            f"Loss: {loss.item():.4f} (Avg: {avg_loss:.4f}) "
                            f"LR: {optimizer.param_groups[0]['lr']:.6f}")
                writer.add_scalar('train/loss', loss.item(), epoch * num_batches + batch_idx)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch * num_batches + batch_idx)

        # C. 记录每个 epoch 的平均 loss
        avg_epoch_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch} completed. Avg Loss: {avg_epoch_loss:.4f}")
        writer.add_scalar('train/avg_epoch_loss', avg_epoch_loss, epoch)

        # D. 评估
        if epoch % config['validation']['val_interval'] == 0:
            print(f"===== 开始验证 (epoch {epoch}) =====")
            net.eval()

            from evaluation.myevaluate import Evaluator
            evaluator = Evaluator(net, config['training']['device'], config, net_copy=net_copy)
            # 更新 Evaluator 中的 current_epoch
            evaluator.set_epoch(epoch)
                
            # 执行评估
            if config['training']['strategy'] == 'supervised':
                mean_mace_final, figs = evaluator.evaluate(val_loader)
                logger.info(f"Validation Metrics (Mean MACE final): {mean_mace_final:.4f}")
                writer.add_scalar('val/mean_mace', mean_mace_final, epoch)
                for idx, fig in enumerate(figs):
                    writer.add_figure(f'val/fig/{idx}', fig, epoch)
                best_mace_candidate = mean_mace_final
            elif config['training']['strategy'] == 'self-supervised':
                mean_mace_final_refine, \
                mean_mace_final_warp2_fix, \
                mean_mace_final_warp_fix, \
                figs_warp_refine, \
                figs_warp2_warp, \
                figs_warp_fix = evaluator.evaluate(val_loader)
                logger.info(f"Refine Metrics (Mean MACE final): {mean_mace_final_refine:.4f}")
                logger.info(f"Warp2-Fix validation Metrics (Mean MACE final): {mean_mace_final_warp2_fix:.4f}")
                logger.info(f"Warp-Fix validation Metrics (Mean MACE final): {mean_mace_final_warp_fix:.4f}")
                writer.add_scalar('val/mean_mace_refine', mean_mace_final_refine, epoch)
                writer.add_scalar('val/mean_mace_warp2_fix', mean_mace_final_warp2_fix, epoch)
                writer.add_scalar('val/mean_mace_warp_fix', mean_mace_final_warp_fix, epoch)
                for idx, fig in enumerate(figs_warp_refine):
                    writer.add_figure(f'val/fig_warp_refine/{idx}', fig, epoch)
                for idx, fig in enumerate(figs_warp2_warp):
                    writer.add_figure(f'val/fig_warp2_refine/{idx}', fig, epoch)
                for idx, fig in enumerate(figs_warp_fix):
                    writer.add_figure(f'val/fig_warp_fix/{idx}', fig, epoch)
                best_mace_candidate = mean_mace_final_warp_fix

            # 保存mean_mace最小的模型
            if best_mace_candidate < best_mace:
                best_mace = best_mace_candidate
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_mace': best_mace,
                }, checkpoint_dir=os.path.join(log_dir, 'checkpoint'), filename=f'best_model.pth')
            net.train()

        # E. 更新学习率
        scheduler.step()

        # F. 每 save_interval 个 epoch 保存模型
        if epoch % config['logging']['save_interval'] == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_dir=os.path.join(log_dir, 'checkpoints'), filename=f'checkpoint_e{epoch}.pth')

    logger.info("IDR Self-Supervised training finished.")
    writer.close()


def train_loop(net, train_loader, val_loader, optimizer, scheduler, config, device, log_dir, start_epoch=1):
    """
    核心训练循环，实现传统监督训练。
    
    Args:
        net: 主模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        config: 配置字典
        device: 训练设备
        log_dir: 日志目录
        start_epoch: 开始训练的epoch编号（用于resume）
    """
    epochs = config['training']['epochs']
    best_mace = float('inf')
    writer = SummaryWriter(log_dir+'/tensorboard')
    
    # 1. 初始化 Trainer, 负责处理模型的单次前向、损失计算 (L1 Loss on d_gt)
    trainer = Trainer(net, optimizer, device, config)

    # 2. 主训练循环
    for epoch in range(start_epoch, epochs + 1):
        logger.info(f"--- Epoch {epoch}/{epochs} ---" )
        net.train() # 确保主模型处于训练模式
        
        total_loss = 0
        num_batches = len(train_loader)
        
        # 迭代训练
        for batch_idx, data in enumerate(train_loader):
            loss = trainer.train_step(data)
            total_loss += loss.item()

            # 记录 loss, lr
            if batch_idx % config['logging']['log_interval'] == 0:
                avg_loss = total_loss / (batch_idx + 1)
                logger.info(f"Epoch {epoch} [{batch_idx}/{num_batches}] "
                            f"Loss: {loss.item():.4f} (Avg: {avg_loss:.4f}) "
                            f"LR: {optimizer.param_groups[0]['lr']:.6f}")
                writer.add_scalar('train/loss', loss.item(), epoch * num_batches + batch_idx)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch * num_batches + batch_idx)

        # 记录每个 epoch 的平均 loss
        avg_epoch_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch} completed. Avg Loss: {avg_epoch_loss:.4f}")
        writer.add_scalar('train/avg_epoch_loss', avg_epoch_loss, epoch)

        # 每 val_interval 个 epoch 进行一次验证
        if epoch % config['validation']['val_interval'] == 0:
            print(f"===== 开始验证 (epoch {epoch}) =====")
            net.eval()
            from evaluation.myevaluate import Evaluator
            evaluator = Evaluator(net, device, config)
            mean_mace_final_val, figs = evaluator.evaluate(val_loader)
            logger.info(f"Validation Metrics (Mean MACE final): {mean_mace_final_val:.4f}")
            writer.add_scalar('val/mean_mace', mean_mace_final_val, epoch)
            for idx, fig in enumerate(figs):
                writer.add_figure(f'val/fig_{idx}', fig, epoch)

            # 保存mean_mace最小的模型
            if mean_mace_final_val < best_mace:
                best_mace = mean_mace_final_val
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_mace': best_mace,
                }, checkpoint_dir=os.path.join(log_dir, 'checkpoints1'), filename=f'best_model.pth')
            net.train()

        # 更新学习率
        scheduler.step()

        # 每 save_interval 个 epoch 保存模型
        if epoch % config['logging']['save_interval'] == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_dir=os.path.join(log_dir, 'checkpoints1'), filename=f'checkpoint_e{epoch}.pth')

    logger.info("Supervised training finished.")
    writer.close()
