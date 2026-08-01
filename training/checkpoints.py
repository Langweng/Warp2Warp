import torch
import os
import logging

logger = logging.getLogger(__name__)

def save_checkpoint(state, checkpoint_dir, filename='checkpoint.pth'):
    """
    保存训练检查点。
    
    Args:
        state (dict): 包含 'epoch', 'model_state_dict', 'optimizer_state_dict' 等的字典。
        checkpoint_dir (str): 检查点保存目录。
        filename (str): 文件名。
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    filepath = os.path.join(checkpoint_dir, filename)
    
    # 保存当前检查点
    torch.save(state, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, scheduler, load_path, load_optimizer=True, load_scheduler=True):
    """
    加载检查点。
    
    Args:
        model (nn.Module): 模型实例。
        optimizer (torch.optim.Optimizer): 优化器实例。
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器实例。
        load_path (str): 检查点文件路径。
        load_optimizer (bool): 是否恢复优化器状态。
        load_scheduler (bool): 是否恢复学习率调度器状态。
        
    Returns:
        int: 下一个开始训练的 epoch 编号。
    """
    if not os.path.exists(load_path):
        logger.warning(f"No checkpoint found at {load_path}")
        return 1

    checkpoint = torch.load(load_path, map_location=lambda storage, loc: storage)
    # checkpoint = {'epoch': ..., 'model_state_dict': ..., 'optimizer_state_dict': ..., 'scheduler_state_dict': ...}
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and load_optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    if scheduler and load_scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    start_epoch = checkpoint.get('epoch', 0) + 1

    restored_items = ['model']
    if optimizer and load_optimizer and 'optimizer_state_dict' in checkpoint:
        restored_items.append('optimizer')
    if scheduler and load_scheduler and 'scheduler_state_dict' in checkpoint:
        restored_items.append('scheduler')

    logger.info(
        f"Loaded {', '.join(restored_items)} from {load_path}. Resuming from epoch {start_epoch}"
    )
    return start_epoch


def align_scheduler_to_epoch(scheduler, target_epoch):
    """
    使用当前配置重新对齐学习率调度器到目标 epoch。

    适用于“只恢复模型参数，但不恢复优化器 / scheduler 状态”的场景。
    """
    if scheduler is None or target_epoch <= 0:
        return

    # 对支持 closed-form 的 scheduler，直接根据 target_epoch 重建当前 lr，
    # 这样修改过的调度配置可以在 resume_model_only 场景下立即生效。
    if hasattr(scheduler, '_get_closed_form_lr'):
        scheduler.last_epoch = target_epoch
        closed_form_lrs = scheduler._get_closed_form_lr()
        for param_group, lr in zip(scheduler.optimizer.param_groups, closed_form_lrs):
            param_group['lr'] = lr
        scheduler._last_lr = closed_form_lrs
        if hasattr(scheduler, '_step_count'):
            scheduler._step_count = target_epoch + 1
    else:
        scheduler.step(target_epoch)

    logger.info(
        "Aligned scheduler to epoch %d using current config. Current lr: %s",
        target_epoch,
        ", ".join(f"{lr:.8f}" for lr in scheduler.get_last_lr()),
    )
