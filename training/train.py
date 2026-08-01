import argparse
import os
import sys
import shlex
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from training.train_loop import train_loop_idr, train_loop
from datasets.warp2_fix_online_dataset import Warp2FixOnlineDataset
from datasets.warp_fix_dataset import WarpFixDataset
from models.base_model_factory import build_base_model
from utils.io import load_config, setup_logger, save_config, log_experiment_config
from utils.seed import set_seed
from training.checkpoints import load_checkpoint, align_scheduler_to_epoch

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Self-Supervised Homography Estimation Training")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (YAML)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for training (覆盖配置文件中的设置，例如 'cuda:0' 或 'cpu')")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint file to resume training from")
    parser.add_argument("--resume_model_only", action="store_true",
                        help="Only load model weights from the resume checkpoint and rebuild optimizer/scheduler from current config")
    parser.add_argument("--reset_lr", action="store_true",
                        help="When resuming without scheduler state, keep the freshly initialized learning rate instead of aligning it to the resume epoch")
    parser.add_argument("--loss", type=str, choices=["l1", "l2", "speedupl1"], default=None,
                        help="Override training.loss.type from the config")
    parser.add_argument("--loss-alpha", type=float, default=None,
                        help="Override training.loss.alpha from the config")
    parser.add_argument("--loss-epsilon", type=float, default=None,
                        help="Override training.loss.epsilon from the config")
    parser.add_argument("--loss-speed-threshold", type=float, default=None,
                        help="Override training.loss.speed_threshold from the config")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. 加载配置
    config = load_config(args.config)
    
    # 如果提供了命令行设备参数，则覆盖配置文件中的设置
    if args.device is not None:
        config['training']['device'] = args.device
        if 'validation' in config and 'device' in config['validation']:
            config['validation']['device'] = args.device
    config['training']['resume_model_only'] = (
        args.resume_model_only or config['training'].get('resume_model_only', False)
    )
    config['training']['reset_lr'] = (
        args.reset_lr or config['training'].get('reset_lr', False)
    )
    if (
        args.loss is not None
        or args.loss_alpha is not None
        or args.loss_epsilon is not None
        or args.loss_speed_threshold is not None
    ):
        loss_cfg = config['training'].setdefault('loss', {})
        if args.loss is not None:
            loss_cfg['type'] = args.loss
        if args.loss_alpha is not None:
            loss_cfg['alpha'] = args.loss_alpha
        if args.loss_epsilon is not None:
            loss_cfg['epsilon'] = args.loss_epsilon
        if args.loss_speed_threshold is not None:
            loss_cfg['speed_threshold'] = args.loss_speed_threshold

    # 2. 设置日志和输出目录
    exp_name = config['experiment']['exp_name']
    log_dir = os.path.join("logs", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = setup_logger(log_dir)
    save_config(config, os.path.join(log_dir, "config.yaml"))
    run_name = os.path.splitext(os.path.basename(log_path))[0]
    save_config(config, os.path.join(log_dir, f"{run_name}_config.yaml"))
    runtime_args = {key: value for key, value in vars(args).items() if value is not None}
    log_experiment_config(
        config,
        config_source=args.config,
        runtime_args=runtime_args,
        command=" ".join(shlex.quote(arg) for arg in sys.argv)
    )
    
    # 3. 设置随机种子
    set_seed(config['training']['seed'])
    
    # 从配置文件获取设备设置
    device = torch.device(config['training']['device'])
    print(f"Using device: {device}")
    
    # 4. 初始化模型
    net = build_base_model(config).to(device)

    # 5. 准备数据加载器
    data_config = config['data']
    training_strategy = config['training'].get('strategy', 'self-supervised')  # 默认使用self-supervised
    
    # 根据训练策略使用不同的数据集
    if training_strategy == 'supervised':
        epochs = config['training']['epochs']
        print(f"Using supervised training strategy with WarpFixDataset")
        train_dataset = WarpFixDataset(
            manifest_path=data_config['train_manifest'],
            image_root=data_config['image_root'],
            config=config,
            dataset_type='train'
        )
        val_dataset = WarpFixDataset(
            manifest_path=data_config['val_manifest'],
            image_root=data_config['image_root'],
            config=config,
            dataset_type='val'
        )

    elif training_strategy == 'self-supervised':
        epochs = sum(config['training']['epoch_list'])
        print(f"Using self-supervised training strategy with Warp2FixOnlineDataset")
        train_dataset = Warp2FixOnlineDataset(
            manifest_path=data_config['train_manifest'],
            image_root=data_config['image_root'],
            config=config,
            dataset_type='train'
        )
        val_dataset = Warp2FixOnlineDataset(
            manifest_path=data_config['val_manifest'],
            image_root=data_config['image_root'],
            config=config,
            dataset_type='val'
        )   

    else:
        raise ValueError(f"Unsupported training strategy: {training_strategy}. Supported strategies are 'supervised' and 'self-supervised'")
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['validation']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=False
    )
    
    # 6. 初始化优化器和学习率调度器
    if config['optimizer']['type'] == 'Adam':
        print(f"Using Adam optimizer with lr={config['optimizer']['lr']} and weight_decay={config['optimizer']['weight_decay']}")
        optimizer = optim.Adam(net.parameters(), 
                            lr=config['optimizer']['lr'], 
                            weight_decay=config['optimizer']['weight_decay'])
    elif config['optimizer']['type'] == 'AdamW':
        print(f"Using AdamW optimizer with lr={config['optimizer']['lr']} and weight_decay={config['optimizer']['weight_decay']}")
        optimizer = optim.AdamW(net.parameters(), 
                            lr=config['optimizer']['lr'], 
                            weight_decay=config['optimizer']['weight_decay'])

    # 定义 LR scheduler
    if config['scheduler']['type'] == 'StepLR':
        print(f"Using StepLR scheduler with step_size={config['scheduler']['step_size']} and gamma={config['scheduler']['gamma']}")
        scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                            step_size=config['scheduler']['step_size'], 
                                            gamma=config['scheduler']['gamma'])
    elif config['scheduler']['type'] == 'CosineAnnealingLR':
        print(f"Using CosineAnnealingLR scheduler with T_max={epochs}")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                        T_max=epochs,
                                                        eta_min=config['scheduler']['eta_min'])
    
    # 7. 处理resume逻辑
    start_epoch = 1
    resume_path = args.resume or config['training'].get('resume') or None
    resume_model_only = config['training'].get('resume_model_only', False)
    reset_lr = config['training'].get('reset_lr', False)
    resume_load_optimizer = config['training'].get('resume_load_optimizer', not resume_model_only)
    resume_load_scheduler = config['training'].get('resume_load_scheduler', not resume_model_only)
    if resume_model_only:
        resume_load_optimizer = False
        resume_load_scheduler = False
    if resume_path is not None:
        start_epoch = load_checkpoint(
            net,
            optimizer,
            scheduler,
            resume_path,
            load_optimizer=resume_load_optimizer,
            load_scheduler=resume_load_scheduler,
        )
        if not resume_load_scheduler and not reset_lr:
            align_scheduler_to_epoch(scheduler, start_epoch - 1)
    
    # 8. 启动训练循环
    print(f"Starting training for {epochs} epochs...")
    print(f"Resuming from epoch {start_epoch}")

    if training_strategy == 'supervised':
        train_loop(
            net=net,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            log_dir=log_dir,
            start_epoch=start_epoch
        )
        
    elif training_strategy == 'self-supervised':
        train_loop_idr(
            net=net,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            log_dir=log_dir,
            start_epoch=start_epoch
        )


if __name__ == "__main__":

    main()
