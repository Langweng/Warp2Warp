# utils/io.py
import yaml
import logging
import os
import sys
from datetime import datetime

# ====================================================================
# A. Configuration
# ====================================================================

def load_config(config_path):
    """加载 YAML 配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, save_path):
    """保存配置到 YAML 文件"""
    with open(save_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def _yaml_lines(data):
    """将字典格式化为适合逐行写入日志的 YAML 文本。"""
    dumped = yaml.safe_dump(
        data,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True
    )
    return dumped.rstrip().splitlines()
        
# ====================================================================
# B. Logging
# ====================================================================

def setup_logger(log_dir, exp_name="train"):
    """
    初始化全局日志系统，将日志输出到文件和控制台。
    """
    log_path = os.path.join(log_dir, f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 根 Logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除旧的 handlers
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # File Handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    logging.info(f"Logger initialized. Log file: {log_path}")
    return log_path


def log_experiment_config(config, config_source=None, runtime_args=None, command=None):
    """
    将本次实验的配置快照写入日志文件，便于回溯实验参数。
    """
    logging.info("=" * 80)
    logging.info("Experiment configuration snapshot")

    if config_source is not None:
        logging.info(f"Config source: {os.path.abspath(config_source)}")

    if command is not None:
        logging.info(f"Command: {command}")

    if runtime_args:
        logging.info("Runtime args:")
        for line in _yaml_lines(runtime_args):
            logging.info(line)

    logging.info("Resolved config:")
    for line in _yaml_lines(config):
        logging.info(line)

    logging.info("=" * 80)
