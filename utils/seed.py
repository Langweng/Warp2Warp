# utils/seed.py
import torch
import numpy as np
import random
import os

def set_seed(seed):
    """
    为所有随机源设置固定的随机种子，以确保实验的可复现性。
    
    Args:
        seed (int or None): 随机种子值，如果为None则不设置随机种子
    """
    if seed is None or seed == 'None':
        return
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 推荐设置，以保证 CUDA 上的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
