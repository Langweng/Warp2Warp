# utils/metrics.py
import torch
import torch.nn.functional as F
import numpy as np
from utils.homography_utils import corners_to_H

def compute_ACE(d_hat: torch.Tensor, H_local_true: torch.Tensor, crop_H: int, crop_W: int) -> float:
    pass