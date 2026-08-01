import torch
import numpy as np
import torchvision.transforms as T

class NormalizeAndToTensor:
    """
    将 numpy 数组图像转换为 PyTorch Tensor 并进行归一化。
    同时支持单个图像和图像对的处理。
    """
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img):
        # 处理单个图像
        if isinstance(img, np.ndarray):
            # 1. To Tensor (HWC -> CHW)
            tensor_img = torch.from_numpy(img).permute(2, 0, 1).float()
            # 2. Normalize
            # 假设输入图像是 [0, 255] 的 float32
            normalized_img = (tensor_img / 255.0 - self.mean[:, None, None]) / self.std[:, None, None]
            return normalized_img
        # 处理图像对
        elif isinstance(img, (list, tuple)) and len(img) == 2:
            I_warp, I_fix = img
            # 1. To Tensor (HWC -> CHW)
            I_warp = torch.from_numpy(I_warp).permute(2, 0, 1).float()
            I_fix = torch.from_numpy(I_fix).permute(2, 0, 1).float()
            # 2. Normalize
            I_warp = (I_warp / 255.0 - self.mean[:, None, None]) / self.std[:, None, None]
            I_fix = (I_fix / 255.0 - self.mean[:, None, None]) / self.std[:, None, None]
            return I_warp, I_fix
        else:
            raise TypeError("Input must be a numpy array or a tuple/list of two numpy arrays")

def get_transforms(config, is_train=True):
    """
    根据配置获取数据集所需的 Transform。
    """
    # 假设跨模态场景使用通用的归一化参数
    mean = config.get('normalize_mean', [0.5, 0.5, 0.5])
    std = config.get('normalize_std', [0.5, 0.5, 0.5])
    
    transforms = []
    
    # 训练阶段可以添加随机颜色抖动、模糊等增强，但要确保 Warp 和 Fix 图像独立增强或联合增强。
    # 对于单应性估计，几何增强（如 RandomRotation, RandomScale）通常在数据生成管线中处理（即 Hx 的生成）。
    
    if is_train and config.get('random_flip', False):
        # 垂直或水平翻转（如果适用），需要确保 H 矩阵也相应更新，但 Warp2FixOnlineDataset 中
        # H_t 是随机生成的，所以这里只在图像级应用，不影响伪标签 H_t 的生成。
        # 几何变换（如翻转）通常放在 transform 外部处理或自定义实现，确保 H 矩阵也随之变化。
        # 为了简化，这里只用基础的 ToTensor 和 Normalize。
        pass
        
    transforms.append(NormalizeAndToTensor(mean, std))
    
    return T.Compose(transforms)