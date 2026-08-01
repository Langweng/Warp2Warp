import os
from unittest import result
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from models.ihn_utils import reshape_delta_222_to_42


def save_debug_image(image: np.ndarray, filename: str, debug_dir: str = "/media/data/xwh/code/Self-Supervised-Homo/debug_view"):
    """
    Debug only
    保存调试图像到指定目录。
    
    Args:
        image (np.ndarray): 要保存的图像，形状为 (H, W, C) 或 (H, W)，支持 float32 [0,1] 或 uint8 [0,255]
        filename (str): 保存的文件名，不包含扩展名
        debug_dir (str): 保存目录的路径
    """
    # 确保调试目录存在
    os.makedirs(debug_dir, exist_ok=True)
    
    # 处理图像数据类型
    img = image.copy()
    if img.dtype == np.float32 and np.max(img) <= 1.0:
        # 将 [0,1] 范围的 float32 转换为 [0,255] 的 uint8
        img = (img * 255).astype(np.uint8)
    
    # 添加时间戳以避免覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    full_filename = f"{debug_dir}/{filename}_{timestamp}.png"
    
    # 保存图像
    if len(img.shape) == 3 and img.shape[2] == 3:
        # 确保是 BGR 格式（OpenCV 默认格式）
        cv2.imwrite(full_filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        # 灰度图或单通道图
        cv2.imwrite(full_filename, img)
        
    print(f"Debug image saved: {full_filename}")


def save_debug_image_grid(images: list, labels: list, filename: str, 
                          debug_dir: str = "/media/data/xwh/code/Self-Supervised-Homo/debug_view",
                          n_cols: int = 2):
    """
    Debug only
    保存多张图像组成的网格到指定目录。
    
    Args:
        images (list): 图像列表，每个元素是 (H, W, C) 或 (H, W) 的 numpy 数组
        labels (list): 图像标签列表，与 images 一一对应
        filename (str): 保存的文件名，不包含扩展名
        debug_dir (str): 保存目录的路径
        n_cols (int): 网格的列数
    """
    # 确保调试目录存在
    os.makedirs(debug_dir, exist_ok=True)
    
    # 计算网格形状
    n_rows = (len(images) + n_cols - 1) // n_cols
    
    # 创建图像网格
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # 显示每张图像
    for i, (img, ax) in enumerate(zip(images, axes)):
        if len(img.shape) == 3 and img.shape[2] == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap='gray')
        ax.set_title(labels[i])
        ax.axis('off')
    
    # 隐藏多余的子图
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    # 添加时间戳以避免覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    full_filename = f"{debug_dir}/{filename}_{timestamp}.png"
    
    # 保存网格图像
    plt.tight_layout()
    plt.savefig(full_filename, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Debug image grid saved: {full_filename}")


def visualize_homo(
    original_image, 
    warp_image, 
    warp_image_crop,
    fpt_pair
    ):
    """
    可视化原始图像和变换后的图像并保存到指定目录
    可用于观察扭曲数据集创建是否正确
    Args:
        original_image: 原始图像
        warp_image: 应用H^-1变换后的图像
        warp_image_crop: warped_image裁剪后的图像
        fpt_pair: 源角点和目标角点对，[fpt_src, fpt_tar]
    """
    # 解析角点对
    fpt_src, fpt_tar = fpt_pair
    
    # 创建保存目录
    save_dir = "/media/data/xwh/code/Self-Supervised-Homo/debug_view"
    os.makedirs(save_dir, exist_ok=True)
    
    # 生成唯一的文件名（使用时间戳和随机ID）
    import time
    import uuid
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    random_id = uuid.uuid4().hex[:8]
    filename = f'visualize_homo_{timestamp}_{random_id}.png'
    save_path = os.path.join(save_dir, filename)
    
    # 创建1x3子图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # (1,1) 位置显示原始图像，并用四边形框出源角点和目标角点
    axes[0].set_title('Original Image with Corner Points')
    axes[0].imshow(original_image)
    
    # 绘制源角点构成的四边形（绿色）
    # 四边形连接顺序：从[左上、右上、左下、右下]改为[左上、右上、右下、左下]
    src_order = [0, 1, 3, 2, 0]
    src_polygon = fpt_src[src_order]
    axes[0].plot(src_polygon[:, 0], src_polygon[:, 1], 'g-', linewidth=2, label='Source Points')
    axes[0].plot(fpt_src[:, 0], fpt_src[:, 1], 'go', markersize=6)
    
    # 绘制目标角点构成的四边形（红色）
    tar_order = [0, 1, 3, 2, 0]
    tar_polygon = fpt_tar[tar_order]
    axes[0].plot(tar_polygon[:, 0], tar_polygon[:, 1], 'r-', linewidth=2, label='Target Points')
    axes[0].plot(fpt_tar[:, 0], fpt_tar[:, 1], 'ro', markersize=6)
    
    axes[0].legend()
    axes[0].axis('off')
    
    # (1,2) 位置显示逆变换后的图像
    axes[1].set_title('Warp Image (H^-1)')
    axes[1].imshow(warp_image)
    axes[1].axis('off')
    
    # (1,3) 位置显示裁剪后的逆变换图像
    axes[2].set_title('Cropped Warp Image')
    axes[2].imshow(warp_image_crop)
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化结果已保存到: {save_path}")

def visualize_compare(
    d_pred: torch.Tensor,
    d_true: torch.Tensor,
    I_fix: torch.Tensor,
    I_warp: torch.Tensor,
    marginal: int
    ):
    """
    验证时可视化预测角点偏移与真实角点偏移
    Args:
        d_pred: 预测的角点偏移, (2x2x2)
        d_true: 真实的角点偏移, (2x2x2)
        I_fix: 固定图像, (C, H, W)
        I_warp: 扭曲图像, (C, H, W)
        marginal: 图像裁剪的边距
    
    Returns:
        fig: 可视化图
    """
    d_pred = d_pred.cpu().detach().numpy()
    d_true = d_true.cpu().detach().numpy()
    I_fix = I_fix.permute(1, 2, 0).cpu().detach().numpy()
    I_warp = I_warp.permute(1, 2, 0).cpu().detach().numpy()
    
    # 确保图像数据在正确范围内
    # if I_fix.max() <= 1.0:
    #     I_fix = (I_fix * 255).astype(np.uint8)
    # if I_warp.max() <= 1.0:
    #     I_warp = (I_warp * 255).astype(np.uint8)
    
    h, w = I_warp.shape[:2]
    
    pt_src = np.array([
        [marginal, marginal],                 # 左上
        [w - 1 - marginal, marginal],         # 右上
        [marginal, h - 1 - marginal],         # 左下
        [w - 1 - marginal, h - 1 - marginal]  # 右下
    ], dtype=np.float32)
    
    # 将d_true和d_pred的形状从(2x2x2)转换为(4x2)格式
    d_true_flat = reshape_delta_222_to_42(d_true)
    d_pred_flat = reshape_delta_222_to_42(d_pred)
    
    # 计算目标角点
    pt_true = pt_src + d_true_flat
    pt_pred = pt_src + d_pred_flat
    
    # 创建图形和两个子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=200)
    
    # 子图1：显示I_warp，并绘制pt_src围成的绿色矩形
    axes[0].imshow(I_warp)
    axes[0].set_title('I_warp')
    
    # 绘制pt_src围成的绿色矩形
    src_order = [0, 1, 3, 2, 0]  # 左上->右上->右下->左下->左上
    axes[0].plot(pt_src[src_order, 0], pt_src[src_order, 1], 'g-', linewidth=4)
    
    # 标记角点
    for i, pt in enumerate(pt_src):
        axes[0].plot(pt[0], pt[1], 'go', markersize=6)
    
    axes[0].axis('off')
    
    # 子图2：显示I_fix，并绘制真实和预测的四边形
    axes[1].imshow(I_fix)
    axes[1].set_title('I_fix')
    
    true_order = [0, 1, 3, 2, 0]
    axes[1].plot(pt_true[true_order, 0], pt_true[true_order, 1], 'g-', linewidth=4, label='d_true')
    
    pred_order = [0, 1, 3, 2, 0]
    axes[1].plot(pt_pred[pred_order, 0], pt_pred[pred_order, 1], 'r-', linewidth=4, label='d_pred')
    
    # 标记角点
    for i, (pt_t, pt_p) in enumerate(zip(pt_true, pt_pred)):
        axes[1].plot(pt_t[0], pt_t[1], 'go', markersize=6)
        axes[1].plot(pt_p[0], pt_p[1], 'ro', markersize=6)
        
    axes[1].legend(loc='upper right')
    axes[1].axis('off')

    result = fig
    plt.close(fig)
    
    return result


def visualize_warp_refine(
    I_warp: torch.Tensor,
    I_refined: torch.Tensor,
    mace: torch.Tensor,
    ):
    """
    可视化扭曲图像和refined图像
    Args:
        I_warp: 扭曲图像, (C, H, W)
        I_refined: refined图像, (C, H, W)
        mace: MACE值
    
    Returns:
        fig: 可视化图
    """
    # 将张量从 (C, H, W) 转换为 (H, W, C)，并搬到CPU
    I_warp_np = I_warp.permute(1, 2, 0).cpu().detach().numpy()
    I_refined_np = I_refined.permute(1, 2, 0).cpu().detach().numpy()

    # 创建图形和两个子图（左右并排）
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=200)

    # 左图：I_warp
    axes[0].imshow(I_warp_np)
    axes[0].set_title('I_warp')
    axes[0].axis('off')

    # 右图：I_refined
    axes[1].imshow(I_refined_np)
    axes[1].set_title('I_refined')
    axes[1].axis('off')

    # 在整体标题上展示 MACE
    try:
        mace_val = float(mace)
    except Exception:
        # 若不能直接转换为 float，则退回到张量取 item
        mace_val = mace.item() if hasattr(mace, 'item') else mace
    fig.suptitle(f"MACE: {mace_val:.4f}")

    result = fig
    plt.close(fig)
    return result
