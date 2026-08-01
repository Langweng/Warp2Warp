import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import kornia.geometry.transform as tgm 

# --- 1. 坐标采样工具 ---
def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    # coords should be B x H x W x 2
    xgrid, ygrid = coords.split([1,1], dim=-1) 
    
    # 归一化到 [-1, 1]
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    # F.grid_sample 需要 (B, C, H_in, W_in) 和 (B, H_out, W_out, 2)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def coords_grid(batch, ht, wd):
    """ 生成 HxW 的像素坐标网格 (x, y) """
    # 注意：这里的实现与提供的 utils.py 略有不同，原代码是 (y, x)
    # 我们采用 (x, y) 顺序，即 (w, h)
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd), indexing='ij')
    # coords[0]是y（行）, coords[1]是x（列）。stack[::-1] 变为 (x, y)
    coords = torch.stack(coords[::-1], dim=0).float() 
    return coords[None].expand(batch, -1, -1, -1) # B x 2 x H x W


# --- 2. IHN 内部 Flow/Homography 转换工具 ---

def get_flow_now_4(four_point, h, w, device):
    """
    根据四角点位移和图像尺寸，计算 1/4 特征图上的流场坐标。
    """
    four_point = four_point / 4
    four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)
    four_point_org[:, 0, 0] = torch.Tensor([0, 0])
    four_point_org[:, 0, 1] = torch.Tensor([w-1, 0])
    four_point_org[:, 1, 0] = torch.Tensor([0, h-1])
    four_point_org[:, 1, 1] = torch.Tensor([w-1, h-1])

    four_point_org = four_point_org.unsqueeze(0)
    four_point_org = four_point_org.repeat(four_point.shape[0], 1, 1, 1)
    four_point_new = four_point_org + four_point
    four_point_org = four_point_org.flatten(2).permute(0, 2, 1)
    four_point_new = four_point_new.flatten(2).permute(0, 2, 1)

    H = tgm.get_perspective_transform(four_point_org.contiguous(), four_point_new.contiguous())
    gridy, gridx = torch.meshgrid(torch.linspace(0, w-1, steps=w), torch.linspace(0, h-1, steps=h))
    points = torch.cat((gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0), torch.ones((1, w * h))),
                        dim=0).unsqueeze(0).repeat(four_point.shape[0], 1, 1).to(four_point.device)
    points_new = H.bmm(points)
    points_new = points_new / points_new[:, 2, :].unsqueeze(1)
    points_new = points_new[:, 0:2, :]
    flow = torch.cat((points_new[:, 0, :].reshape(four_point.shape[0], w, h).unsqueeze(1),
                        points_new[:, 1, :].reshape(four_point.shape[0], w, h).unsqueeze(1)), dim=1)
    return flow


def get_flow_now_2(four_point, h, w, device):
    """
    根据四角点位移和图像尺寸，计算 1/2 特征图上的流场坐标。
    """
    four_point = four_point / 2
    four_point_org = torch.zeros((2, 2, 2)).to(four_point.device)
    four_point_org[:, 0, 0] = torch.Tensor([0, 0])
    four_point_org[:, 0, 1] = torch.Tensor([w-1, 0])
    four_point_org[:, 1, 0] = torch.Tensor([0, h-1])
    four_point_org[:, 1, 1] = torch.Tensor([w-1, h-1])

    four_point_org = four_point_org.unsqueeze(0)
    four_point_org = four_point_org.repeat(four_point.shape[0], 1, 1, 1)
    four_point_new = four_point_org + four_point
    four_point_org = four_point_org.flatten(2).permute(0, 2, 1)
    four_point_new = four_point_new.flatten(2).permute(0, 2, 1)

    H = tgm.get_perspective_transform(four_point_org.contiguous(), four_point_new.contiguous())
    gridy, gridx = torch.meshgrid(torch.linspace(0, w-1, steps=w), torch.linspace(0, h-1, steps=h))
    points = torch.cat((gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0), torch.ones((1, w * h))),
                        dim=0).unsqueeze(0).repeat(four_point.shape[0], 1, 1).to(four_point.device)
    points_new = H.bmm(points)
    points_new = points_new / points_new[:, 2, :].unsqueeze(1)
    points_new = points_new[:, 0:2, :]
    flow = torch.cat((points_new[:, 0, :].reshape(four_point.shape[0], w, h).unsqueeze(1),
                        points_new[:, 1, :].reshape(four_point.shape[0], w, h).unsqueeze(1)), dim=1)
    return flow


def predict_flow_hor(four_point_disp, W, H):
    """
    将四角点位移恢复到原始图像尺寸，用于最终输出。
    """
    sz = (H, W)
    flow = coords_grid(four_point_disp.shape[0], sz[0], sz[1])
    flow = flow / torch.tensor([sz[1], sz[0]]).view(1, 2, 1, 1)

    flow_disp = four_point_disp.permute(0, 2, 3, 1) # B x 2 x 2 x 2

    new_grid = F.interpolate(flow_disp.permute(0, 3, 1, 2), size=sz, mode='bilinear', align_corners=True)
    new_grid = new_grid * torch.tensor([sz[1], sz[0]]).view(1, 2, 1, 1)

    return flow + new_grid

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.to(x.device)
    vgrid = torch.autograd.Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output * mask


# --- 3. IHN 损失计算 ---
def compute_weighted_loss(
    pred_list,
    gt_disp,
    alpha=0.8,
    loss_type="l1",
    epsilon=0.1,
    speed_threshold=1.0,
):
    """
    加权多迭代角点位移损失。
    pred_list: List[Tensor], 每次迭代的预测位移 (B, 2, 2, 2)
    gt_disp: Tensor, 真值位移 (B, 2, 2, 2)
    alpha: 越早的预测权重越小，weight_k = alpha ** (K - k - 1)
    loss_type:
      - "l1": 平均绝对误差
      - "l2": 均方误差
      - "speedupl1": MCNet 风格的 speed-up L1, x - 1/(x + epsilon)
        其中 x 为平均绝对误差，仅当 x < speed_threshold 时启用 speed-up 项
    """
    if len(pred_list) == 0:
        return 0.0
    loss_type = str(loss_type).lower()
    if loss_type in ("l1", "mae"):
        pointwise_loss = lambda pred: torch.abs(pred - gt_disp).mean()
    elif loss_type in ("l2", "mse"):
        pointwise_loss = lambda pred: ((pred - gt_disp) ** 2).mean()
    elif loss_type == "speedupl1":
        def pointwise_loss(pred):
            l1_error = torch.abs(pred - gt_disp).mean()
            speed_flag = (l1_error < speed_threshold).to(dtype=l1_error.dtype)
            return l1_error - speed_flag * (1.0 / (l1_error + epsilon))
    else:
        raise ValueError(f"Unsupported loss_type={loss_type!r}. Expected 'l1', 'l2' or 'speedupl1'.")

    K = len(pred_list)
    loss = 0.0
    for k in range(K):
        weight = alpha ** (K - k - 1)
        loss += weight * pointwise_loss(pred_list[k])
    return loss

# --- 4. reshape d ---
def reshape_delta_42_to_222(d_gt):
    """
    将 Bx4x2 格式的目标位移转换为 Bx2x2x2 格式。
        - 第一维保持批次大小不变
        - 第二维: 0表示x方向位移，1表示y方向位移
        - 第三维第四维: 2x2网格，对应[[左上,右上],[左下,右下]]
    """
    batch_size = d_gt.shape[0]
    # 创建新的目标形状 Bx2x2x2 的张量
    d_gt_reshaped = torch.zeros((batch_size, 2, 2, 2), device=d_gt.device)
    
    # 填充x方向位移 (四个角点的x坐标偏移)
    d_gt_reshaped[:, 0, 0, 0] = d_gt[:, 0, 0]  # 左上x
    d_gt_reshaped[:, 0, 0, 1] = d_gt[:, 1, 0]  # 右上x
    d_gt_reshaped[:, 0, 1, 0] = d_gt[:, 2, 0]  # 左下x
    d_gt_reshaped[:, 0, 1, 1] = d_gt[:, 3, 0]  # 右下x
    
    # 填充y方向位移 (四个角点的y坐标偏移)
    d_gt_reshaped[:, 1, 0, 0] = d_gt[:, 0, 1]  # 左上y
    d_gt_reshaped[:, 1, 0, 1] = d_gt[:, 1, 1]  # 右上y
    d_gt_reshaped[:, 1, 1, 0] = d_gt[:, 2, 1]  # 左下y
    d_gt_reshaped[:, 1, 1, 1] = d_gt[:, 3, 1]  # 右下y
    
    return d_gt_reshaped


def reshape_delta_222_to_42(d_gt):
    """
    将 Bx2x2x2 格式的目标位移转换为 Bx4x2 格式。
        - 第一维保持批次大小不变
        - 第二维: 依次表示左上、右上、左下、右下
        - 第三维: 每个角点的x, y坐标偏移
    
    支持输入为PyTorch tensor或numpy数组，输出类型与输入类型保持一致。
    """
    
    if isinstance(d_gt, torch.Tensor):
        batch_size = d_gt.shape[0]
        device = d_gt.device
        dtype = d_gt.dtype
        d_gt_reshaped = torch.zeros((batch_size, 4, 2), device=device, dtype=dtype)
        
        d_gt_reshaped[:, 0, 0] = d_gt[:, 0, 0, 0]  # 左上x
        d_gt_reshaped[:, 0, 1] = d_gt[:, 1, 0, 0]  # 左上y
        d_gt_reshaped[:, 1, 0] = d_gt[:, 0, 0, 1]  # 右上x
        d_gt_reshaped[:, 1, 1] = d_gt[:, 1, 0, 1]  # 右上y
        d_gt_reshaped[:, 2, 0] = d_gt[:, 0, 1, 0]  # 左下x
        d_gt_reshaped[:, 2, 1] = d_gt[:, 1, 1, 0]  # 左下y
        d_gt_reshaped[:, 3, 0] = d_gt[:, 0, 1, 1]  # 右下x
        d_gt_reshaped[:, 3, 1] = d_gt[:, 1, 1, 1]  # 右下y
        
    elif isinstance(d_gt, np.ndarray):
        d_gt_reshaped = np.array([
            [d_gt[0, 0, 0], d_gt[1, 0, 0]],  # 左上 x,y 偏移
            [d_gt[0, 0, 1], d_gt[1, 0, 1]],  # 右上 x,y 偏移
            [d_gt[0, 1, 0], d_gt[1, 1, 0]],  # 左下 x,y 偏移
            [d_gt[0, 1, 1], d_gt[1, 1, 1]]   # 右下 x,y 偏移
        ], dtype=np.float32)
    
    return d_gt_reshaped
