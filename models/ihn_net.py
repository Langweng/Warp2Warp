import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.geometry.transform as tgm # 假设 kornia 已安装
from models.ihn_utils import bilinear_sampler, coords_grid, get_flow_now_4, get_flow_now_2, predict_flow_hor, warp

autocast = torch.cuda.amp.autocast

# ====================================================================
# A. Feature Extractor Blocks (来自 IHN 的 extractor.py)
# ====================================================================

class ResidualBlock(nn.Module):
    '''
    从 IHN 的 extractor.py 复制
    残差块
    '''
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoderQuarter(nn.Module):
    '''
    从 IHN 的 extractor.py 复制
    1/4 特征图编码器
    '''
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoderQuarter, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer2 = self._make_layer(64, stride=1)
        self.layer3 = self._make_layer(96, stride=1)

        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)
        self.conv3 = nn.Conv2d(64, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):

        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = F.max_pool2d(x, 2, stride=2)
        x = self.layer2(x)      # 两层残差块, 64通道，H/2, W/2
        x_64 = self.conv3(x)    # 256通道, H/2, W/2
        x = F.max_pool2d(x, 2, stride=2)
        x = self.layer3(x)      # 两层残差块，96通道, H/4, W/4
        x = self.conv2(x)       # 256通道, H/4, W/4
        if self.training and self.dropout is not None:
            x = self.dropout(x)
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x, x_64          # 返回128通道的特征图，分别为4倍下采样、2倍下采样


# ====================================================================
# B. Correlation Volume Blocks (来自 IHN 的 corr.py)
# ====================================================================

class CorrBlock:
    '''
    从 IHN 的 corr.py 复制
    计算特征图之间的相关性_volume_
    '''
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4, sz=32):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        corr = CorrBlock.corr(fmap1, fmap2,sz)
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2, sz):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.relu(torch.matmul(fmap1.transpose(1, 2), fmap2))
        corr = corr.view(batch, ht, wd, 1, ht, wd)

        return corr


# ====================================================================
# C. Update Block / GMA (来自 update.py)
# ====================================================================

class CNN(nn.Module):
    '''
    从 IHN 的 update.py 复制
    '''
    def __init__(self, input_dim=256):
        super(CNN, self).__init__()

        outputdim = input_dim
        self.layer1 = nn.Sequential(nn.Conv2d(164, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim//8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))

        input_dim = outputdim
        outputdim = input_dim
        self.layer2 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer3 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = outputdim
        outputdim = input_dim
        self.layer4 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride=2))

        input_dim = outputdim
        outputdim_final = outputdim

        ### global motion
        self.layer10 = nn.Sequential(nn.Conv2d(input_dim, outputdim_final, 3,  padding=1, stride=1), nn.GroupNorm(num_groups=(outputdim_final) // 8, num_channels=outputdim_final),
                                     nn.ReLU(), nn.Conv2d(outputdim_final, 2, 1))


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer10(x)

        return x

class CNN_64(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super(CNN_64, self).__init__()

        outputdim = input_dim
        self.layer1 = nn.Sequential(nn.Conv2d(164, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=outputdim//8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))

        input_dim = outputdim
        outputdim = input_dim
        self.layer2 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride=2))

        input_dim = input_dim
        outputdim = input_dim
        self.layer3 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride=2))

        input_dim = input_dim
        outputdim = input_dim
        self.layer4 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride=2))
        input_dim = input_dim
        outputdim = input_dim
        self.layer5 = nn.Sequential(nn.Conv2d(input_dim, outputdim, 3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=(outputdim) // 8, num_channels=outputdim), nn.ReLU(),
                                    nn.MaxPool2d(kernel_size = 2, stride=2))

        outputdim_final = outputdim
        self.layer10 = nn.Sequential(nn.Conv2d(outputdim_final, outputdim_final, 3,  padding=1, stride=1), nn.GroupNorm(num_groups=(outputdim_final) // 8, num_channels=outputdim_final),
                                     nn.ReLU(), nn.Conv2d(outputdim_final, 2, 1))


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer10(x)

        return x

class GMA(nn.Module):
    """
    通用运动感知块 (GRU-like Update Block)，对应 IHN 中的 update_block。
    简化实现：只保留 CNN 分支，忽略权重分支 (CNN_weight) 。
    """
    def __init__(self, args, sz):
        super().__init__()
        self.args = args

        if sz == 32:
            self.cnn = CNN(128)
        elif sz == 64:
            self.cnn = CNN_64(80)
        else:
            raise ValueError(f"GMA 尚不支持输入尺寸 {sz}x{sz}。")
        
    def forward(self, corr, flow):
        delta_flow = self.cnn(torch.cat((corr, flow), dim=1))
        return delta_flow

# ====================================================================
# D. Main Network: IHN (来自 network.py: IHN)
# ====================================================================

class IHN(nn.Module):
    """
    主网络类，实现了 IHN 的迭代预测流程。
    输入：拼接的 (I_warp, I_fix) 图像，B x 6 x H x W
    输出：预测的四角点位移 d_hat (B x 8)
    """
    def __init__(self, config):
        """
        config: 包含迭代次数和层级开关的配置字典。
        """
        super().__init__()
        
        self.iters_lev0 = config.get('iters_lev0', 6) # 1/4 特征图迭代次数
        self.iters_lev1 = config.get('iters_lev1', 0) # 1/2 特征图迭代次数
        self.lev0 = config.get('lev0', True) # 是否使用 1/4 级
        self.lev1 = config.get('lev1', True) # 是否使用 1/2 级
        self.crop_size = int(config.get('crop_size', 128))
        if self.crop_size % 4 != 0:
            raise ValueError(f"IHN expects crop_size to be divisible by 4, got {self.crop_size}.")
        self.lev0_size = self.crop_size // 4
        self.lev1_size = self.crop_size // 2

        # fnet - 特征提取器 (输入 3 通道，因为两个图像是分开提取特征的)
        # 我们需要在 forward 中拆分输入并单独传入
        self.fnet1 = BasicEncoderQuarter(output_dim=256, norm_fn='instance')
        
        # update_block_4 - 流场更新块，处理 1/4 分辨率的特征
        if self.lev0:
            self.update_block_4 = GMA(config, sz=self.lev0_size)
            
        # update_block_2 - 流场更新块，处理 1/2 分辨率的特征
        if self.lev1:
            self.update_block_2 = GMA(config, sz=self.lev1_size)


    def forward(self, input_pair, iters_lev0=None, iters_lev1=None, test_mode=False):
        """
        Args:
            input_pair (tensor): 拼接的 (I_warp, I_fix) 图像 (B, 6, H, W)
        Returns:
            tensor: 预测的四角点位移 d_hat (B x 8)
        """
        
        if iters_lev0 is None: iters_lev0 = self.iters_lev0
        if iters_lev1 is None: iters_lev1 = self.iters_lev1
        
        # 1. 拆分输入: I_warp (target) 和 I_fix (reference)
        # 假设输入是 (I_warp, I_fix) 拼接
        image1 = input_pair[:, :3] # I_warp
        image2 = input_pair[:, 3:] # I_fix

        # # Debug only
        # from utils.visualize import save_debug_image
        # save_debug_image(image1[0].permute(1, 2, 0).cpu().numpy(), filename='input_pair_image1')
        # save_debug_image(image2[0].permute(1, 2, 0).cpu().numpy(), filename='input_pair_image2')
        
        B, C, H, W = image1.shape
        device = image1.device
        if H != self.crop_size or W != self.crop_size:
            raise ValueError(
                f"IHN expected cropped inputs of shape ({self.crop_size}, {self.crop_size}), "
                f"but received ({H}, {W})."
            )

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        # 用于存储迭代预测结果
        preds_lev0 = [] 
        preds_lev1 = []

        # 2. 特征提取 (共享权重)
        with torch.amp.autocast('cuda', enabled=False):
            fmap1_quarter, fmap1_half = self.fnet1(image1)  # (B, 256, H/4, W/4), (B, 256, H/2, W/2)
            fmap2_quarter, fmap2_half = self.fnet1(image2)  # (B, 256, H/4, W/4), (B, 256, H/2, W/2)

        fmap1_half = fmap1_half.float()
        fmap2_half = fmap2_half.float()
        fmap1_quarter = fmap1_quarter.float()
        fmap2_quarter = fmap2_quarter.float()
        
        # ====================================================
        # Level 0: 1/4 特征图迭代
        # ====================================================
        if self.lev0:
            # 3. 相关性计算
            corr_fn_lev0 = CorrBlock(fmap1_quarter, fmap2_quarter, num_levels=2, radius=4, sz=self.lev0_size)
            
            # 4. 初始化流场 (coords0, coords1)
            sz_quarter = fmap1_quarter.shape
            coords0_q = coords_grid(B, sz_quarter[2], sz_quarter[3]).to(device) # B x 2 x H/4 x W/4
            coords1_q = coords0_q.clone()

            # 初始化四角点位移 (B x 2 x 2 x 2)
            four_point_disp_q = torch.zeros((B, 2, 2, 2)).to(device) 

            # 5. 迭代更新
            for _ in range(iters_lev0):
                corr = corr_fn_lev0(coords1_q) # B x C_corr x H/4 x W/4
                flow = coords1_q - coords0_q # 当前累计流场, B x 2 x H/4 x W/4
                
                # 更新块预测 Delta flow
                with torch.amp.autocast('cuda', enabled=False):
                    delta_four_point_q = self.update_block_4(corr, flow) # B x 2 x 2 x 2

                # 累计位移
                four_point_disp_q = four_point_disp_q + delta_four_point_q 
                
                # 更新流场坐标
                coords1_q = get_flow_now_4(four_point_disp_q, sz_quarter[2], sz_quarter[3], device) 
                
                preds_lev0.append(four_point_disp_q.clone()) # 记录预测
                
            # 将低分辨率的累计预测叠回高分辨率预测
            four_point_disp_h = four_point_disp_q # 传递到下一级


        # ====================================================
        # Level 1: 1/2 特征图迭代 (未实验)
        # ====================================================
        if self.lev1:

            flow_med = coords1_q - coords0_q
            flow_med = F.upsample_bilinear(flow_med, None, [4, 4]) * 4  # 四倍上采样
            image2_warp = warp(image2, flow_med)

            with torch.amp.autocast('cuda', enabled=False):
                _, fmap2_half = self.fnet1(image2_warp)
            fmap1_half = fmap1_half.float()
            fmap2_half = fmap2_half.float()

            # 6. 相关性计算
            corr_fn_lev1 = CorrBlock(fmap1_half, fmap2_half, num_levels=2, radius=4, sz=self.lev1_size)
            
            # 7. 初始化流场 (从低分辨率预测初始化)
            sz_half = fmap1_half.shape
            coords0_h = coords_grid(B, sz_half[2], sz_half[3]).to(device)   # B x 2 x H/2 x W/2
            coords1_h = coords0_h.clone()

            # 初始化四角点位移
            four_point_disp_h = torch.zeros((B, 2, 2, 2)).to(fmap1_half.device)

            # 8. 迭代更新
            for _ in range(iters_lev1):
                corr = corr_fn_lev1(coords1_h)
                flow = coords1_h - coords0_h

                with torch.amp.autocast('cuda', enabled=False):
                    delta_four_point_h = self.update_block_2(corr, flow)
                    
                four_point_disp_h = four_point_disp_h + delta_four_point_h
                coords1_h = get_flow_now_2(four_point_disp_h, sz_half[2], sz_half[3], device)
                
                preds_lev1.append(four_point_disp_h.clone()) # 记录预测
        
        # 9. 最终输出（测试模式返回预测值，训练模式返回预测列表）
        if test_mode:
            return four_point_disp_h
        else:
            return preds_lev0, preds_lev1
