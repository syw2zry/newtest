"""
Adaptive Modules for High-Resolution Stereo Matching
====================================================
此文件包含三大核心创新模块：
1. EdgeGuidance: 提取物理边缘的结构性先验
2. FrequencyDecoupler: 特征高低频解耦
3. AdaptiveScaleVolume: (原GBC) 自适应多尺度代价卷聚合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 边缘结构先验引导模块
# ==========================================
class EdgeGuidance(nn.Module):
    def __init__(self, downsample_factor=4, blur_kernel_size=5, sigma=1.5):
        super(EdgeGuidance, self).__init__()
        self.downsample_factor = downsample_factor
        self.blur_kernel_size = blur_kernel_size
        self.sigma = sigma

        rgb_to_gray_weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        self.register_buffer('rgb_to_gray_weights', rgb_to_gray_weights)

        gaussian_kernel = self._create_gaussian_kernel(blur_kernel_size, sigma)
        self.register_buffer('gaussian_kernel', gaussian_kernel)

        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)

        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_y', sobel_y)

    def _create_gaussian_kernel(self, kernel_size, sigma):
        x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        gauss_2d = gauss_1d.view(-1, 1) @ gauss_1d.view(1, -1)
        gauss_2d = gauss_2d / gauss_2d.sum()
        return gauss_2d.view(1, 1, kernel_size, kernel_size)

    def forward(self, left_rgb):
        gray = F.conv2d(left_rgb, self.rgb_to_gray_weights, bias=None)
        
        pad_size = self.blur_kernel_size // 2
        gray_padded = F.pad(gray, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        gray_smooth = F.conv2d(gray_padded, self.gaussian_kernel, bias=None)

        grad_x = F.conv2d(gray_smooth, self.sobel_x, bias=None, padding=1)
        grad_y = F.conv2d(gray_smooth, self.sobel_y, bias=None, padding=1)
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        downsampled = F.avg_pool2d(
            gradient_magnitude,
            kernel_size=self.downsample_factor,
            stride=self.downsample_factor
        )

        # 非线性软阈值映射
        scale, offset = 5.0, 0.17
        edge_mask = torch.sigmoid(scale * (downsampled - offset))
        edge_mask = edge_mask ** 2.0
        
        return edge_mask


# ==========================================
# 2. 频域解耦模块
# ==========================================
class FrequencyDecoupler(nn.Module):
    def __init__(self, kernel_size: int = 3):
        super(FrequencyDecoupler, self).__init__()
        assert kernel_size % 2 == 1, f"kernel_size must be odd, got {kernel_size}"
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
    def forward(self, features: torch.Tensor) -> tuple:
        f_low = F.avg_pool2d(
            features,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            count_include_pad=False
        )
        f_high = features - f_low
        return f_low, f_high


# ==========================================
# 3. 自适应尺度代价卷 (原 GBC)
# ==========================================
class AdaptiveScaleVolume(nn.Module):
    """
    Adaptive Scale Cost Volume (取代原来的 GBC_Volume)
    利用语义边缘 Mask，动态预测感受野尺度权重，自适应融合多尺度 3D 代价卷。
    """
    def __init__(self, in_channels: int, out_channels: int, max_disp: int, num_groups: int = 8):
        super(AdaptiveScaleVolume, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_disp = max_disp
        self.num_groups = num_groups
        
        assert in_channels % num_groups == 0, "in_channels 必须能被 num_groups 整除"
            
        self.adaptive_temp = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

        # 尺度预测器：输入通道 + 1 (接收 edge_mask)
        self.scale_predictor = nn.Sequential(
            nn.Conv2d(in_channels + 1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1, bias=True)
        )
        
        # 多尺度 3D 聚合 (小/中/大感受野)
        self.agg_small = nn.Conv3d(num_groups, num_groups, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.agg_medium = nn.Conv3d(num_groups, num_groups, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.agg_large = nn.Conv3d(num_groups, num_groups, kernel_size=3, stride=1, padding=4, dilation=4, bias=False)
        
        self.final_conv = nn.Conv3d(num_groups, out_channels, kernel_size=1, bias=True)
    
    def _build_volume(self, feat_l: torch.Tensor, feat_r: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat_l.shape
        group_channels = C // self.num_groups
        feat_l_grouped = feat_l.view(B, self.num_groups, group_channels, H, W)
        feat_r_grouped = feat_r.view(B, self.num_groups, group_channels, H, W)
        
        feat_l_norm = F.normalize(feat_l_grouped, p=2, dim=2)
        feat_r_norm = F.normalize(feat_r_grouped, p=2, dim=2)
        
        cost_list = []
        pad_size = self.max_disp - 1
        feat_r_padded = F.pad(feat_r_norm, (pad_size, 0, 0, 0), mode='constant', value=0)
        
        for d in range(self.max_disp):
            start_idx = pad_size - d
            feat_r_shifted = feat_r_padded[:, :, :, :, start_idx : start_idx + W]
            correlation = (feat_l_norm * feat_r_shifted).sum(dim=2) / (group_channels ** 0.5)
            cost_list.append(correlation)
            
        base_volume = torch.stack(cost_list, dim=2)
        return base_volume

    def forward(self, feat_l: torch.Tensor, feat_r: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        base_volume = self._build_volume(feat_l, feat_r)

        # 结合语义先验进行尺度预测
        predictor_input = torch.cat([feat_l, edge_mask], dim=1)
        scale_weights = self.scale_predictor(predictor_input)
        
        temp = torch.clamp(self.adaptive_temp, min=0.1)
        scale_weights = F.softmax(scale_weights / temp, dim=1)
        
        w_s_exp = scale_weights[:, 0:1, :, :].unsqueeze(2)  
        w_m_exp = scale_weights[:, 1:2, :, :].unsqueeze(2)  
        w_l_exp = scale_weights[:, 2:3, :, :].unsqueeze(2)  
        
        # 极致显存优化的原地操作
        vol_s = self.agg_small(base_volume)
        fused_volume = vol_s.mul_(w_s_exp)
        
        vol_m = self.agg_medium(base_volume)
        vol_m.mul_(w_m_exp)
        fused_volume.add_(vol_m)
        del vol_m
        
        vol_l = self.agg_large(base_volume)
        vol_l.mul_(w_l_exp)
        fused_volume.add_(vol_l)
        del vol_l, base_volume
        
        cost_volume = self.final_conv(fused_volume)
        return cost_volume