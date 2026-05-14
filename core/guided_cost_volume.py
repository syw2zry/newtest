"""
V2.0 Frequency-Orthogonal Split & Soft-Binding Cross-Frequency Cost Volume
============================================================================
核心架构重构：基于频域正交切分与软绑定的跨频协同代价体

核心组件：
1. LearnableEdgeGuidance: 可学习边缘引导模块（替代Sobel，支持端到端学习，解决低频抹平问题）
2. FrequencyDecoupler: 频域解耦模块
3. CrossFreqInteraction: 跨频交互模块（1x1x1 3D卷积）
4. AdaptiveScaleVolume: 自适应尺度代价体（通道切分 + 软路由，已修复 Autograd 兼容性）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LearnableEdgeGuidance(nn.Module):
    """
    可学习边缘引导模块
    
    使用步长卷积（Stride=2）替代暴力的平均池化，
    在提取高频几何特征的同时完成 1/4 下采样，输出单通道边缘概率掩码。
    
    Args:
        in_channels (int): 输入图像通道数，默认3 (RGB)
        hidden_channels (int): 隐藏层通道数，默认16
    """
    def __init__(self, in_channels: int = 3, hidden_channels: int = 16):
        super(LearnableEdgeGuidance, self).__init__()
        
        # 使用两次 stride=2 的卷积实现 1/4 降采样，保护高频特征不被抹平
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.InstanceNorm2d(hidden_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(hidden_channels, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(hidden_channels, affine=True),
            nn.ReLU(inplace=True),
        )
        
        self.edge_head = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.edge_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.InstanceNorm2d):
                if m.affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        
        nn.init.zeros_(self.edge_head.weight)
        # 初始偏置设为 -2.0，使 Sigmoid 初始输出在 0.11 左右，防止训练初期特征全灭
        nn.init.constant_(self.edge_head.bias, -2.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        edge_feat = self.edge_conv(x)
        edge_logits = self.edge_head(edge_feat)
        edge_mask = torch.sigmoid(edge_logits)
        
        return edge_mask


class FrequencyDecoupler(nn.Module):
    """
    频域解耦模块：将特征图分解为低频和高频分量
    """
    def __init__(self, kernel_size: int = 5):
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


class CrossFreqInteraction(nn.Module):
    """
    跨频交互模块：使用 1x1x1 3D 卷积实现高低频通道交互
    """
    def __init__(self, num_groups: int = 8):
        super(CrossFreqInteraction, self).__init__()
        self.num_groups = num_groups
        
        self.interaction = nn.Sequential(
            nn.Conv3d(num_groups, num_groups, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm3d(num_groups, affine=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.interaction.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.InstanceNorm3d):
                if m.affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.interaction(x)


class AdaptiveScaleVolume(nn.Module):
    """
    V2.0 自适应尺度代价体模块
    """
    def __init__(self, in_channels: int, out_channels: int, max_disp: int, num_groups: int = 8):
        super(AdaptiveScaleVolume, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_disp = max_disp
        self.num_groups = num_groups
        self.half_groups = num_groups // 2
        
        assert in_channels % num_groups == 0, "in_channels 必须能被 num_groups 整除"
        assert num_groups % 2 == 0, "num_groups 必须为偶数以支持通道切分"
        
        # 将温度参数注册为 Buffer，避免被优化器错误更新导致调度失效
        self.register_buffer('adaptive_temp', torch.tensor([2.0], dtype=torch.float32))
        
        self.scale_predictor = nn.Sequential(
            nn.Conv2d(in_channels + 1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1, bias=True)
        )
        
        self.cross_freq = CrossFreqInteraction(num_groups=num_groups)
        
        self.agg_small = nn.Conv3d(num_groups, num_groups, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.agg_medium = nn.Conv3d(num_groups, num_groups, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.agg_large = nn.Conv3d(num_groups, num_groups, kernel_size=3, stride=1, padding=4, dilation=4, bias=False)
        
        self.final_conv = nn.Conv3d(num_groups, out_channels, kernel_size=1, bias=True)
        
        self._init_agg_weights()
    
    def _init_agg_weights(self):
        for m in [self.agg_small, self.agg_medium, self.agg_large]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)
        
        nn.init.zeros_(self.scale_predictor[-1].weight)
        nn.init.zeros_(self.scale_predictor[-1].bias)
    
    def update_temperature(self, current_epoch: int, total_epochs: int, 
                           start_temp: float = 2.0, end_temp: float = 0.1) -> None:
        """余弦退火更新温度参数"""
        progress = min(current_epoch / total_epochs, 1.0)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        new_temp = end_temp + (start_temp - end_temp) * cosine_factor
        self.adaptive_temp.fill_(new_temp)
    
    def _build_volume(self, feat_l: torch.Tensor, feat_r: torch.Tensor, groups: int) -> torch.Tensor:
        B, C, H, W = feat_l.shape
        group_channels = C // groups
        
        feat_l_grouped = feat_l.view(B, groups, group_channels, H, W)
        feat_r_grouped = feat_r.view(B, groups, group_channels, H, W)
        
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
        
        cost_volume = torch.stack(cost_list, dim=2)
        return cost_volume
    
    def forward(self, match_low_l: torch.Tensor, match_low_r: torch.Tensor,
                match_high_l: torch.Tensor, match_high_r: torch.Tensor,
                feat_l_full: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        
        cost_low = self._build_volume(match_low_l, match_low_r, self.half_groups)
        cost_high = self._build_volume(match_high_l, match_high_r, self.half_groups)
        
        cost_base = torch.cat([cost_low, cost_high], dim=1)
        del cost_low, cost_high
        
        cost_fused = self.cross_freq(cost_base)
        
        predictor_input = torch.cat([feat_l_full, edge_mask], dim=1)
        scale_logits = self.scale_predictor(predictor_input)
        
        temp = torch.clamp(self.adaptive_temp, min=0.1)
        scale_weights = F.softmax(scale_logits / temp, dim=1)
        
        w_s = scale_weights[:, 0:1, :, :].unsqueeze(2)
        w_m = scale_weights[:, 1:2, :, :].unsqueeze(2)
        w_l = scale_weights[:, 2:3, :, :].unsqueeze(2)
        
        # 严格使用纯函数运算，杜绝 in-place (.mul_ / .add_)，防止 PyTorch 计算图反向传播报错
        vol_s = self.agg_small(cost_fused) * w_s
        vol_m = self.agg_medium(cost_fused) * w_m
        vol_l = self.agg_large(cost_fused) * w_l
        
        fused_volume = vol_s + vol_m + vol_l
        del vol_s, vol_m, vol_l, cost_fused
        
        cost_volume = self.final_conv(fused_volume)
        return cost_volume

# EdgeGuidance (Sobel) 类已移除。由于已完全重构为 LearnableEdgeGuidance，保留旧版代码只会增加冗余和误用的风险。