"""
FrequencyDecoupler: 零参数、即插即用的频域解耦模块

该模块将特征图分解为低频和高频分量，用于IGEV++立体匹配网络。
- 低频分量：承载平滑区域的几何共识（屋顶、道路等）
- 高频分量：承载细粒度几何特征（建筑边缘、小目标物体）

作者: IGEV++ Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyDecoupler(nn.Module):
    """
    频域解耦模块：将特征图分解为低频和高频分量
    
    特点：
    - 零学习参数，完全基于数学运算
    - 即插即用，无需训练
    - 无损分解：f_low + f_high = input (在浮点误差范围内)
    
    Args:
        kernel_size (int): 平均池化核大小，默认为5
    """
    
    def __init__(self, kernel_size: int = 5):
        super(FrequencyDecoupler, self).__init__()
        
        assert kernel_size % 2 == 1, f"kernel_size must be odd, got {kernel_size}"
        assert kernel_size >= 3, f"kernel_size must be >= 3, got {kernel_size}"
        
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
    def forward(self, features: torch.Tensor) -> tuple:
        """
        前向传播：将特征图分解为低频和高频分量
        
        Args:
            features: 输入特征图，形状为 [B, C, H, W]
            
        Returns:
            tuple: (f_low, f_high)
                - f_low: 低频特征图，形状 [B, C, H, W]
                - f_high: 高频特征图，形状 [B, C, H, W]
        """
        f_low = F.avg_pool2d(
            features,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            count_include_pad=False
        )
        
        f_high = features - f_low
        
        return f_low, f_high
    
    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, padding={self.padding}"
