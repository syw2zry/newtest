"""
EdgeGuidance Module for IGEV++ Stereo Matching Network
(边缘结构先验引导模块)

该模块通过纯物理/数学算子提取左视 RGB 图像中的结构性高频边缘（如建筑边界），
同时利用高斯低通物理特性抑制无序的高频噪声（如树冠纹理）。
它为后续的动态代价聚合（GBC）提供了零参数、零显存负担的绝对几何先验。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeGuidance(nn.Module):
    """
    边缘先验引导模块 (Edge Guidance Module)

    处理流程:
    1. RGB 转灰度图
    2. 高斯平滑 (物理抑制树冠等细碎高频噪声)
    3. Sobel 梯度提取 (严格捕捉空间几何断崖)
    4. 平均池化下采样 (保持感受野内的几何连贯性)
    5. Sigmoid 非线性映射 (取代极易引发崩塌的 Min-Max 归一化)
    """

    def __init__(self, downsample_factor=4, blur_kernel_size=5, sigma=1.5):
        super(EdgeGuidance, self).__init__()

        self.downsample_factor = downsample_factor
        self.blur_kernel_size = blur_kernel_size
        self.sigma = sigma

        # 1. RGB 转灰度权重 (固定物理常量)
        rgb_to_gray_weights = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        self.register_buffer('rgb_to_gray_weights', rgb_to_gray_weights)

        # 2. 高斯平滑核 (固定数学算子)
        gaussian_kernel = self._create_gaussian_kernel(blur_kernel_size, sigma)
        self.register_buffer('gaussian_kernel', gaussian_kernel)

        # 3. Sobel 边缘检测算子 (固定数学算子)
        sobel_x = torch.tensor([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0]
        ]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)

        sobel_y = torch.tensor([
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0]
        ]).view(1, 1, 3, 3)
        self.register_buffer('sobel_y', sobel_y)

    def _create_gaussian_kernel(self, kernel_size, sigma):
        """生成 2D 高斯平滑核"""
        x = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        gauss_2d = gauss_1d.view(-1, 1) @ gauss_1d.view(1, -1)
        gauss_2d = gauss_2d / gauss_2d.sum()
        return gauss_2d.view(1, 1, kernel_size, kernel_size)

    def forward(self, left_rgb):
        """
        前向传播
        Args:
            left_rgb: [B, 3, H, W] 的归一化图像 (-1~1 或 0~1)
        Returns:
            edge_mask: [B, 1, H/4, W/4] 的边缘先验掩膜，值域 (0, 1)
        """
        B, C, H, W = left_rgb.shape
        
        # 步骤 1: 灰度化
        gray = F.conv2d(left_rgb, self.rgb_to_gray_weights, bias=None)

        # 步骤 2: 高斯去噪 (极重要：抹平树冠内部的乱纹理)
        pad_size = self.blur_kernel_size // 2
        gray_padded = F.pad(gray, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        gray_smooth = F.conv2d(gray_padded, self.gaussian_kernel, bias=None)

        # 步骤 3: Sobel 提取结构边缘
        grad_x = F.conv2d(gray_smooth, self.sobel_x, bias=None, padding=1)
        grad_y = F.conv2d(gray_smooth, self.sobel_y, bias=None, padding=1)
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

        # 步骤 4: 平均池化下采样 (对齐主干网络分辨率，同时揉平局部毛刺)
        downsampled = F.avg_pool2d(
            gradient_magnitude,
            kernel_size=self.downsample_factor,
            stride=self.downsample_factor
        )

        # 步骤 5: 绝对阈值非线性映射 (【极其关键的修复】取代 Min-Max 归一化)
        # 通过 Sigmoid 函数，将强梯度推向 1 (锐利边缘)，将弱梯度压向 0 (平滑区与残余噪声)
        # scale=5.0 和 offset=0.2 是一组经验参数，使小噪声被抑制，大边缘被保留
        scale = 5.0
        offset = 0.2
        edge_mask = torch.sigmoid(scale * (downsampled - offset))

        # 进一步增加对比度，让边缘更清晰
        edge_mask = edge_mask ** 2.0

        return edge_mask


if __name__ == "__main__":
    # 模块健康自检
    print("=" * 60)
    print("EdgeGuidance 模块自检")
    module = EdgeGuidance()
    dummy_input = torch.randn(2, 3, 512, 512)
    output = module(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape} (期望: [2, 1, 128, 128])")
    print(f"输出值域: [{output.min().item():.4f}, {output.max().item():.4f}] (期望: >0.0 且 <1.0)")
    print("=" * 60)