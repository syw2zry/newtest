"""
GBC_Volume (Granular Ball Computing Volume) 模块
=====================================
该模块作为 IGEV++ 网络中 APM (Adaptive Patch Matching) 的直接平替模块。
全流程基于 PyTorch 原生算子，无需自定义 CUDA 编译，保证跨平台稳定性。

核心思想：
1. 使用 Group-wise Correlation 构建基础代价卷
2. 通过语义先验预测多尺度聚合权重
3. 并行多尺度 3D 卷积捕获不同感受野信息
4. 动态软路由融合生成最终代价卷
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GBC_Volume(nn.Module):
    """
    Granular Ball Computing Volume 模块
    
    该模块通过多尺度粒球计算策略构建代价卷，用于立体匹配网络。
    
    Args:
        in_channels (int): 左右图特征的输入通道数
        out_channels (int): 最终输出的 3D 代价卷通道数
        max_disp (int): 最大视差范围
        num_groups (int, optional): 分组相关的组数。默认为 8
    """
    
    def __init__(self, in_channels: int, out_channels: int, max_disp: int, num_groups: int = 8):
        super(GBC_Volume, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_disp = max_disp
        self.num_groups = num_groups
        
        # 验证通道数是否能被组数整除
        assert in_channels % num_groups == 0, \
            f"in_channels ({in_channels}) 必须能被 num_groups ({num_groups}) 整除"
            
        # 自适应温度系数 (限制最小值为0.1防止除以0)
        self.adaptive_temp = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

        # Step 2: 粒球尺度预测器 (Scale Predictor)
        # 注意：输入通道是 in_channels + 1，因为我们要拼接 edge_mask
        self.scale_predictor = nn.Sequential(
            nn.Conv2d(in_channels + 1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1, bias=True)
        )
        
        # Step 3: 并行多尺度 3D 聚合层
        # 三个不同膨胀率的 3D 卷积，捕获不同感受野的信息
        # 小感受野: dilation=1, 标准卷积
        self.agg_small = nn.Conv3d(
            num_groups, num_groups, 
            kernel_size=3, stride=1, padding=1, dilation=1,
            bias=False
        )
        # 中感受野: dilation=2, 扩大感受野
        self.agg_medium = nn.Conv3d(
            num_groups, num_groups,
            kernel_size=3, stride=1, padding=2, dilation=2,
            bias=False
        )
        # 大感受野: dilation=4, 进一步扩大感受野
        self.agg_large = nn.Conv3d(
            num_groups, num_groups,
            kernel_size=3, stride=1, padding=4, dilation=4,
            bias=False
        )
        
        # Step 4: 最终通道映射层
        # 将 num_groups 通道映射为 out_channels 通道
        self.final_conv = nn.Conv3d(num_groups, out_channels, kernel_size=1, bias=True)
    
    
    def _build_volume(self, feat_l: torch.Tensor, feat_r: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat_l.shape
        group_channels = C // self.num_groups
        feat_l_grouped = feat_l.view(B, self.num_groups, group_channels, H, W)
        feat_r_grouped = feat_r.view(B, self.num_groups, group_channels, H, W)
        
        feat_l_norm = F.normalize(feat_l_grouped, p=2, dim=2)
        feat_r_norm = F.normalize(feat_r_grouped, p=2, dim=2)
        
        cost_list = []
        
        # 【性能优化核心】一次性在左侧 Pad 最大视差减1的宽度
        pad_size = self.max_disp - 1
        feat_r_padded = F.pad(feat_r_norm, (pad_size, 0, 0, 0), mode='constant', value=0)
        
        for d in range(self.max_disp):
            # 利用底层指针偏移直接切片，避免循环内重复开辟显存
            start_idx = pad_size - d
            feat_r_shifted = feat_r_padded[:, :, :, :, start_idx : start_idx + W]
            
            # 计算组内点积，并且除以通道数的平方根 (Scaled Dot-Product)
            # 类似 Transformer 的做法，防止通道过多导致方差过大，利于后续 3D 卷积的平稳收敛
            correlation = (feat_l_norm * feat_r_shifted).sum(dim=2) / (group_channels ** 0.5)
            cost_list.append(correlation)
            
        base_volume = torch.stack(cost_list, dim=2)
        return base_volume


    # 【修改点 1】: 将 edge_mask 作为形参显式加进去
    def forward(self, feat_l: torch.Tensor, feat_r: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            feat_l: 左图特征 [B, in_channels, H, W]
            feat_r: 右图特征 [B, in_channels, H, W]
            edge_mask: 左图边缘先验 [B, 1, H, W]
            
        Returns:
            cost_volume: 最终代价卷 [B, out_channels, max_disp, H, W]
        """
        # Step 1: 构建基础代价卷
        base_volume = self._build_volume(feat_l, feat_r)

        # Step 2: 粒球尺度预测 (结合语义先验)
        # 将左图特征与边缘 Mask 拼接，让预测器“长眼”
        predictor_input = torch.cat([feat_l, edge_mask], dim=1)
        scale_weights = self.scale_predictor(predictor_input)
        
        # 3. 使用自适应温度系数 (限制最小值为0.1防止除以0)
        temp = torch.clamp(self.adaptive_temp, min=0.1)
        scale_weights = F.softmax(scale_weights / temp, dim=1)
        
        # 分离三个尺度的权重
        # w_s: 小感受野权重, w_m: 中感受野权重, w_l: 大感受野权重
        w_s = scale_weights[:, 0:1, :, :]  # [B, 1, H, W]
        w_m = scale_weights[:, 1:2, :, :]  # [B, 1, H, W]
        w_l = scale_weights[:, 2:3, :, :]  # [B, 1, H, W]
        
        # ==================== [终极显存榨干版] Step 3 & 4 ====================
        w_s_exp = w_s.unsqueeze(2)  
        w_m_exp = w_m.unsqueeze(2)  
        w_l_exp = w_l.unsqueeze(2)  
        
        # 1. 小粒球分支：计算 -> 原地乘法 -> 赋值给 fused_volume
        vol_s = self.agg_small(base_volume)
        fused_volume = vol_s.mul_(w_s_exp)  # mul_ 是原地操作，不产生中间变量，直接复用 vol_s 的显存
        # 注意：这里不能 del vol_s，因为 fused_volume 已经指向了同一块内存
        
        # 2. 中粒球分支：计算 -> 原地乘法 -> 原地累加 -> 销毁
        vol_m = self.agg_medium(base_volume)
        vol_m.mul_(w_m_exp)                 # 极致优化：先原地乘权重，吃掉 w_m_exp
        fused_volume.add_(vol_m)            # 原地加到主卷上
        del vol_m                           # 立即释放
        
        # 3. 大粒球分支：计算 -> 原地乘法 -> 原地累加 -> 销毁
        vol_l = self.agg_large(base_volume)
        vol_l.mul_(w_l_exp)                 # 极致优化：原地乘法
        fused_volume.add_(vol_l)
        del vol_l, base_volume              # 释放最后的分支和基础卷
        
        # 最终通道映射
        cost_volume = self.final_conv(fused_volume)
        # =====================================================================
        
        return cost_volume


if __name__ == '__main__':
    """
    测试模块：验证 GBC_Volume 的功能正确性
    
    测试场景：
    - B=2, in_channels=128, max_disp=48, H=64, W=64
    - out_channels=32, num_groups=8
    """
    print("=" * 60)
    print("GBC_Volume 模块测试")
    print("=" * 60)
    
    # 设置测试参数
    B = 2
    in_channels = 128
    max_disp = 48
    H = 64
    W = 64
    out_channels = 32
    num_groups = 8
    
    # 创建模型实例
    print(f"\n[1] 创建模型实例...")
    print(f"    参数配置:")
    print(f"    - in_channels: {in_channels}")
    print(f"    - out_channels: {out_channels}")
    print(f"    - max_disp: {max_disp}")
    print(f"    - num_groups: {num_groups}")
    
    model = GBC_Volume(
        in_channels=in_channels,
        out_channels=out_channels,
        max_disp=max_disp,
        num_groups=num_groups
    )
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[2] 模型参数统计:")
    print(f"    - 总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"    - 可训练参数量: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    
    # 创建模拟输入
    print(f"\n[3] 创建模拟输入...")
    feat_l = torch.randn(B, in_channels, H, W)
    feat_r = torch.randn(B, in_channels, H, W)
    # 【修改点 2】: 同步生成 edge_mask
    edge_mask = torch.rand(B, 1, H, W)  
    print(f"    - feat_l 形状: {list(feat_l.shape)}")
    print(f"    - feat_r 形状: {list(feat_r.shape)}")
    print(f"    - edge_mask 形状: {list(edge_mask.shape)}")
    
    # 前向传播
    print(f"\n[4] 执行前向传播...")
    model.eval()
    with torch.no_grad():
        # 【修改点 3】: 传入 edge_mask
        cost_volume = model(feat_l, feat_r, edge_mask) 
    
    # 验证输出形状
    print(f"\n[5] 验证输出形状...")
    expected_shape = [B, out_channels, max_disp, H, W]
    actual_shape = list(cost_volume.shape)
    
    print(f"    - 期望形状: {expected_shape}")
    print(f"    - 实际形状: {actual_shape}")
    
    assert actual_shape == expected_shape, \
        f"输出形状不匹配! 期望 {expected_shape}, 实际 {actual_shape}"
    
    print(f"\n[6] [OK] 形状验证通过!")
    
    # 检查输出值范围（相关性值应在合理范围内）
    print(f"\n[7] 输出值统计:")
    print(f"    - 最小值: {cost_volume.min().item():.6f}")
    print(f"    - 最大值: {cost_volume.max().item():.6f}")
    print(f"    - 均值: {cost_volume.mean().item():.6f}")
    print(f"    - 标准差: {cost_volume.std().item():.6f}")
    
    # 测试梯度反向传播
    print(f"\n[8] 测试梯度反向传播...")
    model.train()
    feat_l_grad = torch.randn(B, in_channels, H, W, requires_grad=True)
    feat_r_grad = torch.randn(B, in_channels, H, W, requires_grad=True)
    # 【修改点 4】: 测试反向传播时传入 edge_mask (通常先验不需要梯度)
    edge_mask_grad = torch.rand(B, 1, H, W, requires_grad=False)
    
    output = model(feat_l_grad, feat_r_grad, edge_mask_grad)
    loss = output.mean()
    loss.backward()
    
    assert feat_l_grad.grad is not None, "feat_l 梯度计算失败"
    assert feat_r_grad.grad is not None, "feat_r 梯度计算失败"
    print(f"    - feat_l 梯度形状: {list(feat_l_grad.grad.shape)}")
    print(f"    - feat_r 梯度形状: {list(feat_r_grad.grad.shape)}")
    print(f"    [OK] 梯度反向传播正常!")
    
    print("\n" + "=" * 60)
    print("[OK] 所有测试通过! GBC_Volume 模块运行正常")
    print("=" * 60)