"""
FrequencyDecoupler 单元测试脚本

测试内容：
1. 形状检查：验证输出特征图的空间维度保持不变
2. 重构检查：验证 f_low + f_high = input (无损分解)
3. 梯度检查：验证模块完全可微，梯度传播正常
"""

import sys
import torch
from core.frequency_decoupler import FrequencyDecoupler


def test_frequency_decoupler():
    print("=" * 60)
    print("FrequencyDecoupler 单元测试")
    print("=" * 60)
    
    decoupler = FrequencyDecoupler(kernel_size=5)
    print(f"\n[INFO] 模块实例化成功: {decoupler}")
    
    B, C, H, W = 2, 32, 128, 128
    dummy_input = torch.randn(B, C, H, W, requires_grad=True)
    print(f"[INFO] 生成模拟输入: shape={dummy_input.shape}, requires_grad={dummy_input.requires_grad}")
    
    print("\n[TEST 1] 执行前向传播...")
    f_low, f_high = decoupler(dummy_input)
    print(f"  - f_low shape: {f_low.shape}")
    print(f"  - f_high shape: {f_high.shape}")
    
    print("\n[TEST 2] 形状检查 (Shape Check)...")
    assert f_low.shape == (B, C, H, W), f"f_low shape错误: 期望 {(B, C, H, W)}, 实际 {f_low.shape}"
    assert f_high.shape == (B, C, H, W), f"f_high shape错误: 期望 {(B, C, H, W)}, 实际 {f_high.shape}"
    print("  ✓ 通过: f_low 和 f_high 形状均为 [2, 32, 128, 128]")
    
    print("\n[TEST 3] 重构检查 (Reconstruction Check)...")
    reconstructed = f_low + f_high
    is_close = torch.allclose(reconstructed, dummy_input, atol=1e-5)
    max_diff = torch.max(torch.abs(reconstructed - dummy_input)).item()
    print(f"  - 最大重构误差: {max_diff:.2e}")
    assert is_close, f"重构检查失败: 最大误差 {max_diff:.2e} > 1e-5"
    print("  ✓ 通过: f_low + f_high 完美还原原始输入 (atol=1e-5)")
    
    print("\n[TEST 4] 梯度检查 (Gradient Check)...")
    loss = f_low.mean() + f_high.mean()
    loss.backward()
    
    assert dummy_input.grad is not None, "梯度传播失败: dummy_input.grad 为 None"
    assert dummy_input.grad.shape == dummy_input.shape, f"梯度形状错误: {dummy_input.grad.shape}"
    
    grad_norm = torch.norm(dummy_input.grad).item()
    print(f"  - 梯度范数: {grad_norm:.6f}")
    print("  ✓ 通过: 梯度传播正常，模块完全可微")
    
    print("\n" + "=" * 60)
    print("所有测试通过！FrequencyDecoupler 模块验证成功")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        test_frequency_decoupler()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n[ERROR] 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] 运行时错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
