import torch
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from PIL import Image
import os
import sys

# 导入你刚刚写好的模块
from core.edge_guidance import EdgeGuidance

def test_on_real_dataset(image_path):
    print("=" * 60)
    print("EdgeGuidance - 真实卫星影像可视化测试")
    print("=" * 60)
    
    # 1. 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 找不到图片 {image_path}")
        print("请把 image_path 替换为你数据集里的真实图片路径！")
        return

    # 2. 实例化模块 (你可以调节这里的参数看效果)
    # sigma 越大，对树冠的抑制越强，但也可能导致较弱的建筑边缘丢失
    module = EdgeGuidance(downsample_factor=4, blur_kernel_size=5, sigma=1.5)
    
    # 3. 读取真实图片并转换为 PyTorch 张量
    print(f"读取图片: {image_path}")
    img = Image.open(image_path).convert('RGB')
    
    # TF.to_tensor 会自动把 PIL 图像转换为 [C, H, W] 形状，并将像素值归一化到 [0.0, 1.0]
    input_tensor = TF.to_tensor(img)
    
    # 增加 Batch 维度，使其变成 [1, 3, H, W]
    input_tensor = input_tensor.unsqueeze(0) 
    print(f"输入 Tensor 形状: {input_tensor.shape}")
    
    # 4. 前向传播提取语义边缘 Mask
    with torch.no_grad():
        mask_output = module(input_tensor)
        
    print(f"输出 Mask 形状: {mask_output.shape}")
    
    # 5. 保存结果用于肉眼比对
    output_dir = "real_data_outputs/3"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存原始输入图片 (方便对比)
    vutils.save_image(input_tensor[0], os.path.join(output_dir, "01_original_image.png"))
    
    # 保存提取出的 Edge Mask
    vutils.save_image(mask_output[0], os.path.join(output_dir, "02_edge_mask.png"))
    
    print(f"\n可视化完成！请打开 {output_dir} 文件夹查看效果。")
    print("观察重点：")
    print("1. 建筑的边缘是否呈现高亮的白色？")
    print("2. 树冠/植被区域是否呈现暗淡的灰黑色（被抑制）？")

if __name__ == "__main__":
    # 【请在这里修改为你的真实图片路径】
    # 比如: "demo-imgs/PipesH/im0.png" 或者是你的 DFC2019 测试集路径
    REAL_IMAGE_PATH = "/home/roy/projects/YAOWEI/data/dfc2019-big/left/JAX_017_004_011_LEFT_RGB.tif"
    
    test_on_real_dataset(REAL_IMAGE_PATH)