import sys

sys.path.append('core')

import os
import argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from core.igev_stereo import IGEVStereo, autocast
import core.stereo_datasets as datasets
from core.utils.utils import InputPadder
import torch.utils.data as data

import warnings
import rasterio

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def apply_jet_colormap(disp, mask=None, vmax=None):
    """将单通道视差图转换为 Jet 伪彩色图"""
    vmin = 0
    if vmax is None:
        if mask is not None and mask.sum() > 0:
            vmax = np.percentile(disp[mask], 98)
        else:
            vmax = disp.max()

    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6

    disp_norm = (disp - vmin) / (vmax - vmin)
    disp_norm = np.clip(disp_norm, 0, 1)

    cmap = plt.get_cmap('jet')
    colored = (cmap(disp_norm)[:, :, :3] * 255).astype(np.uint8)
    colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

    if mask is not None:
        colored_bgr[~mask] = 0

    return colored_bgr


def apply_error_colormap(err, mask=None, max_err=5.0):
    """
    专门用于误差图的伪彩色映射 (通常使用 hot 或 turbo 色带)
    低误差显示为黑/暗红，高误差显示为亮黄/白
    :param max_err: 截断误差阈值，默认超过 5 个像素的误差显示为最亮
    """
    err_norm = err / max_err
    err_norm = np.clip(err_norm, 0, 1)

    # 使用 'hot' 或 'inferno' 色带非常适合展示误差
    cmap = plt.get_cmap('hot')
    colored = (cmap(err_norm)[:, :, :3] * 255).astype(np.uint8)
    colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

    if mask is not None:
        colored_bgr[~mask] = 0  # 无效区域全黑

    return colored_bgr


def add_label(img, text):
    """在图片左上角添加带半透明背景的专业标签"""
    img_copy = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    # 获取文字大小
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # 绘制半透明黑色背景框
    overlay = img_copy.copy()
    padding = 10
    cv2.rectangle(overlay, (0, 0), (text_width + padding * 2, text_height + padding * 2 + baseline), (0, 0, 0), -1)
    alpha = 0.6  # 透明度
    cv2.addWeighted(overlay, alpha, img_copy, 1 - alpha, 0, img_copy)

    # 绘制白色文字
    cv2.putText(img_copy, text, (padding, text_height + padding), font, font_scale, (255, 255, 255), thickness,
                cv2.LINE_AA)

    return img_copy


@torch.no_grad()
def visualize_dfc_output(model, args):
    model.eval()

    root_path = args.data_path if args.data_path else '/home/roy/projects/YAOWEI/data/dfc2019-big'
    val_dataset = datasets.DFC2019(aug_params=None, root=root_path, split=args.split)

    max_count = args.max_files if args.max_files > 0 else len(val_dataset)
    print(f"Dataset Loaded. Total valid images: {len(val_dataset)}. Will process: {max_count}")

    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    for i, (file_info, image1, image2, flow_gt, valid_gt) in enumerate(tqdm(val_loader)):
        if i >= max_count:
            break

        left_img_path = file_info[0][0]
        base_name = os.path.basename(left_img_path).split('.')[0]

        image1_cuda = image1.cuda()
        image2_cuda = image2.cuda()

        padder = InputPadder(image1_cuda.shape, divis_by=32)
        image1_pad, image2_pad = padder.pad(image1_cuda, image2_cuda)

        with autocast(enabled=args.mixed_precision, dtype=getattr(torch, args.precision_dtype, torch.float16)):
            disp_pr = model(image1_pad, image2_pad, iters=args.valid_iters, test_mode=True)

        disp_pr = padder.unpad(disp_pr).cpu().numpy().squeeze()

        # 1. 准备左图 RGB
        img_np = image1[0].permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        img_show = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 2. 准备 GT 和 Mask
        disp_gt = flow_gt[0, 0].cpu().numpy()
        valid_mask = (valid_gt[0].cpu().numpy() > 0.5).squeeze()

        # 计算绝对误差
        err_map = np.abs(disp_pr - disp_gt)

        # 3. 渲染伪彩色
        if valid_mask.sum() > 0:
            vmax = np.percentile(disp_gt[valid_mask], 98)
        else:
            vmax = disp_pr.max()

        viz_gt = apply_jet_colormap(disp_gt, mask=valid_mask, vmax=vmax)
        viz_pr = apply_jet_colormap(disp_pr, mask=None, vmax=vmax)

        # 渲染误差图 (阈值设为 5px，超过5px的误差将显示为最亮的颜色)
        viz_err = apply_error_colormap(err_map, mask=valid_mask, max_err=5.0)

        # 4. 添加文字标签
        img_show = add_label(img_show, "Left Image")
        viz_gt = add_label(viz_gt, "Ground Truth")
        viz_pr = add_label(viz_pr, "Prediction")
        viz_err = add_label(viz_err, "Error Map (0-5px)")

        # 5. 拼接成四格图: [ 原图 | 真实视差 | 预测视差 | 误差图 ]
        concat = np.hstack([img_show, viz_gt, viz_pr, viz_err])

        # 保存图像
        save_path = output_directory / f"{base_name}_vis.png"
        cv2.imwrite(str(save_path), concat)

    print(f"Done! Results saved to {args.output_directory}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--data_path', help="dataset root path", default=None)
    parser.add_argument('--output_directory', default="result_vis_dfc")
    parser.add_argument('--max_files', type=int, default=50)
    parser.add_argument('--split', default='test', choices=['val', 'validation', 'test'])
    parser.add_argument('--dataset', default='dfc2019', choices=['dfc2019', 'whu'], help="which dataset to use")
    # 核心网络参数
    parser.add_argument('--mixed_precision', action='store_true', default=False)
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--valid_iters', type=int, default=32)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3)
    parser.add_argument('--corr_levels', type=int, default=2)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--n_downsample', type=int, default=2)
    parser.add_argument('--n_gru_layers', type=int, default=3)
    parser.add_argument('--max_disp', type=int, default=768)
    parser.add_argument('--s_disp_range', type=int, default=48)
    parser.add_argument('--m_disp_range', type=int, default=96)
    parser.add_argument('--l_disp_range', type=int, default=192)
    parser.add_argument('--s_disp_interval', type=int, default=1)
    parser.add_argument('--m_disp_interval', type=int, default=2)
    parser.add_argument('--l_disp_interval', type=int, default=4)
    parser.add_argument('--model_arch', default='ours', choices=['baseline', 'ours'], help='Choose model architecture')

    args = parser.parse_args()

    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])

    print(f"Loading checkpoint: {args.restore_ckpt}")
    checkpoint = torch.load(args.restore_ckpt)

    # ==================== [新增：兼容旧版权重命名] ====================
    # 将旧的 'gbc_volume' 键名动态替换为新的 'guided_volume'
    new_checkpoint = {}
    for k, v in checkpoint.items():
        if 'gbc_volume' in k:
            new_key = k.replace('gbc_volume', 'guided_volume')
            new_checkpoint[new_key] = v
        else:
            new_checkpoint[k] = v
    # =================================================================

    # 使用转换后的 new_checkpoint 
    model.load_state_dict(new_checkpoint, strict=True)
    model.cuda()

    visualize_dfc_output(model, args)