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
    err_norm = err / max_err
    err_norm = np.clip(err_norm, 0, 1)
    cmap = plt.get_cmap('hot')
    colored = (cmap(err_norm)[:, :, :3] * 255).astype(np.uint8)
    colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

    if mask is not None:
        colored_bgr[~mask] = 0

    return colored_bgr


def add_label(img, text):
    img_copy = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    overlay = img_copy.copy()
    padding = 10
    cv2.rectangle(overlay, (0, 0), (text_width + padding * 2, text_height + padding * 2 + baseline), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, img_copy, 1 - alpha, 0, img_copy)
    cv2.putText(img_copy, text, (padding, text_height + padding), font, font_scale, (255, 255, 255), thickness,
                cv2.LINE_AA)

    return img_copy


# ==================== [核心功能：智能加载器] ====================
def load_model(args, ckpt_path, arch):
    """动态实例化模型并加载权重，自动处理字典兼容性问题"""
    # 临时修改 args，让 IGEVStereo.__init__ 正确初始化对应架构的模块
    args.model_arch = arch
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])

    print(f"Loading {arch} checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path)

    # 兼容旧版权重的 'gbc_volume' -> 'guided_volume'
    new_checkpoint = {}
    for k, v in checkpoint.items():
        if 'gbc_volume' in k:
            new_key = k.replace('gbc_volume', 'guided_volume')
            new_checkpoint[new_key] = v
        else:
            new_checkpoint[k] = v

    model.load_state_dict(new_checkpoint, strict=True)
    model.cuda()
    model.eval()
    return model


# ================================================================

@torch.no_grad()
def visualize_compare_output(model1, model2, args):
    root_path = args.data_path if args.data_path else '/home/roy/projects/YAOWEI/data/dfc2019-big'
    val_dataset = datasets.DFC2019(aug_params=None, root=root_path, split=args.split)

    max_count = args.max_files if args.max_files > 0 else len(val_dataset)
    print(f"Dataset Loaded. Will process: {max_count} images for comparison.")

    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    for i, (file_info, image1, image2, flow_gt, valid_gt) in enumerate(tqdm(val_loader)):
        if i >= max_count: break

        left_img_path = file_info[0][0]
        base_name = os.path.basename(left_img_path).split('.')[0]

        image1_cuda = image1.cuda()
        image2_cuda = image2.cuda()

        padder = InputPadder(image1_cuda.shape, divis_by=32)
        image1_pad, image2_pad = padder.pad(image1_cuda, image2_cuda)

        # ==== 前向传播：模型 1 ====
        args.model_arch = args.arch1
        with autocast(enabled=args.mixed_precision, dtype=getattr(torch, args.precision_dtype, torch.float16)):
            disp_pr1 = model1(image1_pad, image2_pad, iters=args.valid_iters, test_mode=True)
        disp_pr1 = padder.unpad(disp_pr1).cpu().numpy().squeeze()

        # ==== 前向传播：模型 2 ====
        args.model_arch = args.arch2
        with autocast(enabled=args.mixed_precision, dtype=getattr(torch, args.precision_dtype, torch.float16)):
            disp_pr2 = model2(image1_pad, image2_pad, iters=args.valid_iters, test_mode=True)
        disp_pr2 = padder.unpad(disp_pr2).cpu().numpy().squeeze()

        # --- 准备基础图像 ---
        img_np = image1[0].permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        img_show = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        disp_gt = flow_gt[0, 0].cpu().numpy()
        valid_mask = (valid_gt[0].cpu().numpy() > 0.5).squeeze()

        # --- 计算误差 ---
        err_map1 = np.abs(disp_pr1 - disp_gt)
        err_map2 = np.abs(disp_pr2 - disp_gt)

        # --- 统一色彩空间 ---
        vmax = np.percentile(disp_gt[valid_mask], 98) if valid_mask.sum() > 0 else disp_gt.max()

        viz_gt = apply_jet_colormap(disp_gt, mask=valid_mask, vmax=vmax)
        viz_pr1 = apply_jet_colormap(disp_pr1, mask=None, vmax=vmax)
        viz_pr2 = apply_jet_colormap(disp_pr2, mask=None, vmax=vmax)

        viz_err1 = apply_error_colormap(err_map1, mask=valid_mask, max_err=5.0)
        viz_err2 = apply_error_colormap(err_map2, mask=valid_mask, max_err=5.0)

        # --- 添加标签 ---
        img_show = add_label(img_show, "Left Image")
        viz_gt = add_label(viz_gt, "Ground Truth")
        viz_pr1 = add_label(viz_pr1, f"Pred: {args.name1}")
        viz_pr2 = add_label(viz_pr2, f"Pred: {args.name2}")
        viz_err1 = add_label(viz_err1, f"Err: {args.name1}")
        viz_err2 = add_label(viz_err2, f"Err: {args.name2}")

        # --- 组装 2x3 论文网格图 ---
        row1 = np.hstack([img_show, viz_pr1, viz_pr2])
        row2 = np.hstack([viz_gt, viz_err1, viz_err2])
        concat = np.vstack([row1, row2])

        save_path = output_directory / f"{base_name}_compare.png"
        cv2.imwrite(str(save_path), concat)

    print(f"Comparison Done! Results saved to {args.output_directory}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ==== [新增：双模型对比参数] ====
    parser.add_argument('--ckpt1', required=True, help="Checkpoint for Model 1")
    parser.add_argument('--arch1', required=True, choices=['baseline', 'ours'], help="Architecture for Model 1")
    parser.add_argument('--name1', default='Baseline', help="Display name for Model 1")

    parser.add_argument('--ckpt2', required=True, help="Checkpoint for Model 2")
    parser.add_argument('--arch2', required=True, choices=['baseline', 'ours'], help="Architecture for Model 2")
    parser.add_argument('--name2', default='Ours', help="Display name for Model 2")
    # ================================

    parser.add_argument('--data_path', default=None)
    parser.add_argument('--output_directory', default="vis_test/compare_results")
    parser.add_argument('--max_files', type=int, default=50)
    parser.add_argument('--split', default='test', choices=['val', 'validation', 'test'])
    parser.add_argument('--dataset', default='dfc2019', choices=['dfc2019', 'whu'])

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

    args = parser.parse_args()

    # 加载模型 1
    model1 = load_model(args, args.ckpt1, args.arch1)
    # 加载模型 2
    model2 = load_model(args, args.ckpt2, args.arch2)

    visualize_compare_output(model1, model2, args)