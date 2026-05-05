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


def format_image_with_label(img, text, bg_color=(255, 255, 255), text_color=(0, 0, 0)):
    """专业的白底隔离+居中标签排版"""
    top_margin = 60
    bottom_margin = 15
    left_margin = 15
    right_margin = 15

    padded_img = cv2.copyMakeBorder(
        img, top_margin, bottom_margin, left_margin, right_margin,
        cv2.BORDER_CONSTANT, value=bg_color
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.1
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    text_x = left_margin + max(0, (img.shape[1] - text_width) // 2)
    text_y = top_margin - 15

    cv2.putText(
        padded_img, text, (text_x, text_y),
        font, font_scale, text_color, thickness, cv2.LINE_AA
    )

    return padded_img


@torch.no_grad()
def visualize_single_output(model, args):
    model.eval()

    # ==================== [动态数据集路由] ====================
    if args.dataset == 'whu':
        root_path = args.data_path if args.data_path else '/home/roy/projects/YAOWEI/data/WHU_dataset_big'
        val_dataset = datasets.WHUStereo(aug_params=None, root=root_path, split=args.split)
    else:
        root_path = args.data_path if args.data_path else '/home/roy/projects/YAOWEI/data/dfc2019-big'
        val_dataset = datasets.DFC2019(aug_params=None, root=root_path, split=args.split)
    # ==========================================================

    max_count = args.max_files if args.max_files > 0 else len(val_dataset)
    print(f"Dataset Loaded ({args.dataset}). Total valid images: {len(val_dataset)}. Will process: {max_count}")

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

        # 准备左图 RGB
        img_np = image1[0].permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        img_show = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 准备 GT 和 Mask
        disp_gt = flow_gt[0, 0].cpu().numpy()
        valid_mask = (valid_gt[0].cpu().numpy() > 0.5).squeeze()

        # 计算绝对误差
        err_map = np.abs(disp_pr - disp_gt)

        if valid_mask.sum() > 0:
            vmax = np.percentile(disp_gt[valid_mask], 98)
        else:
            vmax = disp_pr.max()

        viz_gt = apply_jet_colormap(disp_gt, mask=valid_mask, vmax=vmax)
        viz_pr = apply_jet_colormap(disp_pr, mask=None, vmax=vmax)
        viz_err = apply_error_colormap(err_map, mask=valid_mask, max_err=5.0)

        # --- 重点：应用论文级别的排版 ---
        img_show = format_image_with_label(img_show, "Left Image")
        viz_gt = format_image_with_label(viz_gt, "Ground Truth")
        viz_pr = format_image_with_label(viz_pr, "Our Prediction")
        viz_err = format_image_with_label(viz_err, "Error Map")

        # 拼接成 1x4 的长条图 (中间会有漂亮的白底空隙)
        concat = np.hstack([img_show, viz_gt, viz_pr, viz_err])

        save_path = output_directory / f"{base_name}_vis.png"
        cv2.imwrite(str(save_path), concat)

    print(f"Done! Results saved to {args.output_directory}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
    parser.add_argument('--data_path', help="dataset root path", default=None)
    parser.add_argument('--output_directory', default="result_vis_whu")
    parser.add_argument('--max_files', type=int, default=50)
    parser.add_argument('--split', default='test', choices=['val', 'validation', 'test'])
    parser.add_argument('--dataset', default='whu', choices=['dfc2019', 'whu'], help="which dataset to use")
    
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

    new_checkpoint = {}
    for k, v in checkpoint.items():
        if 'gbc_volume' in k:
            new_key = k.replace('gbc_volume', 'guided_volume')
            new_checkpoint[new_key] = v
        else:
            new_checkpoint[k] = v

    model.load_state_dict(new_checkpoint, strict=False)
    model.cuda()

    visualize_single_output(model, args)