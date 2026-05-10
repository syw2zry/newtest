import sys

sys.path.append('core')

import os
import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import torch.utils.data as data
import gc

from core.igev_stereo import IGEVStereo, autocast
import core.stereo_datasets as datasets
from core.utils.utils import InputPadder

import warnings
import rasterio

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_inference(model, image1, image2, iters, mixed_prec):
    padder = InputPadder(image1.shape, divis_by=32)
    image1, image2 = padder.pad(image1, image2)
    with autocast(enabled=mixed_prec):
        flow_pr = model(image1, image2, iters=iters, test_mode=True)
    return padder.unpad(flow_pr)


def compute_physical_edge_metrics(flow_pr, flow_gt, image1, valid_gt, sobel_x, sobel_y, edge_threshold=30.0):
    if valid_gt.dim() == 3:
        valid_gt = valid_gt.unsqueeze(1)

    gray = 0.299 * image1[:, 0:1, :, :] + 0.587 * image1[:, 1:2, :, :] + 0.114 * image1[:, 2:3, :, :]
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    edge_mask = (gradient_magnitude > edge_threshold)
    smooth_mask = ~edge_mask

    diff = (flow_pr - flow_gt).abs()

    global_epe = (diff * valid_gt).sum() / (valid_gt.sum() + 1e-6)

    edge_valid = (valid_gt > 0.5) & edge_mask
    edge_epe = (diff * edge_valid.float()).sum() / (edge_valid.sum() + 1e-6)

    smooth_valid = (valid_gt > 0.5) & smooth_mask
    smooth_epe = (diff * smooth_valid.float()).sum() / (smooth_valid.sum() + 1e-6)

    outlier_1px = ((diff > 1.0) & (valid_gt > 0.5)).float().sum() / (valid_gt.sum() + 1e-6)
    outlier_3px = ((diff > 3.0) & (valid_gt > 0.5)).float().sum() / (valid_gt.sum() + 1e-6)

    return {
        'global_epe': global_epe.item(),
        'edge_epe': edge_epe.item(),
        'smooth_epe': smooth_epe.item(),
        '1px_error': outlier_1px.item(),
        '3px_error': outlier_3px.item()
    }


@torch.no_grad()
def evaluate_dataset(model, val_loader, iters=32, mixed_prec=False):
    model.eval()

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()

    metrics_accumulator = {k: [] for k in ['global_epe', 'edge_epe', 'smooth_epe', '1px_error', '3px_error']}

    for _, image1, image2, flow_gt, valid_gt in tqdm(val_loader, desc=f"Evaluating", leave=False):
        image1, image2 = image1.cuda(), image2.cuda()
        flow_gt, valid_gt = flow_gt.cuda(), valid_gt.cuda()

        flow_pr = run_inference(model, image1, image2, iters=iters, mixed_prec=mixed_prec)
        metrics = compute_physical_edge_metrics(flow_pr, flow_gt, image1, valid_gt, sobel_x, sobel_y)

        for key in metrics_accumulator:
            metrics_accumulator[key].append(metrics[key])

    results = {key: np.mean(values) for key, values in metrics_accumulator.items()}
    return results


def build_model(args):
    model = IGEVStereo(args)
    model = torch.nn.DataParallel(model, device_ids=[0])
    return model


def load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    new_checkpoint = {}
    for k, v in checkpoint.items():
        new_key = k.replace('gbc_volume', 'guided_volume')
        new_checkpoint[new_key] = v
    model.load_state_dict(new_checkpoint, strict=False)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single Model Leaderboard Evaluation Engine")

    # 1. 明确指向单个 .pth 文件
    parser.add_argument('--restore_ckpt', type=str, required=True,
                        help="Path to the specific .pth file you want to evaluate")

    # 2. 架构拨片与数据集
    parser.add_argument('--model_arch', type=str, required=True, choices=['baseline', 'ours'],
                        help="Model architecture switch: 'baseline' (IGEV++) or 'ours' (EdgeFreq-Net)")
    parser.add_argument('--dataset', type=str, required=True, choices=['whu', 'dfc2019'], help="Dataset to evaluate")
    parser.add_argument('--split', type=str, default='test', choices=['val', 'validation', 'test'],
                        help="Data split to use")

    # 3. 架构底层参数
    parser.add_argument('--mixed_precision', action='store_true', default=False)
    parser.add_argument('--precision_dtype', default='float16', choices=['float16', 'bfloat16', 'float32'],
                        help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--valid_iters', type=int, default=32)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128, 128, 128])
    parser.add_argument('--corr_levels', type=int, default=2)
    parser.add_argument('--corr_radius', type=int, default=4)
    parser.add_argument('--n_downsample', type=int, default=2)
    parser.add_argument('--n_gru_layers', type=int, default=2)
    parser.add_argument('--max_disp', type=int, default=768)
    parser.add_argument('--s_disp_range', type=int, default=48)
    parser.add_argument('--m_disp_range', type=int, default=96)
    parser.add_argument('--l_disp_range', type=int, default=192)
    parser.add_argument('--s_disp_interval', type=int, default=1)
    parser.add_argument('--m_disp_interval', type=int, default=2)
    parser.add_argument('--l_disp_interval', type=int, default=4)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    ckpt_path = Path(args.restore_ckpt)
    if not ckpt_path.is_file() or ckpt_path.suffix != '.pth':
        raise ValueError(f"The path provided must be a valid .pth file: {ckpt_path}")

    # 自动生成能识别身份的模型名称：[架构]_文件夹名_权重名
    model_name = f"[{args.model_arch.upper()}]_{ckpt_path.parent.name}_{ckpt_path.stem}"

    print(f"\n{'=' * 70}")
    print(f"  Leaderboard Evaluation -> Dataset: {args.dataset.upper()} | Split: {args.split}")
    print(f"  Model Name: {model_name}")
    print(f"{'=' * 70}\n")

    # 1. 准备数据集
    if args.dataset == 'dfc2019':
        val_dataset = datasets.DFC2019({}, split=args.split)
    elif args.dataset == 'whu':
        val_dataset = datasets.WHUStereo({}, split=args.split)
    val_loader = data.DataLoader(val_dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=4)

    # 2. 准备模型
    model = build_model(args)
    model.cuda()
    model = load_checkpoint(model, ckpt_path)

    # 3. 开始评估
    results = evaluate_dataset(model, val_loader, args.valid_iters, args.mixed_precision)

    print(
        f"\n     [Result] G-EPE: {results['global_epe']:.3f} | Edge: {results['edge_epe']:.3f} | Smooth: {results['smooth_epe']:.3f} | 1px: {results['1px_error']:.3f} | 3px: {results['3px_error']:.3f}")

    # ==========================================
    # 4. 原地追加 CSV 汇总表逻辑
    # ==========================================
    csv_filename = f"summary_{args.dataset}_{args.split}.csv"
    csv_path = Path(csv_filename)

    file_exists = csv_path.exists()

    # 使用 'a' 模式（append），如果文件存在则追加，不存在则创建
    with open(csv_path, 'a', encoding='utf-8') as f:
        # 如果是新创建的文件，先写入表头
        if not file_exists:
            f.write("Model_Name,Global_EPE,Edge_EPE,Smooth_EPE,1px_Error,3px_Error\n")

        # 写入本次模型的数据
        f.write(
            f"{model_name},{results['global_epe']:.4f},{results['edge_epe']:.4f},{results['smooth_epe']:.4f},{results['1px_error']:.4f},{results['3px_error']:.4f}\n")

    print(f"\n{'=' * 70}")
    print(f" Evaluation Completed!")
    print(f" Result appended to: -> {csv_path.absolute()} <-")
    print(f"{'=' * 70}\n")

    # 清理缓存
    gc.collect()
    torch.cuda.empty_cache()