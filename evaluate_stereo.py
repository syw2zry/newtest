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
    edge_x = F.conv2d(gray, sobel_x, padding=1)
    edge_y = F.conv2d(gray, sobel_y, padding=1)
    edge_mag = torch.sqrt(edge_x**2 + edge_y**2)
    
    edge_mask = (edge_mag > edge_threshold).float()
    edge_valid = edge_mask * valid_gt
    
    smooth_mask = (edge_mag <= edge_threshold).float()
    smooth_valid = smooth_mask * valid_gt
    
    epe = torch.abs(flow_pr - flow_gt)
    
    edge_epe = (epe * edge_valid).sum() / (edge_valid.sum() + 1e-6)
    smooth_epe = (epe * smooth_valid).sum() / (smooth_valid.sum() + 1e-6)
    
    return edge_epe.item(), smooth_epe.item()


@torch.no_grad()
def evaluate_dataset(model, loader, iters, mixed_prec):
    model.eval()
    results = {}
    
    all_epe = []
    all_1px = []
    all_3px = []
    all_edge_epe = []
    all_smooth_epe = []
    
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 1, 2], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()

    for i_batch, (_, image1, image2, flow_gt, valid_gt) in enumerate(tqdm(loader)):
        image1 = image1.cuda()
        image2 = image2.cuda()
        flow_gt = flow_gt.cuda()
        valid_gt = valid_gt.cuda()

        flow_pr = run_inference(model, image1, image2, iters, mixed_prec)
        
        if valid_gt.dim() == 3:
            valid_gt = valid_gt.unsqueeze(1)

        epe = torch.abs(flow_pr - flow_gt)
        val_epe = epe[valid_gt > 0]
        
        if val_epe.numel() > 0:
            all_epe.append(val_epe.mean().item())
            all_1px.append((val_epe < 1.0).float().mean().item())
            all_3px.append((val_epe < 3.0).float().mean().item())
            
            e_epe, s_epe = compute_physical_edge_metrics(flow_pr, flow_gt, image1, valid_gt, sobel_x, sobel_y)
            all_edge_epe.append(e_epe)
            all_smooth_epe.append(s_epe)

        if i_batch % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    results['global_epe'] = np.mean(all_epe)
    results['1px_error'] = 1.0 - np.mean(all_1px)
    results['3px_error'] = 1.0 - np.mean(all_3px)
    results['edge_epe'] = np.mean(all_edge_epe)
    results['smooth_epe'] = np.mean(all_smooth_epe)

    return results


def build_model(args):
    model = IGEVStereo(args)
    return model


def load_checkpoint(model, ckpt_path):
    logging.info(f"Loading checkpoint from {ckpt_path}...")
    state_dict = torch.load(ckpt_path)
    # 处理 DataParallel 包装
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=True)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='dataset name', choices=['sceneflow', 'kitti', 'middlebury', 'eth3d', 'dfc2019', 'whu'])
    parser.add_argument('--split', help='dataset split', default='test')
    parser.add_argument('--dfc_region', default='all', choices=['jax', 'oma', 'all'], help='DFC2019 region selection')
    parser.add_argument('--ckpt', help='checkpoint path')
    
    # 架构相关参数
    parser.add_argument('--max_disp', type=int, default=768)
    parser.add_argument('--model_arch', default='ours', choices=['baseline', 'ours'])
    parser.add_argument('--n_downsample', type=int, default=2)
    parser.add_argument('--n_gru_layers', type=int, default=2)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3)
    parser.add_argument('--corr_levels', type=int, default=2)
    parser.add_argument('--corr_radius', type=int, default=4)
    
    # 代价体参数
    parser.add_argument('--s_disp_range', type=int, default=48)
    parser.add_argument('--m_disp_range', type=int, default=96)
    parser.add_argument('--l_disp_range', type=int, default=192)
    parser.add_argument('--s_disp_interval', type=int, default=1)
    parser.add_argument('--m_disp_interval', type=int, default=2)
    parser.add_argument('--l_disp_interval', type=int, default=4)

    # 运行配置
    parser.add_argument('--valid_iters', type=int, default=32)
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--precision_dtype', default='float16')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    # 1. 准备验证集
    if args.dataset == 'dfc2019':
        # 传递 region 参数以确保划分一致
        val_dataset = datasets.DFC2019({}, split=args.split, region=args.dfc_region)
    elif args.dataset == 'whu':
        val_dataset = datasets.WHUStereo({}, split=args.split)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
        
    val_loader = data.DataLoader(val_dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=4)

    # 2. 准备模型
    model = build_model(args)
    model.cuda()
    model = load_checkpoint(model, args.ckpt)

    # 3. 开始评估
    results = evaluate_dataset(model, val_loader, args.valid_iters, args.mixed_precision)

    print(f"\n     [Result] Dataset: {args.dataset} | Region: {args.dfc_region} | Split: {args.split}")
    print(f"     G-EPE: {results['global_epe']:.3f} | Edge: {results['edge_epe']:.3f} | Smooth: {results['smooth_epe']:.3f} | 1px: {results['1px_error']:.3f} | 3px: {results['3px_error']:.3f}")

    # 4. 汇总 CSV
    csv_filename = f"summary_{args.dataset}_{args.dfc_region}_{args.split}.csv"
    csv_path = Path(csv_filename)
    file_exists = csv_path.exists()

    with open(csv_path, 'a', encoding='utf-8') as f:
        if not file_exists:
            f.write("Model_Name,Global_EPE,Edge_EPE,Smooth_EPE,1px_Error,3px_Error\n")
        f.write(f"{args.ckpt},{results['global_epe']:.4f},{results['edge_epe']:.4f},{results['smooth_epe']:.4f},{results['1px_error']:.4f},{results['3px_error']:.4f}\n")