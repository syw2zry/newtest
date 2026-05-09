import sys
sys.path.append('core')

import os
import re
import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data

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
def evaluate_dataset(model, dataset_name, split, iters=32, mixed_prec=False):
    model.eval()

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    if dataset_name == 'dfc2019':
        val_dataset = datasets.DFC2019({}, split=split)
    elif dataset_name == 'whu':
        val_dataset = datasets.WHUStereo({}, split=split)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    val_loader = data.DataLoader(val_dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=4)

    print(f"\n{'='*60}")
    print(f"  Evaluating {dataset_name.upper()} ({split}) - {len(val_dataset)} samples")
    print(f"  Physical Boundary Metrics: Global, Edge, Smooth, 1px, 3px")
    print(f"{'='*60}\n")

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).cuda()

    metrics_accumulator = {
        'global_epe': [],
        'edge_epe': [],
        'smooth_epe': [],
        '1px_error': [],
        '3px_error': []
    }

    for _, image1, image2, flow_gt, valid_gt in tqdm(val_loader, desc=f"Validating {dataset_name.upper()}"):
        image1, image2 = image1.cuda(), image2.cuda()
        flow_gt, valid_gt = flow_gt.cuda(), valid_gt.cuda()

        flow_pr = run_inference(model, image1, image2, iters=iters, mixed_prec=mixed_prec)

        metrics = compute_physical_edge_metrics(flow_pr, flow_gt, image1, valid_gt, sobel_x, sobel_y)

        for key in metrics_accumulator:
            metrics_accumulator[key].append(metrics[key])

    results = {key: np.mean(values) for key, values in metrics_accumulator.items()}

    print(f"\n{'='*60}")
    print(f"  [{dataset_name.upper()} - {split}] Final Results:")
    print(f"{'='*60}")
    print(f"  Global-EPE:  {results['global_epe']:.4f}")
    print(f"  Edge-EPE:    {results['edge_epe']:.4f}")
    print(f"  Smooth-EPE:  {results['smooth_epe']:.4f}")
    print(f"  1px-Error:   {results['1px_error']:.4f}")
    print(f"  3px-Error:   {results['3px_error']:.4f}")
    print(f"{'='*60}\n")

    return results


def extract_step_from_ckpt(ckpt_path):
    ckpt_name = Path(ckpt_path).stem
    match = re.match(r'(\d+)_', ckpt_name)
    if match:
        return int(match.group(1))
    match = re.search(r'(\d+)', ckpt_name)
    if match:
        return int(match.group(1))
    return 0


def save_results_to_txt(results, args):
    out_dir = Path("eval_results") / args.dataset / args.model_arch
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_name = Path(args.restore_ckpt).stem
    out_file = out_dir / f"{args.split}_{ckpt_name}.txt"

    with open(out_file, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("       Physical Boundary Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset:       {args.dataset.upper()}\n")
        f.write(f"Split:         {args.split}\n")
        f.write(f"Architecture:  {args.model_arch}\n")
        f.write(f"Checkpoint:    {args.restore_ckpt}\n")
        f.write("-" * 50 + "\n\n")
        f.write("Metrics:\n")
        f.write(f"  Global-EPE:  {results['global_epe']:.4f}\n")
        f.write(f"  Edge-EPE:    {results['edge_epe']:.4f}\n")
        f.write(f"  Smooth-EPE:  {results['smooth_epe']:.4f}\n")
        f.write(f"  1px-Error:   {results['1px_error']:.4f}\n")
        f.write(f"  3px-Error:   {results['3px_error']:.4f}\n")
        f.write("\n" + "=" * 50 + "\n")

    print(f"Results saved to: {out_file.absolute()}")
    return out_file


def write_to_tensorboard(results, args, step):
    log_dir = Path("runs") / f"eval_{args.dataset}_{args.model_arch}"
    writer = SummaryWriter(log_dir=str(log_dir))

    prefix = f"{args.split}/"

    writer.add_scalar(f"{prefix}Global-EPE", results['global_epe'], step)
    writer.add_scalar(f"{prefix}Edge-EPE", results['edge_epe'], step)
    writer.add_scalar(f"{prefix}Smooth-EPE", results['smooth_epe'], step)
    writer.add_scalar(f"{prefix}1px-Error", results['1px_error'], step)
    writer.add_scalar(f"{prefix}3px-Error", results['3px_error'], step)

    writer.close()

    print(f"TensorBoard logs written to: {log_dir.absolute()}")


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
    parser = argparse.ArgumentParser(description="Unified Physical Boundary Evaluation Engine")

    parser.add_argument('--restore_ckpt', type=str, required=True,
                        help="Path to trained weights (.pth)")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['whu', 'dfc2019'],
                        help="Dataset to evaluate")
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'validation', 'test'],
                        help="Data split to use")
    parser.add_argument('--model_arch', type=str, required=True,
                        choices=['baseline', 'ours'],
                        help="Model architecture")

    parser.add_argument('--mixed_precision', action='store_true', default=False,
                        help="Enable mixed precision inference")
    parser.add_argument('--precision_dtype', type=str, default='float32',
                        choices=['float16', 'bfloat16', 'float32'],
                        help="Precision type for inference")
    parser.add_argument('--valid_iters', type=int, default=32,
                        help="Number of iterations during inference")

    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128, 128, 128])
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

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    print(f"\n{'#'*60}")
    print(f"  Physical Boundary Evaluation Engine")
    print(f"{'#'*60}")
    print(f"  Dataset:     {args.dataset.upper()}")
    print(f"  Split:       {args.split}")
    print(f"  Architecture: {args.model_arch}")
    print(f"  Checkpoint:  {args.restore_ckpt}")
    print(f"{'#'*60}\n")

    model = build_model(args)
    model = load_checkpoint(model, args.restore_ckpt)
    model.cuda()
    model.eval()

    print(f"Model parameters: {count_parameters(model) / 1e6:.2f}M\n")

    results = evaluate_dataset(
        model=model,
        dataset_name=args.dataset,
        split=args.split,
        iters=args.valid_iters,
        mixed_prec=args.mixed_precision
    )

    save_results_to_txt(results, args)

    step = extract_step_from_ckpt(args.restore_ckpt)
    write_to_tensorboard(results, args, step)

    print(f"\n{'#'*60}")
    print(f"  Evaluation Complete!")
    print(f"{'#'*60}\n")
