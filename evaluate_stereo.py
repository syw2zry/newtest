import sys

sys.path.append('core')

import os
import argparse
import logging
import numpy as np
import torch
from tqdm import tqdm
from core.igev_stereo import IGEVStereo, autocast
import core.stereo_datasets as datasets
from core.utils.utils import InputPadder
import torch.utils.data as data

import warnings
import rasterio

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_inference(model, image1, image2, iters, mixed_prec):
    """统一的推理辅助函数，消除样板代码"""
    padder = InputPadder(image1.shape, divis_by=32)
    image1, image2 = padder.pad(image1, image2)
    with autocast(enabled=mixed_prec):
        flow_pr = model(image1, image2, iters=iters, test_mode=True)
    return padder.unpad(flow_pr)


@torch.no_grad()
def validate_dfc2019(model, iters=32, mixed_prec=False, args=None, split='val'):
    """ Peform validation/testing on the DFC2019 dataset """
    model.eval()

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    aug_params = {}
    val_dataset = datasets.DFC2019(aug_params, split=split)

    batch_size = 1 
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=4, pin_memory=True)

    epe_list = []
    outliers_1px = []
    outliers_3px = []

    print(f"Start validation on DFC2019 with {len(val_dataset)} images...")

    for val_id, (_, image1, image2, flow_gt, valid_gt) in enumerate(tqdm(val_loader)):
        image1 = image1.cuda()
        image2 = image2.cuda()
        flow_gt = flow_gt.cuda()
        valid_gt = valid_gt.cuda()

        flow_pr = run_inference(model, image1, image2, iters=iters, mixed_prec=mixed_prec)

        assert flow_pr.shape == flow_gt.shape, f"{flow_pr.shape} vs {flow_gt.shape}"

        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=1).sqrt()
        epe = epe.flatten()
        val = valid_gt.flatten() >= 0.5

        if val.sum() > 0:
            epe = epe[val]
            epe_list.append(epe.mean().item())
            outliers_1px.append((epe > 1.0).float().mean().item())
            outliers_3px.append((epe > 3.0).float().mean().item())
            
        del image1, image2, flow_gt, valid_gt, flow_pr, padder, epe, val
        torch.cuda.empty_cache()

    mean_epe = np.mean(epe_list)
    mean_1px = np.mean(outliers_1px)
    mean_3px = np.mean(outliers_3px)

    print(f"Validation DFC2019: EPE: {mean_epe:.4f}, 1px: {mean_1px:.4f}, 3px: {mean_3px:.4f}")
    return {'dfc-epe': mean_epe, 'dfc-1px': mean_1px, 'dfc-3px': mean_3px}


@torch.no_grad()
def validate_whu(model, iters=32, mixed_prec=False, split='validation'):
    """ Peform validation/testing using the WHU-Stereo dataset """
    model.eval()
    aug_params = {}

    val_dataset = datasets.WHUStereo(aug_params, split=split)

    val_loader = data.DataLoader(val_dataset, batch_size=1,
                                 pin_memory=True, shuffle=False, num_workers=4)

    print(f"Validating on WHU-Stereo ({len(val_dataset)} samples)...")

    epe_list = []
    outliers_1px_list = []
    outliers_3px_list = []

    for val_id, (_, image1, image2, flow_gt, valid_gt) in enumerate(tqdm(val_loader)):
        image1 = image1.cuda()
        image2 = image2.cuda()
        flow_gt = flow_gt.cuda()
        valid_gt = valid_gt.cuda()

        flow_pr = run_inference(model, image1, image2, iters=iters, mixed_prec=mixed_prec)

        assert flow_pr.shape == flow_gt.shape, f"Shape mismatch: {flow_pr.shape} vs {flow_gt.shape}"

        epe = torch.sum((flow_pr - flow_gt).abs() * valid_gt) / (valid_gt.sum() + 1e-6)

        diff = (flow_pr - flow_gt).abs()
        outlier_mask_1px = (diff > 1.0) & (valid_gt > 0.5)
        outlier_1px = outlier_mask_1px.float().sum() / (valid_gt.sum() + 1e-6)

        outlier_mask_3px = (diff > 3.0) & (valid_gt > 0.5)
        outlier_3px = outlier_mask_3px.float().sum() / (valid_gt.sum() + 1e-6)

        epe_list.append(epe.item())
        outliers_1px_list.append(outlier_1px.item())
        outliers_3px_list.append(outlier_3px.item())

    epe_mean = np.mean(epe_list)
    outliers_1px_mean = np.mean(outliers_1px_list)
    outliers_3px_mean = np.mean(outliers_3px_list)

    print(f"Validation WHU: EPE: {epe_mean:.4f}, 1px: {outliers_1px_mean:.4f}, 3px: {outliers_3px_mean:.4f}")

    return {'whu-epe': epe_mean, 'whu-1px': outliers_1px_mean, 'whu-3px': outliers_3px_mean}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint",
                        default='./pretrained_models/igev_plusplus/sceneflow.pth')

    parser.add_argument('--dataset', help="dataset for evaluation", default='dfc2019',
                        choices=['whu', 'dfc2019'])
    parser.add_argument('--split', default='val', choices=['val', 'validation', 'test'],
                        help='Which data split to evaluate on')
    parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'],
                        help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--batch_size', type=int, default=1, help="batch size used during evaluation.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 736],
                        help="size of the images (not used in default eval but kept for compatibility).")

    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
                        help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=768, help="max disp range")
    parser.add_argument('--s_disp_range', type=int, default=48,
                        help="max disp of small disparity-range geometry encoding volume")
    parser.add_argument('--m_disp_range', type=int, default=96,
                        help="max disp of medium disparity-range geometry encoding volume")
    parser.add_argument('--l_disp_range', type=int, default=192,
                        help="max disp of large disparity-range geometry encoding volume")
    parser.add_argument('--s_disp_interval', type=int, default=1,
                        help="disp interval of small disparity-range geometry encoding volume")
    parser.add_argument('--m_disp_interval', type=int, default=2,
                        help="disp interval of medium disparity-range geometry encoding volume")
    parser.add_argument('--l_disp_interval', type=int, default=4,
                        help="disp interval of large disparity-range geometry encoding volume")
    parser.add_argument('--model_arch', default='ours', choices=['baseline', 'ours'], help='Choose model architecture')

    args = parser.parse_args()

    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        new_checkpoint = {}
        for k, v in checkpoint.items():
            if 'gbc_volume' in k:
                new_key = k.replace('gbc_volume', 'guided_volume')
                new_checkpoint[new_key] = v
            else:
                new_checkpoint[k] = v
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model) / 1e6, '.2f')}M learnable parameters.")

    if args.dataset == 'dfc2019':
        validate_dfc2019(model, iters=args.valid_iters, mixed_prec=args.mixed_precision, args=args, split=args.split)

    elif args.dataset == 'whu':
        validate_whu(model, iters=args.valid_iters, mixed_prec=args.mixed_precision, split=args.split)
