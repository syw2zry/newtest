import sys

sys.path.append('core')

import os
import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from core.igev_stereo import IGEVStereo, autocast
import core.stereo_datasets as datasets
from core.utils.utils import InputPadder
from PIL import Image
import torch.utils.data as data
from pathlib import Path
from matplotlib import pyplot as plt

# --- 屏蔽烦人的警告 ---
import warnings
import rasterio

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# --------------------

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def validate_eth3d(model, iters=32, mixed_prec=False):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, imageR_file, GT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()

        epe_flattened = epe.flatten()

        occ_mask = Image.open(GT_file.replace('disp0GT.pfm', 'mask0nocc.png'))

        occ_mask = np.ascontiguousarray(occ_mask).flatten()

        val = (valid_gt.flatten() >= 0.5) & (occ_mask == 255)
        # val = (valid_gt.flatten() >= 0.5)
        out = (epe_flattened > 1.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(
            f"ETH3D {val_id + 1} out of {len(val_dataset)}. EPE {round(image_epe, 4)} D1 {round(image_out, 4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation ETH3D: EPE %f, D1 %f" % (epe, d1))
    return {'eth3d-epe': epe, 'eth3d-d1': d1}


@torch.no_grad()
def validate_kitti(model, iters=32, mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, image_set='training')
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()

        if val_id > 50:
            elapsed_list.append(end - start)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)
        # val = valid_gt.flatten() >= 0.5

        out = (epe_flattened > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id + 1) % 10 == 0:
            logging.info(
                f"KITTI Iter {val_id + 1} out of {len(val_dataset)}. EPE {round(image_epe, 4)} D1 {round(image_out, 4)}. Runtime: {format(end - start, '.3f')}s ({format(1 / (end - start), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    print(f"Validation KITTI: EPE {epe}, D1 {d1}, {format(1 / avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'kitti-epe': epe, 'kitti-d1': d1}


@torch.no_grad()
def validate_sceneflow(model, iters=32, mixed_prec=False):
    """ Peform validation using the Scene Flow (TEST) split """
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)
    val_loader = data.DataLoader(val_dataset, batch_size=8,
                                 pin_memory=True, shuffle=False, num_workers=8)

    out_list, epe_list = [], []
    for i_batch, (_, *data_blob) in enumerate(tqdm(val_loader)):
        image1, image2, disp_gt, valid_gt = [x for x in data_blob]

        image1 = image1.cuda()
        image2 = image2.cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            disp_pr = model(image1, image2, iters=iters, test_mode=True)
        disp_pr = padder.unpad(disp_pr).cpu()
        assert disp_pr.shape == disp_gt.shape, (disp_pr.shape, disp_gt.shape)
        epe = torch.abs(disp_pr - disp_gt)

        epe = epe.flatten()
        val = (disp_gt.abs().flatten() < 768)
        if (np.isnan(epe[val].mean().item())):
            continue

        out = (epe > 3.0)
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    f = open('test_sceneflow.txt', 'a')
    f.write("Validation Scene Flow: %f, %f\n" % (epe, d1))

    print("Validation Scene Flow: %f, %f" % (epe, d1))
    return {'scene-disp-epe': epe, 'scene-disp-d1': d1}


@torch.no_grad()
def validate_middlebury(model, iters=32, split='MiddEval3', resolution='F', mixed_prec=False):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, split=split, resolution=resolution)
    out_list, epe_list = [], []

    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()
        epe_flattened = epe.flatten()

        occ_mask = Image.open(imageL_file.replace('im0.png', 'mask0nocc.png')).convert('L')
        occ_mask = np.ascontiguousarray(occ_mask, dtype=np.float32).flatten()
        val = (valid_gt.reshape(-1) >= 0.5) & (occ_mask == 255)
        out = (epe_flattened > 2.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(
            f"Middlebury Iter {val_id + 1} out of {len(val_dataset)}. EPE {round(image_epe, 4)} D1 {round(image_out, 4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    f = open('test_middlebury.txt', 'a')
    f.write("Validation Middlebury: %f, %f\n" % (epe, d1))

    print(f"Validation Middlebury{split}: EPE {epe}, D1 {d1}")
    return {f'middlebury{split}-epe': epe, f'middlebury{split}-d1': d1}


# --- [修改版] DFC2019 验证函数 (极致显存优化) ---
@torch.no_grad()
def validate_dfc2019(model, iters=32, mixed_prec=False, args=None, split='val'):
    """ Peform validation/testing on the DFC2019 dataset """
    model.eval()

    # [新增] 强制清空训练阶段遗留的显存碎片
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    aug_params = {}
    # 使用传入的 split 变量
    val_dataset = datasets.DFC2019(aug_params, split=split)

    # [极其关键的修改] 验证高分辨率原图时，Batch Size 必须强制为 1！
    # 不要使用 args.batch_size，那是给训练时裁切小 Patch 用的。
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

        # 验证阶段通常不需要 resize，直接 padding 到能被 32 整除即可
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)

        flow_pr = padder.unpad(flow_pr)

        assert flow_pr.shape == flow_gt.shape, f"{flow_pr.shape} vs {flow_gt.shape}"

        # 计算指标
        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=1).sqrt()
        epe = epe.flatten()
        val = valid_gt.flatten() >= 0.5

        if val.sum() > 0:
            epe = epe[val]
            # .item() 已经将张量转换为了 Python 原生浮点数，不会占用显存
            epe_list.append(epe.mean().item())
            outliers_1px.append((epe > 1.0).float().mean().item())
            outliers_3px.append((epe > 3.0).float().mean().item())
            
        # ==================== [新增] 显存碎片清道夫 ====================
        # 每一张图验证完毕后，立刻物理销毁所有的中间大张量
        del image1, image2, flow_gt, valid_gt, flow_pr, padder, epe, val
        # 强制清空缓存，保证下一张验证图进来时有一块完整干净的显存池
        torch.cuda.empty_cache()
        # ===============================================================

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

    # 使用传入的 split 变量
    val_dataset = datasets.WHUStereo(aug_params, split=split)

    # batch_size=1 或者 4 都可以，验证集通常不需要 shuffle
    val_loader = data.DataLoader(val_dataset, batch_size=1,
                                 pin_memory=True, shuffle=False, num_workers=4)

    print(f"Validating on WHU-Stereo ({len(val_dataset)} samples)...")

    # 初始化指标累加器
    epe_list = []
    outliers_1px_list = []  # 你的核心指标
    outliers_3px_list = []

    for val_id, (_, image1, image2, flow_gt, valid_gt) in enumerate(tqdm(val_loader)):
        image1 = image1.cuda()
        image2 = image2.cuda()
        flow_gt = flow_gt.cuda()
        valid_gt = valid_gt.cuda()

        # InputPadder 确保尺寸能被 32 整除
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            flow_pr = model(image1, image2, iters=iters, test_mode=True)

        flow_pr = padder.unpad(flow_pr)

        assert flow_pr.shape == flow_gt.shape, f"Shape mismatch: {flow_pr.shape} vs {flow_gt.shape}"

        # --- 计算指标 ---
        # 1. 平均端点误差 (EPE)
        epe = torch.sum((flow_pr - flow_gt).abs() * valid_gt) / (valid_gt.sum() + 1e-6)

        # 2. 1px 错误率 (Outlier > 1px)
        diff = (flow_pr - flow_gt).abs()
        outlier_mask_1px = (diff > 1.0) & (valid_gt > 0.5)
        outlier_1px = outlier_mask_1px.float().sum() / (valid_gt.sum() + 1e-6)

        # 3. 3px 错误率 (Outlier > 3px)
        outlier_mask_3px = (diff > 3.0) & (valid_gt > 0.5)
        outlier_3px = outlier_mask_3px.float().sum() / (valid_gt.sum() + 1e-6)

        epe_list.append(epe.item())
        outliers_1px_list.append(outlier_1px.item())
        outliers_3px_list.append(outlier_3px.item())

    # 计算平均值
    epe_mean = np.mean(epe_list)
    outliers_1px_mean = np.mean(outliers_1px_list)
    outliers_3px_mean = np.mean(outliers_3px_list)

    print(f"Validation WHU: EPE: {epe_mean:.4f}, 1px: {outliers_1px_mean:.4f}, 3px: {outliers_3px_mean:.4f}")

    # 返回字典以便记录到 TensorBoard
    return {'whu-epe': epe_mean, 'whu-1px': outliers_1px_mean, 'whu-3px': outliers_3px_mean}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint",
                        default='./pretrained_models/igev_plusplus/sceneflow.pth')

    # --- [修复] 参数定义部分 ---
    parser.add_argument('--dataset', help="dataset for evaluation", default='sceneflow',
                        choices=['whu',"eth3d", "kitti", "sceneflow", "dfc2019"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--split', default='val', choices=['val', 'validation', 'test'],
                        help='Which data split to evaluate on')
    parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'],
                        help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--batch_size', type=int, default=1, help="batch size used during evaluation.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[320, 736],
                        help="size of the images (not used in default eval but kept for compatibility).")

    # Architecture choices (不能删，模型初始化需要这些)
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

    if args.dataset == 'eth3d':
        validate_eth3d(model, iters=args.valid_iters, mixed_prec=args.mixed_precision)

    elif args.dataset == 'kitti':
        validate_kitti(model, iters=args.valid_iters, mixed_prec=args.mixed_precision)

    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(model, iters=args.valid_iters, resolution=args.dataset[-1], mixed_prec=args.mixed_precision)

    elif args.dataset == 'sceneflow':
        validate_sceneflow(model, iters=args.valid_iters, mixed_prec=args.mixed_precision)

    elif args.dataset == 'dfc2019':
        validate_dfc2019(model, iters=args.valid_iters, mixed_prec=args.mixed_precision, args=args, split=args.split)

    elif args.dataset == 'whu':
        validate_whu(model, iters=args.valid_iters, mixed_prec=args.mixed_precision, split=args.split)