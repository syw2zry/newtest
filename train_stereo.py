import warnings
import rasterio

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="Detected call of `lr_scheduler.step()`")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import argparse
import logging
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler

from core.igev_stereo import IGEVStereo
from evaluate_stereo import evaluate_dataset
import core.stereo_datasets as datasets
import torch.nn.functional as F


def sequence_loss(args, agg_preds, iter_preds, disp_gt, valid, loss_gamma=0.9):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(iter_preds)
    assert n_predictions >= 1
    
    # [全局修复 1] 动态读取最大视差，释放高分辨率卫星影像中的高楼大厦监督信号！
    max_disp = args.max_disp

    disp_loss = 0.0
    mag = torch.sum(disp_gt ** 2, dim=1).sqrt()
    valid_mask = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    
    # ====================================================================
    # [全局修复 2] 🚨 防御性编程：防止水域/纯黑边缘导致空掩膜，引发 NaN 梯度爆炸 🚨
    # ====================================================================
    if valid_mask.sum() < 10:  # 如果当前裁剪块里有效像素极少
        # 构建一个连着计算图的虚拟 0 Loss，骗过 PyTorch 的反向传播，安全跳过此 Batch
        dummy_loss = sum([p.sum() for p in iter_preds]) * 0.0
        if isinstance(agg_preds, list):
            dummy_loss += sum([p.sum() for p in agg_preds]) * 0.0
        metrics = {'epe': 0.0, '1px': 0.0, '3px': 0.0, '5px': 0.0}
        return dummy_loss, metrics
    # ====================================================================

    assert valid_mask.shape == disp_gt.shape, [valid_mask.shape, disp_gt.shape]
    assert not torch.isinf(disp_gt[valid_mask.bool()]).any()

    disp_loss += 1.0 * F.smooth_l1_loss(agg_preds[0][valid_mask.bool()], disp_gt[valid_mask.bool()], reduction='mean')
    disp_loss += 0.5 * F.smooth_l1_loss(agg_preds[1][valid_mask.bool()], disp_gt[valid_mask.bool()], reduction='mean')
    disp_loss += 0.2 * F.smooth_l1_loss(agg_preds[2][valid_mask.bool()], disp_gt[valid_mask.bool()], reduction='mean')

    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions - 1))
        i_weight = adjusted_loss_gamma ** (n_predictions - i - 1)
        i_loss = (iter_preds[i] - disp_gt).abs()
        assert i_loss.shape == valid_mask.shape, [i_loss.shape, valid_mask.shape, disp_gt.shape, iter_preds[i].shape]
        disp_loss += i_weight * i_loss[valid_mask.bool()].mean()

    epe = torch.sum((iter_preds[-1] - disp_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid_mask.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return disp_loss, metrics


def frequency_orthogonal_loss(feat_low: torch.Tensor, feat_high: torch.Tensor, 
                               ortho_weight: float = 0.1, smooth_weight: float = 0.05, 
                               sharp_weight: float = 0.05) -> torch.Tensor: 
    """ 
    频域正交损失：防止高低频特征语义坍塌 
    """ 
    feat_low = feat_low.float() 
    feat_high = feat_high.float() 
    
    B, C, H, W = feat_low.shape 
    
    feat_low_flat = feat_low.view(B, C, -1) 
    feat_high_flat = feat_high.view(B, C, -1) 
    
    feat_low_norm = F.normalize(feat_low_flat, p=2, dim=2) 
    feat_high_norm = F.normalize(feat_high_flat, p=2, dim=2) 
    
    cos_sim = torch.bmm(feat_low_norm, feat_high_norm.transpose(1, 2)) 
    ortho_loss = cos_sim.abs().mean() 
    
    grad_x_low = feat_low[:, :, :, 1:] - feat_low[:, :, :, :-1] 
    grad_y_low = feat_low[:, :, 1:, :] - feat_low[:, :, :-1, :] 
    smooth_loss = (grad_x_low.abs().mean() + grad_y_low.abs().mean()) * 0.5 
    
    grad_x_high = feat_high[:, :, :, 1:] - feat_high[:, :, :, :-1] 
    grad_y_high = feat_high[:, :, 1:, :] - feat_high[:, :, :-1, :] 
    
    grad_x_high_aligned = grad_x_high[:, :, :-1, :] 
    grad_y_high_aligned = grad_y_high[:, :, :, :-1] 
    
    grad_high_mag = (grad_x_high_aligned.abs() + grad_y_high_aligned.abs()) * 0.5 
    
    margin = 0.1 
    sharp_loss = F.relu(margin - grad_high_mag.mean(dim=1, keepdim=True)).mean() 
    
    total_loss = (ortho_weight * ortho_loss + 
                  smooth_weight * smooth_loss + 
                  sharp_weight * sharp_loss) 
    
    return total_loss


def generate_edge_pseudo_label(image: torch.Tensor) -> torch.Tensor:
    """
    使用 Sobel 算子生成边缘伪标签（仅用于热启动阶段）
    
    Args:
        image: 输入图像 [B, 3, H, W]
    
    Returns:
        edge_label: 边缘伪标签 [B, 1, H/4, W/4]，值域 [0, 1]
    """
    gray_weights = torch.tensor([0.2989, 0.5870, 0.1140],
                                 device=image.device, dtype=image.dtype).view(1, 3, 1, 1)
    gray = (image * gray_weights).sum(dim=1, keepdim=True)
    
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                           device=image.device, dtype=image.dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                           device=image.device, dtype=image.dtype).view(1, 1, 3, 3)
    
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    
    threshold = grad_mag.mean() + grad_mag.std()
    edge_label = (grad_mag > threshold).float()
    
    edge_label = F.interpolate(edge_label, scale_factor=0.25, mode='nearest')
    
    return edge_label


def edge_warmup_loss(image1: torch.Tensor, edge_pred: torch.Tensor, 
                     warmup_epochs: int = 5, current_epoch: int = 0) -> torch.Tensor: 
    """ 
    EdgeNet BCE 热启动辅助损失 
    """ 
    if current_epoch >= warmup_epochs: 
        return torch.tensor(0.0, device=image1.device, dtype=image1.dtype) 
    
    with torch.no_grad(): 
        edge_label = generate_edge_pseudo_label(image1) 
        
        if edge_pred.shape != edge_label.shape: 
            edge_label = F.interpolate(edge_label, size=edge_pred.shape[2:], mode='nearest') 
    
    bce_loss = F.binary_cross_entropy(edge_pred.float(), edge_label.float(), reduction='mean') 
    
    warmup_weight = 1.0 - (current_epoch / warmup_epochs) 
    
    return warmup_weight * bce_loss


def update_temperature_hook(model, current_epoch: int, total_epochs: int):
    """
    在 epoch 循环中调用温度退火
    
    Args:
        model: nn.DataParallel 包装的模型
        current_epoch: 当前训练轮次
        total_epochs: 总训练轮次
    """
    if hasattr(model, 'module'):
        guided_volume = model.module.guided_volume
    else:
        guided_volume = model.guided_volume
    
    if hasattr(guided_volume, 'update_temperature'):
        guided_volume.update_temperature(
            current_epoch=current_epoch,
            total_epochs=total_epochs,
            start_temp=2.0,
            end_temp=0.1
        )
        current_temp = guided_volume.adaptive_temp.item()
        logging.info(f"[Temperature Annealing] Epoch {current_epoch}: temp = {current_temp:.4f}")
    else:
        logging.warning("guided_volume has no update_temperature method")


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    if getattr(args, 'exp_mode', 'full') == 'overfit':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    else:
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100,
                                                  pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler


class Logger:
    SUM_FREQ = 100

    def __init__(self, model, scheduler, logdir):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.logdir = logdir
        self.writer = SummaryWriter(log_dir=self.logdir)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.logdir)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k] / Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == 0:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.logdir)

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    exp_mode = getattr(args, 'exp_mode', 'full')
    if exp_mode == 'overfit':
        args.num_steps = 500
        validation_frequency = 50
    elif exp_mode == 'fast':
        args.num_steps = 20000
        validation_frequency = 200
    else:
        args.num_steps = 100000
        validation_frequency = 1000

    model = nn.DataParallel(IGEVStereo(args))
    print("Parameter Count: %d" % count_parameters(model))

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    global_batch_num = 0
    logger = Logger(model, scheduler, args.logdir)

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

        ckpt_name = os.path.basename(args.restore_ckpt)
        match = re.match(r'(\d+)_', ckpt_name)
        if match:
            total_steps = int(match.group(1))
            global_batch_num = total_steps
            logger.total_steps = total_steps
            logging.info(f"Successfully restored step counter to: {total_steps}")

            for _ in range(total_steps):
                scheduler.step()

    model.cuda()
    model.train()
    model.module.freeze_bn()

    scaler = GradScaler(enabled=args.mixed_precision)

    best_epe = float('inf')
    should_keep_training = True
    
    steps_per_epoch = len(train_loader)
    total_epochs = args.num_steps // steps_per_epoch
    epoch_counter = 0
    warmup_epochs = 5

    while should_keep_training:
        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            image1, image2, disp_gt, valid = [x.cuda() for x in data_blob]

            assert model.training
            
            outputs = model(image1, image2, iters=args.train_iters)
            
            if len(outputs) == 3:
                agg_preds, iter_preds, aux_outputs = outputs
                edge_pred, f_low, f_high = aux_outputs
            else:
                agg_preds, iter_preds = outputs
                edge_pred, f_low, f_high = None, None, None
            
            assert model.training

            loss, metrics = sequence_loss(args, agg_preds, iter_preds, disp_gt, valid)
            
            if f_low is not None and f_high is not None:
                freq_loss = frequency_orthogonal_loss(f_low, f_high,
                                                       ortho_weight=0.1,
                                                       smooth_weight=0.05,
                                                       sharp_weight=0.05)
                loss = loss + 0.01 * freq_loss
                logger.writer.add_scalar("freq_orthogonal_loss", freq_loss.item(), global_batch_num)
            
            if edge_pred is not None:
                e_loss = edge_warmup_loss(image1, edge_pred, warmup_epochs, epoch_counter)
                if e_loss.item() > 0:
                    loss = loss + 0.1 * e_loss
                    logger.writer.add_scalar("edge_warmup_loss", e_loss.item(), global_batch_num)
            
            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1

            # ==========================================================
            # 🛡️ 终极防御装甲：拦截 FP16 溢出导致的 NaN/Inf Loss
            # ==========================================================
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"🚨 [Step {total_steps}] 检测到 NaN/Inf Loss (FP16 溢出)！已拦截，保护模型权重，跳过此 Batch。")
                optimizer.zero_grad() # 清空这个剧毒的梯度
                
                # [新增防御] 必须手动切断引用，否则废弃的计算图会导致下一个 Batch 显存翻倍直接 OOM
                del outputs, loss
                if 'agg_preds' in locals(): del agg_preds
                if 'iter_preds' in locals(): del iter_preds
                if 'freq_loss' in locals(): del freq_loss
                if 'e_loss' in locals(): del e_loss
                if 'f_low' in locals(): del f_low, f_high
                
                # 强制清理 CUDA 缓存碎片，将空间还给显卡
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
                continue              # 安全跳过
            # ==========================================================

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            logger.push(metrics)

            if total_steps % validation_frequency == validation_frequency - 1:
                import math # 确保导入了 math 以监控 NaN
                logging.info(
                    f"\n{'=' * 60}\n>>> STATUS: [Step {total_steps + 1}] STOP Training -> START Validation\n{'=' * 60}")
                
                # [全局修复 3] 路径斜杠安全处理，获取防崩溃的文件名
                safe_name = args.name.replace('/', '_')
                
                # [全局修复 4] 恢复中间过程的阶段性模型保存
                save_path = Path(args.logdir + '/%d_%s.pth' % (total_steps + 1, safe_name))
                logging.info(f"Saving intermediate model to {save_path.absolute()}")
                torch.save(model.state_dict(), save_path)


                
                results = {}
                if 'dfc2019' in args.train_datasets:
                    target_region = getattr(args, 'dfc_region', 'all')
                    val_dataset = datasets.DFC2019({}, split='val', region=target_region)
                    val_loader = data.DataLoader(val_dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=4)
                    
                    dfc_results = evaluate_dataset(model.module, val_loader, iters=args.valid_iters, mixed_prec=args.mixed_precision)
                    for key, value in dfc_results.items():
                        results[f'dfc-{key}'] = value 
                        
                elif 'whu' in args.train_datasets:
                    # 1. 验证同分布 (In-Domain)
                    target_region = getattr(args, 'whu_region', 'all')
                    in_domain_dataset = datasets.WHUStereo({}, split='validation', region=target_region)
                    in_domain_loader = data.DataLoader(in_domain_dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=4)
                    
                    in_domain_results = evaluate_dataset(model.module, in_domain_loader, iters=args.valid_iters, mixed_prec=args.mixed_precision)
                    for key, value in in_domain_results.items():
                        results[f'InDomain/{key}'] = value

                    # 2. 验证跨城泛化 (Zero-Shot)
                    target_region = getattr(args, 'whu_region', 'all')
                    zero_shot_dataset = datasets.WHUStereo({}, split='test', region=target_region)
                    zero_shot_loader = data.DataLoader(zero_shot_dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=4)
                    
                    zero_shot_results = evaluate_dataset(model.module, zero_shot_loader, iters=args.valid_iters, mixed_prec=args.mixed_precision)
                    for key, value in zero_shot_results.items():
                        results[f'ZeroShot/{key}'] = value

                logger.write_dict(results)

                # ==========================================================
                # --- 获取当前评估的核心指标 (严格对齐刚刚修改的清爽键名) ---
                # ==========================================================
                current_metric = None
                if 'dfc-global_epe' in results:
                    current_metric = results['dfc-global_epe']
                # 针对边缘保真度，现在的新键名是带下划线的 edge_epe
                elif 'ZeroShot/edge_epe' in results: 
                    current_metric = results['ZeroShot/edge_epe']
                elif 'InDomain/global_epe' in results:
                    current_metric = results['InDomain/global_epe']

                # ==========================================================
                # [全局修复 6] 最佳模型判定与 NaN 崩溃监控
                # ==========================================================
                if current_metric is not None:
                    if math.isnan(current_metric):
                        logging.error(f"❌ 严重警告: 当前验证指标为 NaN！模型在验证集上已崩！请考虑从上一个 checkpoint 恢复。")
                    else:
                        logging.info(f"Current Metric: {current_metric:.4f} (Best: {best_epe:.4f})")
                        if current_metric < best_epe:
                            best_epe = current_metric
                            # 安全保存最优模型
                            best_save_path = Path(args.logdir + '/%s_best.pth' % safe_name)
                            logging.info(f"🌟 New best model! Saving to {best_save_path.absolute()}")
                            torch.save(model.state_dict(), best_save_path)
                else:
                    logging.warning("Warning: current_metric is None! Check if your dictionary keys match.")

                model.train()
                model.module.freeze_bn()

                logging.info(f"\n{'=' * 60}\n>>> STATUS: Validation Finished -> RESUME Training\n{'=' * 60}")

            total_steps += 1
            
            if total_steps % steps_per_epoch == 0:
                epoch_counter += 1
                if getattr(args, 'model_arch', 'ours') == 'ours':
                    update_temperature_hook(model, epoch_counter, total_epochs)

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    print("FINISHED TRAINING")
    logger.close()
    PATH = args.logdir + '/%s.pth' % args.name.replace('/', '_')
    torch.save(model.state_dict(), PATH)

    return PATH


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='igev-stereo', help="name your experiment")
    parser.add_argument('--restore_ckpt', default=None, help='load the weights from a specific checkpoint')
    parser.add_argument('--logdir', default='./checkpoints', help='the directory to save logs and checkpoints')
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float16', choices=['float16', 'bfloat16', 'float32'],
                        help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--exp_mode', default='full', choices=['full', 'fast', 'overfit'], help='Experiment mode configuration')
    parser.add_argument('--batch_size', type=int, default=8, help="batch size used during training.")
    parser.add_argument('--train_datasets', default='dfc2019',
                        choices=['whu', 'dfc2019'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=200000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 768],
                        help="size of the random image crops used during training.")
    parser.add_argument('--train_iters', type=int, default=22,
                        help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    parser.add_argument('--valid_iters', type=int, default=32,
                        help='number of flow-field updates during validation forward pass')

    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=2, help="number of hidden GRU levels")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3,
                        help="hidden state and context dimensions")
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

    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4], help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'],
                        help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.4, 0.8],
                        help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    
    parser.add_argument('--model_arch', default='ours', choices=['baseline', 'ours'], help='Choose model architecture')
    
    parser.add_argument('--dfc_region', default='all', choices=['jax', 'oma', 'all'], help='Choose specific region for DFC2019 dataset (jax/oma/all)')

    args = parser.parse_args()
    
    torch.manual_seed(666)
    np.random.seed(666)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if not args.logdir.endswith(args.name):
        args.logdir = os.path.join(args.logdir, args.name)

    Path(args.logdir).mkdir(exist_ok=True, parents=True)

    train(args)
