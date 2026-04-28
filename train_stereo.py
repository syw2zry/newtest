import warnings
import rasterio

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="Detected call of `lr_scheduler.step()`")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler

from core.igev_stereo import IGEVStereo
from evaluate_stereo import validate_dfc2019, validate_whu
import core.stereo_datasets as datasets
import torch.nn.functional as F


def sequence_loss(args, agg_preds, iter_preds, disp_gt, valid, loss_gamma=0.9):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(iter_preds)
    assert n_predictions >= 1
    
    max_disp = 192

    disp_loss = 0.0
    mag = torch.sum(disp_gt ** 2, dim=1).sqrt()
    valid_mask = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
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

    while should_keep_training:
        for i_batch, (_, *data_blob) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            image1, image2, disp_gt, valid = [x.cuda() for x in data_blob]

            assert model.training
            agg_preds, iter_preds = model(image1, image2, iters=args.train_iters)
            assert model.training

            loss, metrics = sequence_loss(args, agg_preds, iter_preds, disp_gt, valid)
            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            logger.push(metrics)

            if total_steps % validation_frequency == validation_frequency - 1:
                logging.info(
                    f"\n{'=' * 60}\n>>> STATUS: [Step {total_steps + 1}] STOP Training -> START Validation\n{'=' * 60}")
                save_path = Path(args.logdir + '/%d_%s.pth' % (total_steps + 1, args.name))
                logging.info(f"Saving file {save_path.absolute()}")
                torch.save(model.state_dict(), save_path)

                results = {}
                if 'dfc2019' in args.train_datasets:
                    results = validate_dfc2019(model.module, iters=args.valid_iters)
                elif 'whu' in args.train_datasets:
                    in_domain_results = validate_whu(model.module, iters=args.valid_iters, split='validation')
                    for key, value in in_domain_results.items():
                        results[f'whu-in-domain-{key}'] = value

                    zero_shot_results = validate_whu(model.module, iters=args.valid_iters, split='test')
                    for key, value in zero_shot_results.items():
                        results[f'whu-zero-shot-{key}'] = value

                logger.write_dict(results)

                current_metric = None
                if 'dfc-epe' in results:
                    current_metric = results['dfc-epe']
                # 修复：对齐真实的字典键名（双重 whu-）
                elif 'whu-zero-shot-whu-edge-epe' in results:
                    current_metric = results['whu-zero-shot-whu-edge-epe']
                elif 'whu-in-domain-whu-epe' in results:
                    current_metric = results['whu-in-domain-whu-epe']

                if current_metric is not None:
                    logging.info(f"Current Metric: {current_metric:.4f} (Best: {best_epe:.4f})")
                    if current_metric < best_epe:
                        best_epe = current_metric
                        best_save_path = Path(args.logdir + '/%s_best.pth' % args.name)
                        logging.info(f"New best model! Saving to {best_save_path.absolute()}")
                        torch.save(model.state_dict(), best_save_path)

                model.train()
                model.module.freeze_bn()

                logging.info(f"\n{'=' * 60}\n>>> STATUS: Validation Finished -> RESUME Training\n{'=' * 60}")

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    print("FINISHED TRAINING")
    logger.close()
    PATH = args.logdir + '/%s.pth' % args.name
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
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
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
    args = parser.parse_args()
    parser.add_argument('--model_arch', default='ours', choices=['baseline', 'ours'], help='Choose model architecture')

    torch.manual_seed(666)
    np.random.seed(666)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if not args.logdir.endswith(args.name):
        args.logdir = os.path.join(args.logdir, args.name)

    Path(args.logdir).mkdir(exist_ok=True, parents=True)

    train(args)
