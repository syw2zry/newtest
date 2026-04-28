import numpy as np
import rasterio
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import copy
import random
from glob import glob
import os.path as osp

from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        disp = self.disparity_reader(self.disparity_list[index])

        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 1024

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        disp = np.array(disp).astype(np.float32)
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1024) & (flow[1].abs() < 1024)

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW] * 2 + [padH] * 2)
            img2 = F.pad(img2, [padW] * 2 + [padH] * 2)

        flow = flow[:1]
        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self

    def __len__(self):
        return len(self.image_list)


class DFC2019(StereoDataset):
    def __init__(self, aug_params=None, root='/home/roy/projects/YAOWEI/data/dfc2019-big', split='train'):
        super(DFC2019, self).__init__(aug_params, sparse=True)
        self.root = root
        self.split = split

        self.left_dir = os.path.join(root, 'left')
        self.right_dir = os.path.join(root, 'right')
        self.disp_dir = os.path.join(root, 'disp')

        left_images = sorted(glob(osp.join(self.left_dir, '*_LEFT_RGB.tif')))

        state = np.random.get_state()
        np.random.seed(1000)
        indices = np.random.permutation(len(left_images))
        np.random.set_state(state)

        num_images = len(left_images)
        train_end = int(0.8 * num_images)
        val_end = int(0.9 * num_images)

        if split == 'train':
            split_indices = indices[:train_end]
        elif split in ['val', 'validation']:
            split_indices = indices[train_end:val_end]
        elif split == 'test':
            split_indices = indices[val_end:]
        else:
            split_indices = indices

        for idx in split_indices:
            l_path = left_images[idx]
            fname = osp.basename(l_path)

            fname_right = fname.replace('LEFT', 'RIGHT')
            r_path = osp.join(self.right_dir, fname_right)
            d_path = osp.join(self.disp_dir, fname.replace('RGB', 'DSP'))

            self.image_list.append([l_path, r_path])
            self.disparity_list.append(d_path)

        logging.info(f"DFC2019 {split} set: {len(self.image_list)} pairs loaded.")

    def __getitem__(self, index):
        index = index % len(self.image_list)
        disp_path = self.disparity_list[index]
        img1_path, img2_path = self.image_list[index]

        with rasterio.open(img1_path) as src:
            img1 = src.read([1, 2, 3]).transpose(1, 2, 0)
        with rasterio.open(img2_path) as src:
            img2 = src.read([1, 2, 3]).transpose(1, 2, 0)
        with rasterio.open(disp_path) as src:
            disp = src.read(1)

        img1 = np.array(img1).astype(np.float32)
        img2 = np.array(img2).astype(np.float32)

        if img1.max() > 255:
            img1 = (img1 / img1.max()) * 255.0
        if img2.max() > 255:
            img2 = (img2 / img2.max()) * 255.0

        img1 = np.clip(img1, 0, 255).astype(np.uint8)
        img2 = np.clip(img2, 0, 255).astype(np.uint8)

        disp = np.array(disp).astype(np.float32)

        valid_mask = (disp > -900) & (disp < 10000)
        disp[~valid_mask] = 0.0

        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        if self.augmentor is not None:
            img1, img2, flow, valid_mask = self.augmentor(img1, img2, flow, valid_mask)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid_mask)
        else:
            valid = (flow[0].abs() < 1024) & (flow[1].abs() < 1024)

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW] * 2 + [padH] * 2)
            img2 = F.pad(img2, [padW] * 2 + [padH] * 2)

        flow = flow[:1]

        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()


class WHUStereo(StereoDataset):
    def __init__(self, aug_params=None, root='/home/roy/projects/YAOWEI/data/WHU_stereo_dataset', split='train'):
        super(WHUStereo, self).__init__(aug_params, sparse=True)

        self.root = root
        self.split = split
        self._read_data()

    def _read_data(self):
        split_folder = self.split
        if split_folder == 'validation':
            split_folder = 'val'

        left_dir = os.path.join(self.root, split_folder, 'left')
        right_dir = os.path.join(self.root, split_folder, 'right')
        disp_dir = os.path.join(self.root, split_folder, 'disp')

        if not os.path.exists(left_dir):
            logging.warning(f"Warning: {left_dir} not found, cannot load WHU dataset for split '{self.split}'")
            logging.info(f"WHUStereo ({self.split}) loaded 0 samples.")
            return

        self.image_list = []
        self.disparity_list = []

        left_files = sorted(glob(os.path.join(left_dir, '*_left_*.tiff')))

        for left_path in left_files:
            right_path = left_path.replace('_left_', '_right_').replace(left_dir, right_dir)
            disp_path = left_path.replace('_left_', '_disparity_').replace(left_dir, disp_dir)

            if os.path.exists(left_path) and os.path.exists(right_path) and os.path.exists(disp_path):
                self.image_list.append([left_path, right_path])
                self.disparity_list.append(disp_path)

        logging.info(f"WHUStereo ({self.split}) loaded {len(self.image_list)} samples from physical split folder.")

    def __getitem__(self, index):
        index = index % len(self.image_list)
        disp_path = self.disparity_list[index]
        img1_path, img2_path = self.image_list[index]

        # 1. 强制使用 rasterio 安全读取卫星 tiff 文件
        with rasterio.open(img1_path) as src:
            img1_read = src.read().transpose(1, 2, 0)
        with rasterio.open(img2_path) as src:
            img2_read = src.read().transpose(1, 2, 0)
        with rasterio.open(disp_path) as src:
            disp = src.read(1)  # 视差图只取第一个通道

        # 2. 防御性通道对齐：确保 RGB 三通道
        if img1_read.shape[-1] >= 3:
            img1 = img1_read[..., :3]
            img2 = img2_read[..., :3]
        else:
            # 如果是单波段灰度图，复制为三通道
            img1 = np.tile(img1_read[..., :1], (1, 1, 3))
            img2 = np.tile(img2_read[..., :1], (1, 1, 3))

        img1 = np.array(img1).astype(np.float32)
        img2 = np.array(img2).astype(np.float32)

        # 3. 卫星影像动态范围归一化 (防止 16-bit 像素溢出)
        if img1.max() > 255:
            img1 = (img1 / img1.max()) * 255.0
        if img2.max() > 255:
            img2 = (img2 / img2.max()) * 255.0

        img1 = np.clip(img1, 0, 255).astype(np.uint8)
        img2 = np.clip(img2, 0, 255).astype(np.uint8)

        # 4. 解析视差图
        disp = np.array(disp).astype(np.float32)

        # 5. 生成有效掩膜与 flow 张量
        valid = (disp > 0) & (disp < 10000)
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        # 6. 数据增强与张量化
        if self.augmentor is not None:
            img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1024) & (flow[1].abs() < 1024)

        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW] * 2 + [padH] * 2)
            img2 = F.pad(img2, [padW] * 2 + [padH] * 2)

        flow = flow[:1]
        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()

def fetch_dataloader(args):
    """ Create the data loader for satellite stereo datasets (DFC2019 / WHU) """

    aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1],
                  'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    train_dataset = None
    
    if args.train_datasets == 'dfc2019':
        new_dataset = DFC2019(aug_params, root='/home/roy/projects/YAOWEI/data/dfc2019-big', split='train')
        logging.info(f"Adding {len(new_dataset)} samples from DFC2019")
    elif 'whu' in args.train_datasets:
        aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0],
                      'max_scale': args.spatial_scale[1], 'do_flip': False, 'yjitter': not args.noyjitter}
        new_dataset = WHUStereo(aug_params, root='/home/roy/projects/YAOWEI/data/WHU_stereo_dataset', split='train')
        logging.info(f"Adding {len(new_dataset)} samples from WHU Stereo")
    else:
        raise ValueError(f"Unknown dataset: {args.train_datasets}. Only 'whu' and 'dfc2019' are supported.")

    train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset
    
    shuffle_data = True
    
    exp_mode = getattr(args, 'exp_mode', 'full') 
    
    if exp_mode == 'overfit':
        train_dataset.image_list = train_dataset.image_list[:args.batch_size]
        train_dataset.disparity_list = train_dataset.disparity_list[:args.batch_size]
        train_dataset.augmentor = None
        shuffle_data = False
        logging.warning("[OVERFIT MODE ACTIVATED] Dataset truncated to 1 batch. Augmentation OFF. Shuffle OFF.")
        
    elif exp_mode == 'fast':
        MINI_RATIO = 0.1
        total_samples = len(train_dataset.image_list)
        mini_samples = int(total_samples * MINI_RATIO)
        
        rng = np.random.RandomState(42)
        indices = rng.permutation(total_samples)[:mini_samples]
        
        train_dataset.image_list = [train_dataset.image_list[i] for i in indices]
        train_dataset.disparity_list = [train_dataset.disparity_list[i] for i in indices]
        logging.warning(f"[FAST COMPARE MODE] Dataset shrunk to {MINI_RATIO*100}%: {mini_samples} samples.")
        
    else:
        logging.info(f"[FULL MODE] Using complete dataset.")

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   pin_memory=True, shuffle=shuffle_data, num_workers=8, drop_last=True)

    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader
