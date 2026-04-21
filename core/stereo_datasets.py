import numpy as np
import rasterio
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import re
import copy
import math
import random
from pathlib import Path
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

        # grayscale images
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
        # 重写 __getitem__ 以支持 Rasterio 和 特殊的归一化逻辑
        index = index % len(self.image_list)
        disp_path = self.disparity_list[index]
        img1_path, img2_path = self.image_list[index]

        # --- 读取影像 (使用 Rasterio) ---
        with rasterio.open(img1_path) as src:
            # 读取前3个波段 (C, H, W) -> (H, W, C)
            img1 = src.read([1, 2, 3]).transpose(1, 2, 0)
        with rasterio.open(img2_path) as src:
            img2 = src.read([1, 2, 3]).transpose(1, 2, 0)
        with rasterio.open(disp_path) as src:
            # 视差通常是单波段
            disp = src.read(1)

        # --- 数据预处理与归一化 ---
        # 卫星影像可能是 16-bit 或 float，IGEV++ 的 Augmentor 需要 uint8
        img1 = np.array(img1).astype(np.float32)
        img2 = np.array(img2).astype(np.float32)

        # 简单的最大值归一化
        if img1.max() > 255:
            img1 = (img1 / img1.max()) * 255.0
        if img2.max() > 255:
            img2 = (img2 / img2.max()) * 255.0

        img1 = np.clip(img1, 0, 255).astype(np.uint8)
        img2 = np.clip(img2, 0, 255).astype(np.uint8)

        # --- 处理视差与 Mask ---
        disp = np.array(disp).astype(np.float32)

        # DFC2019 特有的 nodata 处理
        valid_mask = (disp > -900) & (disp < 10000)
        disp[~valid_mask] = 0.0  # 无效区域视差置0

        # 构建 Flow 格式 (IGEV++ 内部逻辑)
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        # --- 数据增强 (IGEV++ Augmentor) ---
        if self.augmentor is not None:
            # 这里的 valid 会被 augmentor 进行裁剪和变换
            img1, img2, flow, valid_mask = self.augmentor(img1, img2, flow, valid_mask)

        # --- 转为 Tensor ---
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid_mask)
        else:
            valid = (flow[0].abs() < 1024) & (flow[1].abs() < 1024)

        # IGEV++ 有时需要 padding
        if self.img_pad is not None:
            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW] * 2 + [padH] * 2)
            img2 = F.pad(img2, [padW] * 2 + [padH] * 2)

        flow = flow[:1]  # 只取第一通道 (Disparity)

        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()


class WHUStereo(StereoDataset):
    def __init__(self, aug_params=None, root='/home/roy/projects/YAOWEI/data/WHU_stereo_dataset', split='train'):
        # 【重要】必须设置 sparse=True，否则会报 augmentor 参数错误
        super(WHUStereo, self).__init__(aug_params, sparse=True)

        self.root = root
        self.split = split
        self._read_data()

    def _read_data(self):
        # 逻辑：如果是 train/val，去读 'train' 文件夹；如果是 test，去读 'test' 文件夹
        if self.split in ['train', 'validation']:
            sub_folder = 'train'
        else:
            sub_folder = 'test'

        root_dir = os.path.join(self.root, sub_folder)

        if not os.path.exists(root_dir):
            logging.warning(f"Warning: {root_dir} not found, trying root directly...")
            root_dir = self.root

        self.image_list = []
        self.disparity_list = []

        all_left_files = []
        all_right_files = []
        all_disp_files = []

        # 遍历场景文件夹
        scenes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        for scene in scenes:
            left_dir = os.path.join(root_dir, scene, 'Left')
            right_dir = os.path.join(root_dir, scene, 'Right')
            disp_dir = os.path.join(root_dir, scene, 'Disparity')  # 注意大小写

            if not os.path.exists(left_dir): continue

            imgs = sorted([img for img in os.listdir(left_dir) if img.endswith('.png')])

            for img in imgs:
                l_path = os.path.join(left_dir, img)
                r_path = os.path.join(right_dir, img)
                d_path = os.path.join(disp_dir, img)

                if os.path.exists(l_path) and os.path.exists(r_path) and os.path.exists(d_path):
                    all_left_files.append(l_path)
                    all_right_files.append(r_path)
                    all_disp_files.append(d_path)

        # --- 核心：9:1 切分 (仅针对 train 文件夹的数据) ---
        if self.split in ['train', 'validation']:
            total_len = len(all_left_files)
            split_idx = int(total_len * 0.9)  # 90% 处切分

            if self.split == 'train':
                # 前 90%
                self.image_list = [list(x) for x in zip(all_left_files[:split_idx], all_right_files[:split_idx])]
                self.disparity_list = all_disp_files[:split_idx]
            elif self.split == 'validation':
                # 后 10%
                self.image_list = [list(x) for x in zip(all_left_files[split_idx:], all_right_files[split_idx:])]
                self.disparity_list = all_disp_files[split_idx:]
        else:
            # Test 模式，不切分，全拿
            self.image_list = [list(x) for x in zip(all_left_files, all_right_files)]
            self.disparity_list = all_disp_files

        logging.info(f"WHUStereo ({self.split}) loaded {len(self.image_list)} samples.")

    def __getitem__(self, index):
        index = index % len(self.image_list)
        disp_path = self.disparity_list[index]
        img1_path, img2_path = self.image_list[index]

        # 读取
        img1 = frame_utils.read_gen(img1_path)
        img2 = frame_utils.read_gen(img2_path)
        # WHU 视差是 uint16, 除以256转为 float
        disp = np.array(frame_utils.read_gen(disp_path)).astype(np.float32) / 256.0

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        disp = np.array(disp).astype(np.float32)

        # 构建 valid mask
        valid = (disp > 0) & (disp < 10000)
        # 构建 Flow (H,W,2) 传给 Augmentor
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        # 调用 Augmentor (必须传入 flow 而非 disp)
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


class SceneFlowDatasets(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/sceneflow/', dstype='frames_finalpass',
                 things_test=False):
        super(SceneFlowDatasets, self).__init__(aug_params)
        self.root = root
        self.dstype = dstype

        if things_test:
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            self._add_monkaa("TRAIN")
            self._add_driving("TRAIN")

    def _add_things(self, split='TRAIN'):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        # root = osp.join(self.root, 'FlyingThings3D')
        root = self.root
        left_images = sorted(glob(osp.join(root, self.dstype, split, '*/*/left/*.png')))
        right_images = [im.replace('left', 'right') for im in left_images]
        disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images]

        # Choose a random subset of 400 images for validation
        state = np.random.get_state()
        np.random.seed(1000)
        # val_idxs = set(np.random.permutation(len(left_images))[:100])
        val_idxs = set(np.random.permutation(len(left_images)))
        np.random.set_state(state)

        for idx, (img1, img2, disp) in enumerate(zip(left_images, right_images, disparity_images)):
            if (split == 'TEST' and idx in val_idxs) or split == 'TRAIN':
                self.image_list += [[img1, img2]]
                self.disparity_list += [disp]
        logging.info(f"Added {len(self.disparity_list) - original_length} from FlyingThings {self.dstype}")

    def _add_monkaa(self, split="TRAIN"):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = self.root
        left_images = sorted(glob(osp.join(root, self.dstype, split, '*/left/*.png')))
        right_images = [image_file.replace('left', 'right') for image_file in left_images]
        disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")

    def _add_driving(self, split="TRAIN"):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = self.root
        left_images = sorted(glob(osp.join(root, self.dstype, split, '*/*/*/left/*.png')))
        right_images = [image_file.replace('left', 'right') for image_file in left_images]
        disparity_images = [im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")


class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/eth3d', split='training'):
        super(ETH3D, self).__init__(aug_params, sparse=True)

        image1_list = sorted(glob(osp.join(root, f'two_view_{split}/*/im0.png')))
        image2_list = sorted(glob(osp.join(root, f'two_view_{split}/*/im1.png')))
        disp_list = sorted(
            glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm'))) if split == 'training' else [osp.join(root,
                                                                                                              'two_view_training_gt/playground_1l/disp0GT.pfm')] * len(
            image1_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class SintelStereo(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/sintelstereo'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispSintelStereo)

        image1_list = sorted(glob(osp.join(root, 'training/*_left/*/frame_*.png')))
        image2_list = sorted(glob(osp.join(root, 'training/*_right/*/frame_*.png')))
        disp_list = sorted(glob(osp.join(root, 'training/disparities/*/frame_*.png'))) * 2

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            assert img1.split('/')[-2:] == disp.split('/')[-2:]
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class FallingThings(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/fallingthings'):
        super().__init__(aug_params, reader=frame_utils.readDispFallingThings)
        assert os.path.exists(root)

        image1_list = sorted(glob(root + '/*/*/*left.jpg'))
        image2_list = sorted(glob(root + '/*/*/*right.jpg'))
        disp_list = sorted(glob(root + '/*/*/*left.depth.png'))

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class TartanAir(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/tartanair'):
        super().__init__(aug_params, reader=frame_utils.readDispTartanAir)
        assert os.path.exists(root)

        image1_list = sorted(glob(osp.join(root, '*/*/*/*/image_left/*.png')))
        image2_list = sorted(glob(osp.join(root, '*/*/*/*/image_right/*.png')))
        disp_list = sorted(glob(osp.join(root, '*/*/*/*/depth_left/*.npy')))

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class CREStereoDataset(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/crestereo'):
        super(CREStereoDataset, self).__init__(aug_params, reader=frame_utils.readDispCREStereo)
        assert os.path.exists(root)

        image1_list = sorted(glob(os.path.join(root, '*/*_left.jpg')))
        image2_list = sorted(glob(os.path.join(root, '*/*_right.jpg')))
        disp_list = sorted(glob(os.path.join(root, '*/*_left.disp.png')))

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class CARLA(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/carla-highres'):
        super(CARLA, self).__init__(aug_params)
        assert os.path.exists(root)

        image1_list = sorted(glob(root + '/trainingF/*/im0.png'))
        image2_list = sorted(glob(root + '/trainingF/*/im1.png'))
        disp_list = sorted(glob(root + '/trainingF/*/disp0GT.pfm'))

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class InStereo2K(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/instereo2k'):
        super(InStereo2K, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispInStereo2K)
        assert os.path.exists(root)

        image1_list = sorted(glob(root + '/train/*/*/left.png') + glob(root + '/test/*/left.png'))
        image2_list = sorted(glob(root + '/train/*/*/right.png') + glob(root + '/test/*/right.png'))
        disp_list = sorted(glob(root + '/train/*/*/left_disp.png') + glob(root + '/test/*/left_disp.png'))

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/kitti', image_set='training', year=2015):
        super(KITTI, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        assert os.path.exists(root)

        if year == 2012:
            root_12 = '/data/StereoDatasets/kitti/2012'
            image1_list = sorted(glob(os.path.join(root_12, image_set, 'colored_0/*_10.png')))
            image2_list = sorted(glob(os.path.join(root_12, image_set, 'colored_1/*_10.png')))
            disp_list = sorted(
                glob(os.path.join(root_12, 'training', 'disp_occ/*_10.png'))) if image_set == 'training' else [osp.join(
                root, 'training/disp_occ/000085_10.png')] * len(image1_list)

        if year == 2015:
            root_15 = '/data/StereoDatasets/kitti/2015'
            image1_list = sorted(glob(os.path.join(root_15, image_set, 'image_2/*_10.png')))
            image2_list = sorted(glob(os.path.join(root_15, image_set, 'image_3/*_10.png')))
            disp_list = sorted(glob(
                os.path.join(root_15, 'training', 'disp_occ_0/*_10.png'))) if image_set == 'training' else [osp.join(
                root, 'training/disp_occ_0/000085_10.png')] * len(image1_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class VKITTI2(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/vkitti2'):
        super(VKITTI2, self).__init__(aug_params, reader=frame_utils.readDispVKITTI2)
        assert os.path.exists(root)

        image1_list = sorted(glob(os.path.join(root, 'Scene*/*/frames/rgb/Camera_0/rgb*.jpg')))
        image2_list = sorted(glob(os.path.join(root, 'Scene*/*/frames/rgb/Camera_1/rgb*.jpg')))
        disp_list = sorted(glob(os.path.join(root, 'Scene*/*/frames/depth/Camera_0/depth*.png')))

        assert len(image1_list) == len(image2_list) == len(disp_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoDatasets/middlebury', split='2014', resolution='F'):
        super(Middlebury, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispMiddlebury)
        assert os.path.exists(root)
        assert split in ["2005", "2006", "2014", "2021", "MiddEval3"]
        if split == "2005":
            scenes = list((Path(root) / "2005").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "view1.png"), str(scene / "view5.png")]]
                self.disparity_list += [str(scene / "disp1.png")]
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:
                        self.image_list += [[str(scene / f"Illum{illum}/Exp{exp}/view1.png"),
                                             str(scene / f"Illum{illum}/Exp{exp}/view5.png")]]
                        self.disparity_list += [str(scene / "disp1.png")]
        elif split == "2006":
            scenes = list((Path(root) / "2006").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "view1.png"), str(scene / "view5.png")]]
                self.disparity_list += [str(scene / "disp1.png")]
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:
                        self.image_list += [[str(scene / f"Illum{illum}/Exp{exp}/view1.png"),
                                             str(scene / f"Illum{illum}/Exp{exp}/view5.png")]]
                        self.disparity_list += [str(scene / "disp1.png")]
        elif split == "2014":
            scenes = list((Path(root) / "2014").glob("*"))
            for scene in scenes:
                for s in ["E", "L", ""]:
                    self.image_list += [[str(scene / "im0.png"), str(scene / f"im1{s}.png")]]
                    self.disparity_list += [str(scene / "disp0.pfm")]
        elif split == "2021":
            scenes = list((Path(root) / "2021/data").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "im0.png"), str(scene / "im1.png")]]
                self.disparity_list += [str(scene / "disp0.pfm")]
                for s in ["0", "1", "2", "3"]:
                    if os.path.exists(str(scene / f"ambient/L0/im0e{s}.png")):
                        self.image_list += [
                            [str(scene / f"ambient/L0/im0e{s}.png"), str(scene / f"ambient/L0/im1e{s}.png")]]
                        self.disparity_list += [str(scene / "disp0.pfm")]
        else:
            image1_list = sorted(glob(os.path.join(root, "MiddEval3", f'training{resolution}', '*/im0.png')))
            image2_list = sorted(glob(os.path.join(root, "MiddEval3", f'training{resolution}', '*/im1.png')))
            disp_list = sorted(glob(os.path.join(root, "MiddEval3", f'training{resolution}', '*/disp0GT.pfm')))
            assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, split]
            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                self.image_list += [[img1, img2]]
                self.disparity_list += [disp]


def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1],
                  'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    train_dataset = None
    # for dataset_name in args.train_datasets:
    if args.train_datasets == 'dfc2019':
        new_dataset = DFC2019(aug_params, root='/home/roy/projects/YAOWEI/data/dfc2019-big', split='train')
        logging.info(f"Adding {len(new_dataset)} samples from DFC2019")
    elif 'whu' in args.train_datasets:
        aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0],
                      'max_scale': args.spatial_scale[1], 'do_flip': False, 'yjitter': not args.noyjitter}
        new_dataset = WHUStereo(aug_params, root='/home/roy/projects/YAOWEI/data/WHU_stereo_dataset', split='train')
        logging.info(f"Adding {len(new_dataset)} samples from WHU Stereo")
    elif args.train_datasets == 'sceneflow':
        aug_params['spatial_scale'] = False
        new_dataset = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
        logging.info(f"Adding {len(new_dataset)} samples from SceneFlow")
    elif args.train_datasets == 'vkitti2':
        new_dataset = VKITTI2(aug_params)
        logging.info(f"Adding {len(new_dataset)} samples from VKITTI2")
    elif args.train_datasets == 'kitti':
        kitti12 = KITTI(aug_params, year=2012)
        logging.info(f"Adding {len(kitti12)} samples from KITTI 2012")
        kitti15 = KITTI(aug_params, year=2015)
        logging.info(f"Adding {len(kitti15)} samples from KITTI 2015")
        new_dataset = kitti12 + kitti15
        logging.info(f"Adding {len(new_dataset)} samples from KITTI")
    elif args.train_datasets == 'eth3d_train':
        tartanair = TartanAir(aug_params)
        logging.info(f"Adding {len(tartanair)} samples from Tartain Air")
        sceneflow = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
        logging.info(f"Adding {len(sceneflow)} samples from SceneFlow")
        sintel = SintelStereo(aug_params)
        logging.info(f"Adding {len(sintel)} samples from Sintel Stereo")
        crestereo = CREStereoDataset(aug_params)
        logging.info(f"Adding {len(crestereo)} samples from CREStereo Dataset")
        eth3d = ETH3D(aug_params)
        logging.info(f"Adding {len(eth3d)} samples from ETH3D")
        instereo2k = InStereo2K(aug_params)
        logging.info(f"Adding {len(instereo2k)} samples from InStereo2K")
        new_dataset = tartanair + sceneflow + sintel * 50 + eth3d * 1000 + instereo2k * 100 + crestereo * 2
        logging.info(f"Adding {len(new_dataset)} samples from ETH3D Mixture Dataset")
    elif args.train_datasets == 'eth3d_finetune':
        crestereo = CREStereoDataset(aug_params)
        logging.info(f"Adding {len(crestereo)} samples from CREStereo Dataset")
        eth3d = ETH3D(aug_params)
        logging.info(f"Adding {len(eth3d)} samples from ETH3D")
        instereo2k = InStereo2K(aug_params)
        logging.info(f"Adding {len(instereo2k)} samples from InStereo2K")
        new_dataset = eth3d * 1000 + instereo2k * 10 + crestereo
        logging.info(f"Adding {len(new_dataset)} samples from ETH3D Mixture Dataset")
    elif args.train_datasets == 'middlebury_train':
        tartanair = TartanAir(aug_params)
        logging.info(f"Adding {len(tartanair)} samples from Tartain Air")
        sceneflow = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
        logging.info(f"Adding {len(sceneflow)} samples from SceneFlow")
        fallingthings = FallingThings(aug_params)
        logging.info(f"Adding {len(fallingthings)} samples from FallingThings")
        carla = CARLA(aug_params)
        logging.info(f"Adding {len(carla)} samples from CARLA")
        crestereo = CREStereoDataset(aug_params)
        logging.info(f"Adding {len(crestereo)} samples from CREStereo Dataset")
        instereo2k = InStereo2K(aug_params)
        logging.info(f"Adding {len(instereo2k)} samples from InStereo2K")
        mb2005 = Middlebury(aug_params, split='2005')
        logging.info(f"Adding {len(mb2005)} samples from Middlebury 2005")
        mb2006 = Middlebury(aug_params, split='2006')
        logging.info(f"Adding {len(mb2006)} samples from Middlebury 2006")
        mb2014 = Middlebury(aug_params, split='2014')
        logging.info(f"Adding {len(mb2014)} samples from Middlebury 2014")
        mb2021 = Middlebury(aug_params, split='2021')
        logging.info(f"Adding {len(mb2021)} samples from Middlebury 2021")
        mbeval3 = Middlebury(aug_params, split='MiddEval3', resolution='H')
        logging.info(f"Adding {len(mbeval3)} samples from Middlebury Eval3")
        new_dataset = tartanair + sceneflow + fallingthings + instereo2k * 50 + carla * 50 + crestereo + mb2005 * 200 + mb2006 * 200 + mb2014 * 200 + mb2021 * 200 + mbeval3 * 200
        logging.info(f"Adding {len(new_dataset)} samples from Middlebury Mixture Dataset")
    elif args.train_datasets == 'middlebury_finetune':
        crestereo = CREStereoDataset(aug_params)
        logging.info(f"Adding {len(crestereo)} samples from CREStereo Dataset")
        instereo2k = InStereo2K(aug_params)
        logging.info(f"Adding {len(instereo2k)} samples from InStereo2K")
        carla = CARLA(aug_params)
        logging.info(f"Adding {len(carla)} samples from CARLA")
        mb2005 = Middlebury(aug_params, split='2005')
        logging.info(f"Adding {len(mb2005)} samples from Middlebury 2005")
        mb2006 = Middlebury(aug_params, split='2006')
        logging.info(f"Adding {len(mb2006)} samples from Middlebury 2006")
        mb2014 = Middlebury(aug_params, split='2014')
        logging.info(f"Adding {len(mb2014)} samples from Middlebury 2014")
        mb2021 = Middlebury(aug_params, split='2021')
        logging.info(f"Adding {len(mb2021)} samples from Middlebury 2021")
        mbeval3 = Middlebury(aug_params, split='MiddEval3', resolution='H')
        logging.info(f"Adding {len(mbeval3)} samples from Middlebury Eval3")
        mbeval3_f = Middlebury(aug_params, split='MiddEval3', resolution='F')
        logging.info(f"Adding {len(mbeval3)} samples from Middlebury Eval3")
        fallingthings = FallingThings(aug_params)
        logging.info(f"Adding {len(fallingthings)} samples from FallingThings")
        new_dataset = crestereo + instereo2k * 50 + carla * 50 + mb2005 * 200 + mb2006 * 200 + mb2014 * 200 + mb2021 * 200 + mbeval3 * 200 + mbeval3_f * 200 + fallingthings * 10
        logging.info(f"Adding {len(new_dataset)} samples from Middlebury Mixture Dataset")

    train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset
    # ==================== [Ours: 实验模式统一管理大脑] ====================
    # 默认开启 shuffle
    shuffle_data = True
    
    # 获取实验模式 (兼容没有传入该参数的旧代码)
    exp_mode = getattr(args, 'exp_mode', 'full') 
    
    if exp_mode == 'overfit':
        # 模式 1：极小样本过拟合 (查Bug专用)
        train_dataset.image_list = train_dataset.image_list[:args.batch_size]
        train_dataset.disparity_list = train_dataset.disparity_list[:args.batch_size]
        train_dataset.augmentor = None  # 必须关闭增强，确保每次输入绝对一致
        shuffle_data = False            # 关闭打乱
        logging.warning("🚀 [OVERFIT MODE ACTIVATED] Dataset truncated to 1 batch. Augmentation OFF. Shuffle OFF.")
        
    elif exp_mode == 'fast':
        # 模式 2：加速对比测试 (缩小数据集至 10%)
        MINI_RATIO = 0.1
        total_samples = len(train_dataset.image_list)
        mini_samples = int(total_samples * MINI_RATIO)
        
        # 固定随机种子抽样，保证 Baseline 和 改进模型 抽到完全相同的 10%
        rng = np.random.RandomState(42)
        indices = rng.permutation(total_samples)[:mini_samples]
        
        train_dataset.image_list = [train_dataset.image_list[i] for i in indices]
        train_dataset.disparity_list = [train_dataset.disparity_list[i] for i in indices]
        logging.warning(f"🚀 [FAST COMPARE MODE] Dataset shrunk to {MINI_RATIO*100}%: {mini_samples} samples.")
        
    else:
        # 模式 3：完整大炼丹 (原汁原味)
        logging.info(f"🚀 [FULL MODE] Using complete dataset.")
    # ========================================================================

    # 注意这里的 shuffle 参数已经被我们的控制大脑接管
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   pin_memory=True, shuffle=shuffle_data, num_workers=8, drop_last=True)

    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader

