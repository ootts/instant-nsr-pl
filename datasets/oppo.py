import cv2
import argparse
import os
import json
import sys

import numpy as np
from PIL import Image

import torch
from wis3d import Wis3D
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions
from utils.misc import get_rank


class OppoDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        self.has_mask = True
        self.apply_mask = True

        # with open(os.path.join(self.config.root_dir, f"transforms_{self.split}.new.json"), 'r') as f:
        with open(os.path.join(self.config.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)['frames']

        W, H = 2656, 3984

        if 'img_wh' in self.config:
            w, h = self.config.img_wh
            assert round(W / w * h) == H
        elif 'img_downscale' in self.config:
            w, h = W // self.config.img_downscale, H // self.config.img_downscale
        else:
            raise KeyError("Either img_wh or img_downscale should be specified.")

        self.w, self.h = w, h
        self.img_wh = (self.w, self.h)

        self.near, self.far = self.config.near_plane, self.config.far_plane

        # self.focal = 0.5 * w / math.tan(0.5 * meta['camera_angle_x'])  # scaled focal length

        # ray directions for all pixels, same for all images (same H, W, focal)
        # self.directions = \
        #     get_ray_directions(self.w, self.h, self.focal, self.focal, self.w // 2, self.h // 2).to(
        #         self.rank)  # (h, w, 3)
        self.directions = []
        self.all_c2w, self.all_images, self.all_fg_masks = [], [], []
        wis3d = Wis3D(
            xyz_pattern=('x', 'y', 'z'),
            out_folder="dbg",
            sequence_name="Oppo_loader",
            # auto_increase=,
            # enable=,
        )
        blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        for k, v in self.meta.items():
            imgid = v['file_path'].split('/')[-1]
            focal = 0.5 * v['calib_imgw'] / np.tan(0.5 * v['camera_angle_x'])  # original focal length
            focal = focal * self.w / v['calib_imgw']
            directions = \
                get_ray_directions(self.w, self.h, focal, focal, self.w // 2, self.h // 2).to(self.rank)  # (h, w, 3)
            self.directions.append(directions)
            c2w = np.array(v['transform_matrix'])
            c2w = c2w @ np.linalg.inv(blender2opencv)
            c2w = torch.FloatTensor(c2w)[:3, :4]
            self.all_c2w.append(c2w)
            wis3d.add_camera_trajectory(torch.cat([c2w, torch.tensor([[0, 0, 0, 1]])], dim=0)[None])
            img_path = os.path.join(self.config.root_dir, f"../Lights/013/raw_undistorted/{imgid}.JPG")
            img = Image.open(img_path)
            img = img.resize(self.img_wh, Image.BICUBIC)
            img = TF.to_tensor(img).permute(1, 2, 0)  # (4, h, w) => (h, w, 4)
            if self.split == 'train':
                mask_path = os.path.join(self.config.root_dir, f"com_masks/{imgid}.png")
            else:
                mask_path = os.path.join(self.config.root_dir, f"obj_masks/{imgid}.png")
            mask = cv2.imread(mask_path, 2) > 0
            mask = cv2.resize(mask.astype(np.uint8), self.img_wh, interpolation=cv2.INTER_NEAREST) > 0
            self.all_fg_masks.append(torch.from_numpy(mask).float())  # (h, w)
            self.all_images.append(img[..., :3])
        self.directions = torch.stack(self.directions, dim=0)
        self.all_c2w, self.all_images, self.all_fg_masks = \
            torch.stack(self.all_c2w, dim=0).float().to(self.rank), \
            torch.stack(self.all_images, dim=0).float().to(self.rank), \
            torch.stack(self.all_fg_masks, dim=0).float().to(self.rank)


class OppoDataset(Dataset, OppoDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        return {
            'index': index
        }


class OppoIterableDataset(IterableDataset, OppoDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('oppo')
class OppoDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = OppoIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = OppoDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = OppoDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = OppoDataset(self.config, self.config.train_split)

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset,
            num_workers=os.cpu_count(),
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )

    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)
