import glob
import os.path as osp
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


class Oneposev2DatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        self.has_mask = True
        self.apply_mask = True
        json_dir = osp.join(self.config.root_dir, "sfm_output/outputs_softmax_loftr_loftr", config.scene)
        with open(os.path.join(json_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        W, H = 512, 512

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
        vis3d = Wis3D(
            xyz_pattern=('x', 'y', 'z'),
            out_folder="dbg",
            sequence_name="Onepose_loader",
            # auto_increase=,
            # enable=,
        )
        blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        image_dir = sorted(glob.glob(osp.join(self.config.root_dir, f"lowtexture_test_data/{config.scene}/*/color")))[self.config.subdir_index]
        self.subseq = image_dir.split('/')[-2]
        mask_dir = osp.join(image_dir, "../GSA")
        for k, v in self.meta.items():
            imgid = v['imgid']
            K = np.loadtxt(osp.join(image_dir, f"../intrin_ba/{imgid}.txt"))
            directions = \
                get_ray_directions(self.w, self.h, K[0, 0], K[1, 1], K[0, 2], K[1, 2]).to(self.rank)  # (h, w, 3)
            self.directions.append(directions)
            c2w = np.loadtxt(osp.join(image_dir, f"../poses_ba/{imgid}.txt"))
            c2w = np.linalg.inv(c2w)
            c2w = c2w @ np.linalg.inv(blender2opencv)
            c2w = torch.FloatTensor(c2w)[:3, :4]
            self.all_c2w.append(c2w)
            vis3d.add_camera_trajectory(torch.cat([c2w, torch.tensor([[0, 0, 0, 1]])], dim=0)[None])
            img_path = osp.join(image_dir, f"{imgid}.png")
            img = Image.open(img_path)
            img = img.resize(self.img_wh, Image.BICUBIC)
            img = TF.to_tensor(img).permute(1, 2, 0)  # (4, h, w) => (h, w, 4)
            mask_path = os.path.join(mask_dir, f"mask_{imgid}.png")
            mask = cv2.imread(mask_path, 2) > 0
            self.all_fg_masks.append(torch.from_numpy(mask).float())  # (h, w)
            self.all_images.append(img[..., :3])
        self.directions = torch.stack(self.directions, dim=0)
        self.all_c2w, self.all_images, self.all_fg_masks = \
            torch.stack(self.all_c2w, dim=0).float().to(self.rank), \
            torch.stack(self.all_images, dim=0).float().to(self.rank), \
            torch.stack(self.all_fg_masks, dim=0).float().to(self.rank)


class Oneposev2Dataset(Dataset, Oneposev2DatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        return {
            'index': index,
            "imgid": list(self.meta.keys())[index]
        }


class Oneposev2IterableDataset(IterableDataset, Oneposev2DatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('oneposev2')
class Oneposev2DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = Oneposev2IterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = Oneposev2Dataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = Oneposev2Dataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = Oneposev2Dataset(self.config, self.config.train_split)

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset,
            num_workers=0 if 'PYCHARM_HOSTED' in os.environ else os.cpu_count(),
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
