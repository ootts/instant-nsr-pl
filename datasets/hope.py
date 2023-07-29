import glob
import os
import os.path as osp

import imageio
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from wis3d import Wis3D
from torch.utils.data import Dataset, DataLoader, IterableDataset

import datasets
from models.ray_utils import get_ray_directions
from utils.misc import get_rank


class HopeDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        self.has_mask = True
        self.apply_mask = True

        K = np.loadtxt(osp.join(self.config.root_dir, "K.txt"))
        camera_poses = np.load(osp.join(self.config.root_dir, "camera_poses.npy"))
        W, H = 800, 800

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

        self.directions = \
            get_ray_directions(self.w, self.h, K[0, 0], K[1, 1],
                               self.w // 2, self.h // 2).to(self.rank)  # (h, w, 3)
        self.all_c2w, self.all_images, self.all_fg_masks = [], [], []
        vis3d = Wis3D(
            xyz_pattern=('x', 'y', 'z'),
            out_folder="dbg",
            sequence_name="Hope_loader",
            # auto_increase=,
            # enable=,
        )
        image_paths = sorted(glob.glob(osp.join(self.config.root_dir, f"rgb/*png")))
        mask_paths = sorted(glob.glob(osp.join(self.config.root_dir, f"mask/*png")))
        blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        for i in range(len(image_paths)):
            c2w = camera_poses[i] @ np.linalg.inv(blender2opencv)
            c2w = torch.FloatTensor(c2w)[:3, :4]
            self.all_c2w.append(c2w)
            vis3d.add_camera_trajectory(torch.cat([c2w, torch.tensor([[0, 0, 0, 1]])], dim=0)[None])
            img = Image.open(image_paths[i])
            img = img.resize(self.img_wh, Image.BICUBIC)
            img = TF.to_tensor(img).permute(1, 2, 0)  # (4, h, w) => (h, w, 4)
            seg_labels = imageio.imread_v2(mask_paths[i])
            mask = seg_labels == self.config.hope_mask_index
            self.all_fg_masks.append(torch.from_numpy(mask).float())  # (h, w)
            self.all_images.append(img[..., :3])
        self.all_c2w, self.all_images, self.all_fg_masks = \
            torch.stack(self.all_c2w, dim=0).float().to(self.rank), \
            torch.stack(self.all_images, dim=0).float().to(self.rank), \
            torch.stack(self.all_fg_masks, dim=0).float().to(self.rank)


class HopeDataset(Dataset, HopeDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        return {
            'index': index,
        }


class HopeIterableDataset(IterableDataset, HopeDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('hope')
class HopeDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = HopeIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = HopeDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = HopeDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = HopeDataset(self.config, self.config.train_split)

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
