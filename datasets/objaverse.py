import tqdm
import os.path as osp
import os
import json
import math
import numpy as np
from PIL import Image

import torch
from wis3d import Wis3D
from pytorch3d.implicitron.dataset.load_blender import pose_spherical
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl

import datasets
from models.ray_utils import get_ray_directions
from utils.misc import get_rank


class ObjaverseDatasetBase():
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        self.has_mask = True
        self.apply_mask = True

        with open(
                "/home/linghao/Datasets/objaverse-processed/zero12345_img/eval/Zero123/zero12345_2stage_8_pose.json") as f:
            meta = json.load(f)
        c2ws = np.array(list(meta['c2ws'].values()))
        if self.split == "test":
            render_path = np.stack(
                [pose_spherical(angle, -10.0, 1.5) for angle in np.linspace(-180, 180, 80 + 1)[:-1]], 0)
            c2ws = np.stack([rp for rp in render_path])

            elev = self.config.objaverse_elevation
            azimuth = self.config.objaverse_azimuth
            elevations = np.radians([elev, elev - 10])  # 45~120
            azimuths = np.radians([azimuth, azimuth - 10])
            render_poses = calc_pose(elevations, azimuths, len(azimuths), radius=1.3, device="cpu")
            # render_pose = render_poses[0:1]
            c2ws = np.stack([rp for rp in render_poses.numpy()])

        if 'w' in meta and 'h' in meta:
            W, H = int(meta['w']), int(meta['h'])
        else:
            W, H = 256, 256

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

        self.focal = 280

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.w, self.h, self.focal, self.focal, self.w // 2, self.h // 2).to(
                self.rank)  # (h, w, 3)

        self.all_c2w, self.all_images, self.all_fg_masks = [], [], []
        img_files = []
        for i in range(8):
            img_files.append(osp.join(self.config.root_dir, f"stage1_8/{i}.png"))
        for i in range(8):
            for j in range(4):
                img_files.append(osp.join(self.config.root_dir, f"stage2_8/{i}_{j}.png"))
        if self.split != "test":
            idxs = self.config.objaverse_indices
        else:
            idxs = range(len(c2ws))
        vis3d = Vis3D(
            xyz_pattern=('x', 'y', 'z'),
            out_folder="dbg",
            sequence="objaverse_loader",
            # auto_increase=,
            # enable=,
        )
        for i in tqdm.tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):  # img_list:#
            # for i, frame in enumerate(meta['frames']):
            #     pose = c2ws[i]
            c2w = torch.from_numpy(c2ws[i][:3, :4])
            self.all_c2w.append(c2w)
            vis3d.add_camera_trajectory(torch.cat([c2w, torch.tensor([[0, 0, 0, 1]])], dim=0)[None])

            # img_path = os.path.join(self.config.root_dir, f"{frame['file_path']}.png")
            if self.split != "test":
                img_path = img_files[i]
                img = Image.open(img_path)
                img = img.resize(self.img_wh, Image.BICUBIC)
            else:
                img = Image.new("RGB", self.img_wh, (0, 0, 0))
            img = TF.to_tensor(img).permute(1, 2, 0)  # (4, h, w) => (h, w, 4)

            self.all_fg_masks.append(1 - (img > 0.97).all(dim=-1).float())  # (h, w)
            self.all_images.append(img)

        self.all_c2w, self.all_images, self.all_fg_masks = \
            torch.stack(self.all_c2w, dim=0).float().to(self.rank), \
            torch.stack(self.all_images, dim=0).float().to(self.rank), \
            torch.stack(self.all_fg_masks, dim=0).float().to(self.rank)


class ObjaverseDataset(Dataset, ObjaverseDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        return {
            'index': index
        }


class ObjaverseIterableDataset(IterableDataset, ObjaverseDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('objaverse')
class ObjaverseDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = ObjaverseIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = ObjaverseDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = ObjaverseDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = ObjaverseDataset(self.config, self.config.train_split)

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


def calc_pose(elevations, azimuths, size, radius=1.2, device="cuda"):
    import torch
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    # device = torch.device('cuda')
    thetas = torch.FloatTensor(azimuths).to(device)
    phis = torch.FloatTensor(elevations).to(device)

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        -radius * torch.cos(thetas) * torch.sin(phis),
        radius * torch.cos(phis),
    ], dim=-1)  # [B, 3]

    # lookat
    forward_vector = normalize(centers).squeeze(0)
    up_vector = torch.FloatTensor([0, 0, 1]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = normalize(torch.cross(up_vector, forward_vector, dim=-1))
    if right_vector.pow(2).sum() < 0.01:
        right_vector = torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    up_vector = normalize(torch.cross(forward_vector, right_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers
    return poses
