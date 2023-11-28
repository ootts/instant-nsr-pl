import numpy as np
import imageio
import csv
import glob
import os.path as osp


def main():
    obj_names = [
        "obj_02_egg",
        "obj_04_stone",
        "obj_05_bird",
        "obj_17_wooden_box",
        "obj_26_pumpkin",
        "obj_29_fabric_toy",
        "obj_35_transparent_plastic_cup",
        "obj_36_sponge",
        "obj_42_food_banana",
        "obj_48_metal_bucket2"
    ]
    for obj_name in obj_names:
        res_dir = sorted(glob.glob(osp.expanduser(f"exp/nerf-oppo-r0.5-half-{obj_name}/@*")))[-1]
        res_dir = osp.join(res_dir, "save/it20000-test")
        files = sorted(glob.glob(osp.join(res_dir, "*png")))
        # files = sorted(glob.glob(f"/home/linghao/PycharmProjects/instant-nsr-pl/exp/oppo/neus-oppo-r0.5-half-{obj_name}//save/it20000-test/*.png", recursive=True))
        psnrs = []
        for file in files:
            img = imageio.imread_v2(file)
            n = img.shape[1] // 4
            gt = img[:, :n]
            pred = img[:, n:2 * n]
            mask = ~(gt == 255).all(axis=-1)
            pred[~mask] = 255
            mse = np.mean((pred / 255.0 - gt / 255.0) ** 2)
            psnr = -10 * np.log(mse) / np.log(10)
            psnrs.append(psnr)
            # gt = gt[mask]
            # pred = pred[mask]
            # print()
        # psnr = float(lines[-1].split(",")[-1])
        print(psnrs)
        print(obj_name, np.mean(psnrs))


if __name__ == '__main__':
    main()
