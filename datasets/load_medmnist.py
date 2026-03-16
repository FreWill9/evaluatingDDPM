import torch
import os
import numpy as np

import medmnist
from medmnist import PathMNIST, ChestMNIST, OrganAMNIST


def load_and_save(
    dataset_class,
    name: str,
    img_size: int,
    save_dir: str,
    max_samples: int,
):
    """
    Load a MedMNIST dataset (train + val + test splits), take up to max_samples,
    normalize images to [-1, 1], and save as a .pt file.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load all splits and concatenate
    tensors = []
    for split in ("train", "val", "test"):
        ds = dataset_class(split=split, download=True)
        imgs = ds.imgs
        tensors.append(imgs)

    all_imgs = np.concatenate(tensors, axis=0)
    all_imgs = all_imgs[:max_samples]
    t = torch.tensor(all_imgs, dtype=torch.float32)
    if t.ndim == 3:
        # Grayscale
        t = t.unsqueeze(1)
    else:
        # RGB
        t = t.permute(0, 3, 1, 2)

    # Normalize [0, 255] -> [-1, 1]
    t = t / 255.0
    t = t * 2.0 - 1.0

    n_channels = t.shape[1]
    channel_tag = "rgb" if n_channels == 3 else "gray"

    out_path = os.path.join(
        save_dir, f"medmnist_{name}_{channel_tag}{img_size}_{len(t)}.pt"
    )
    torch.save(t, out_path)

    print(f"[{name}] shape={tuple(t.shape)}  dtype={t.dtype}  -> {out_path}")


if __name__ == "__main__":
    save_dir = "data"
    max_samples = 5000
    img_size = 64

    datasets = [
        (PathMNIST,   "path"),      # colon pathology – RGB
        (ChestMNIST,  "chest"),     # chest X-rays – grayscale
        (OrganAMNIST, "organa"),    # abdominal CT – grayscale
    ]

    for ds_class, ds_name in datasets:
        load_and_save(
            dataset_class=ds_class,
            name=ds_name,
            img_size=img_size,
            save_dir=save_dir,
            max_samples=max_samples,
        )

    print("Done – all datasets saved to disk.")