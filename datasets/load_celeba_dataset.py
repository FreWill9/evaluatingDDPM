from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import torch
import random
import os

"""
Loads CelebA from disk, preprocesses to grayscale 64×64 tensors, and saves as .pt files:
  - Four training subsets: 1K, 5K, 10K, 20K (nested: 1K c 5K c 10K c 20K)
  - One fixed held-out set for evaluation drawn exclusively from images outside all training sets
"""

TRAIN_SIZES   = [1_000, 5_000, 10_000, 20_000]
HOLDOUT_SIZE  = 10_000

class CelebAGray(Dataset):
    """Lazy-loads CelebA jpg images, converting to img_size × img_size grayscale tensors."""
    def __init__(self, root, img_size):
        self.paths = sorted(Path(root).glob("*.jpg"))
        self.img_size = img_size
        if not self.paths:
            raise FileNotFoundError(f"No jpg images found in {root}")
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as img:
            return self.transform(img.convert("RGB"))


def tensors_from_indices(dataset, indices, desc=""):
    """Stack dataset items at given indices into a single float tensor."""
    tensors = torch.stack([dataset[i] for i in indices])
    if desc:
        print(f"  {desc}: shape={tensors.shape}, dtype={tensors.dtype}")
    return tensors


if __name__ == "__main__":
    data_dir = r"D:\data\img_align_celeba"
    save_dir = "data"
    img_size  = 64

    required = max(TRAIN_SIZES) + HOLDOUT_SIZE
    os.makedirs(save_dir, exist_ok=True)

    full_dataset = CelebAGray(data_dir, img_size=img_size)

    # Randomly shuffle indices and split into nested training subsets and a held-out evaluation set
    random.seed(1)
    all_indices = list(range(len(full_dataset)))
    random.shuffle(all_indices)

    train_pool   = all_indices[:max(TRAIN_SIZES)]
    holdout_idxs = all_indices[max(TRAIN_SIZES) : max(TRAIN_SIZES) + HOLDOUT_SIZE]

    # Save nested training subsets
    print("\nSaving training subsets...")
    for size in TRAIN_SIZES:
        subset_idxs = train_pool[:size]   # nested: always take the first N
        tensors = tensors_from_indices(
            full_dataset, subset_idxs, desc=f"train_{size}"
        )
        out_path = f"{save_dir}/celeba_gray{img_size}_{size}.pt"
        torch.save(tensors, out_path)
        print(f"  Saved → {out_path}")

    # Save held-out evaluation set
    print("\nSaving held-out evaluation set...")
    holdout_tensors = tensors_from_indices(
        full_dataset, holdout_idxs, desc=f"holdout_{HOLDOUT_SIZE}"
    )
    holdout_path = f"{save_dir}/celeba_gray{img_size}_holdout{HOLDOUT_SIZE}.pt"
    torch.save(holdout_tensors, holdout_path)
    print(f"  Saved → {holdout_path}")

    # Sanity check: no index overlap between any split and the held-out set ---
    holdout_set = set(holdout_idxs)
    for size in TRAIN_SIZES:
        overlap = holdout_set & set(train_pool[:size])
        assert not overlap, f"Overlap detected between train_{size} and holdout!"
    print("\nSanity check passed: zero overlap between all training splits and held-out set.")
