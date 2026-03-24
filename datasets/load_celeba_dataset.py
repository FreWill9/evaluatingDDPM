from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import torch
import random
import os

"""
Use this script to load the CelebA dataset from disk, preprocess it and save as pt file.
"""

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


if __name__ == "__main__":
    data_dir = r"/Users/frederikwillger/Downloads/img_align_celeba"
    save_dir = "data"
    img_size = 64
    subset_size = 20000

    os.makedirs(save_dir, exist_ok=True)

    full_dataset = CelebAGray(data_dir, img_size=img_size)

    random.seed(9)
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    dataset = Subset(full_dataset, indices[:subset_size])

    all_tensors = torch.stack([dataset[i] for i in range(len(dataset))])
    print(all_tensors.shape)
    print(all_tensors.dtype)

    torch.save(all_tensors, f"{save_dir}/celeba_gray{img_size}_{subset_size}.pt")
    print("Saved preprocessed CelebA subset to disk")
