from celeba_dataset import CelebAGray32
from torch.utils.data import Subset
import torch
import os
import random

data_dir = r"/Users/frederikwillger/Downloads/img_align_celeba"
save_dir = "data"
img_size = 64
subset_size = 20_000

os.makedirs(save_dir, exist_ok=True)

full_dataset = CelebAGray32(data_dir, img_size=img_size)

random.seed(0)
indices = list(range(len(full_dataset)))
random.shuffle(indices)
indices = indices[:subset_size]
dataset = Subset(full_dataset, indices)

# Stack all tensors into one big tensor
all_tensors = torch.stack([dataset[i] for i in range(len(dataset))])

print(all_tensors.shape)
print(all_tensors.dtype)

# Save to disk
torch.save(all_tensors, f"{save_dir}/celeba_gray{img_size}_{subset_size}.pt")
print(f"Saved preprocessed CelebA subset to disk")
