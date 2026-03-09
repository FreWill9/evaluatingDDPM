import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from models.unet import UNET
import numpy as np
import random
import math
from celeba_dataset import CelebAGray32

def train(data_dir: str, img_size: int, batch_size: int,
          num_epochs: int,
          num_timesteps: int,
          lr: float = 2e-5):

    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CelebAGray32(data_dir, img_size=img_size)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    # choose model
    model = UNET(device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='mean')

    # TODO: checkpoint

    for i in range(num_epochs):
        pass

    # TODO: checkpoint


def sample():
    pass


def main():
    train(data_dir=r"D:/data/img_align_celeba", img_size=32, batch_size=128, num_epochs=15, num_timesteps=1_000)
    # sample()


if __name__ == '__main__':
    main()
