import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from models.unet import UNet
import numpy as np
import random
import math


def train(batch_size: int,
          num_epochs: int,
          num_timesteps: int,
          lr: float = 2e-5):
    # TODO: use correct dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    model = UNet()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='mean')

    # TODO: checkpoint

    for i in range(num_epochs):
        pass

    # TODO: checkpoint


def sample():
    pass


def main():
    train(batch_size=128, num_epochs=15, num_timesteps=1_000)
    # sample()


if __name__ == '__main__':
    main()
