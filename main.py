import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
import torch.optim as optim
from timm.utils import ModelEmaV3
from models.unet import UNET
from utils import DDPM_Scheduler
from einops import rearrange
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List
from celeba_dataset import CelebAGray32


def train(pt_path: str,
          img_size: int = 32,
          batch_size: int = 128,
          num_epochs: int = 15,
          num_timesteps: int = 1_000,
          subset_size: int = 5_000,
          lr: float = 2e-5,
          ema_decay: float = 0.9999,
          checkpoint_path: str = None):
    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset
    all_tensors = torch.load(pt_path, weights_only=True)
    dataset = TensorDataset(all_tensors)
    print(f"Dataset size: {len(dataset)} images loaded from {pt_path}")
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
    ema = ModelEmaV3(model, decay=ema_decay)
    scheduler = DDPM_Scheduler(num_timesteps=num_timesteps)

    # Checkpoint
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    for i in range(num_epochs):
        total_loss = 0
        for bidx, (x,) in enumerate(tqdm(train_loader, desc=f'Epoch {i + 1}/{num_epochs}')):
            x = x.to(device)
            bs = x.size(0)

            # Select timestep t = [0, T]
            t = torch.randint(0, num_timesteps, (bs,), device=x.device)

            # Sample Gaussian Noise e ~ N(0, I)
            e = torch.randn_like(x, requires_grad=False)

            # Create noisy image x_t
            a = scheduler.alpha[t].view(batch_size, 1, 1, 1).to(device)
            x_t = (torch.sqrt(a) * x) + (torch.sqrt(1 - a) * e)

            # Feed noisy image to model
            output = model(x_t, t)

            # Train for output = e
            optimizer.zero_grad()
            loss = criterion(output, e)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            ema.update(model)

        print(f'Epoch {i + 1} | Loss {total_loss / len(train_loader):.5f}')

        # checkpoint
        checkpoint = {
            'weights': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'ema': ema.state_dict()
        }
        torch.save(checkpoint, 'checkpoints/ddpm_checkpoint1')


def sample(checkpoint_path: str = None,
           num_timesteps: int = 1_000,
           ema_decay: float = 0.9999):
    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path)
    model = UNET(device=device)
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPM_Scheduler(num_timesteps=num_timesteps)
    times = [0, 15, 50, 100, 200, 300, 400, 550, 700, 999]
    images = []

    with torch.no_grad():
        model = ema.module.eval()
        for i in range(10):
            z = torch.randn(1, 1, 32, 32)
            for t in reversed(range(1, num_timesteps)):
                t = [t]
                temp = (scheduler.beta[t] / (
                        (torch.sqrt(1 - scheduler.alpha[t])) * (torch.sqrt(1 - scheduler.beta[t]))))
                z = (1 / (torch.sqrt(1 - scheduler.beta[t]))) * z - (temp * model(z.to(device), t).cpu())
                if t[0] in times:
                    images.append(z)
                e = torch.randn(1, 1, 32, 32)
                z = z + (e * torch.sqrt(scheduler.beta[t]))
            temp = scheduler.beta[0] / ((torch.sqrt(1 - scheduler.alpha[0])) * (torch.sqrt(1 - scheduler.beta[0])))
            x = (1 / (torch.sqrt(1 - scheduler.beta[0]))) * z - (temp * model(z.to(device), [0]).cpu())

            images.append(x)
            x = rearrange(x.squeeze(0), 'c h w -> h w c').detach()
            x = x.numpy()
            plt.imshow(x)
            plt.show()
            display_reverse(images)
            images = []


def display_reverse(images: List):
    fig, axes = plt.subplots(1, 10, figsize=(10, 1))
    for i, ax in enumerate(axes.flat):
        x = images[i].squeeze(0)
        x = rearrange(x, 'c h w -> h w c')
        x = x.numpy()
        ax.imshow(x)
        ax.axis('off')
    plt.show()


def main():
    # train(pt_path=r"data/celeba_gray32_5000.pt", img_size=32, batch_size=128, num_epochs=15, num_timesteps=1_000)
    sample('checkpoints/ddpm_checkpoint1')


if __name__ == '__main__':
    main()
