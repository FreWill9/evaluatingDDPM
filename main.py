import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
import torch.optim as optim
from timm.utils import ModelEmaV3
from models.unet import UNET
from utils import DDPM_Scheduler
from einops import rearrange
from tqdm import tqdm
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List


def train(pt_path: str,
          img_size: int = 32,
          batch_size: int = 128,
          num_epochs: int = 15,
          num_timesteps: int = 1_000,
          subset_size: int = 5_000,
          lr: float = 2e-5,
          ema_decay: float = 0.999,
          checkpoint_path: str = None):
    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load dataset
    all_tensors = torch.load(pt_path, weights_only=True)
    dataset = TensorDataset(all_tensors)
    print(f"Dataset size: {len(dataset)} images loaded from {pt_path}")
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
    )

    # choose model
    model = UNET(device=device)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='mean')
    ema = ModelEmaV3(model, decay=ema_decay)
    scheduler = DDPM_Scheduler(num_timesteps=num_timesteps)
    alpha_bar = scheduler.alpha_bar.to(device)

    # Checkpoint
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    for i in range(num_epochs):
        total_loss = 0.0
        for bidx, (x,) in enumerate(tqdm(train_loader, desc=f'Epoch {i + 1}/{num_epochs}')):
            optimizer.zero_grad(set_to_none=True)

            x = x.to(device)
            bs = x.size(0)

            # Select timestep t = [0, T]
            t = torch.randint(0, num_timesteps, (bs,), device=x.device)

            # Sample Gaussian Noise e ~ N(0, I)
            e = torch.randn_like(x, requires_grad=False)

            # Create noisy image x_t
            # a_bar = scheduler.alpha_bar.to(device)[t].view(bs, 1, 1, 1)
            a_bar = alpha_bar[t].view(bs, 1, 1, 1)
            x_t = torch.sqrt(a_bar) * x + torch.sqrt(1 - a_bar) * e

            # Feed noisy image to model
            output = model(x_t, t)

            # Train for output = e
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
        torch.save(checkpoint, 'checkpoints/ddpm_checkpoint_gpu64')


def sample(checkpoint_path: str = None,
           num_timesteps: int = 1_000,
           ema_decay: float = 0.999,
           img_size: int = 32):
    
    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path)

    model = UNET(device=device).to(device)
    model.load_state_dict(checkpoint['weights'])

    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    model.eval()

    scheduler = DDPM_Scheduler(num_timesteps=num_timesteps).to(device)

    times = np.linspace(0, 999, num=8, dtype=int)

    with torch.no_grad():
        for i in range(5):
            z = torch.randn(1, 1, img_size, img_size).to(device)
            images = []

            for t in reversed(range(num_timesteps)):
                t_tensor = torch.full((1,), t, device=device, dtype=torch.long)

                beta_t = scheduler.beta[t]
                alpha_t = scheduler.alpha[t]
                alpha_bar_t = scheduler.alpha_bar[t]

                eps_theta = model(z, t_tensor)

                mean = (1.0 / torch.sqrt(alpha_t)) * (
                    z - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * eps_theta
                )

                if t > 0:
                    noise = torch.randn_like(z)
                    z = mean + torch.sqrt(beta_t) * noise
                else:
                    z = mean

                if t in times:
                    images.append(z.detach().cpu())

            x = z.detach().cpu()

            x_vis = x.detach().cpu().clamp(-1, 1)
            x_vis = (x_vis + 1) / 2
            img = x_vis.squeeze().numpy()

            plt.imshow(img, cmap="gray", vmin=0, vmax=1)
            plt.axis("off")
            plt.savefig(f"outdir/img_{i}.png", bbox_inches="tight", pad_inches=0)
            plt.close()
            display_reverse(images, f"outdir/rev_img_{i}.png")
            print(f"Saved image {i} successfully!")


def display_reverse(images: List, filename: str):
    n = min(len(images), 10)

    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))

    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        x = images[i].squeeze(0).detach().cpu().clamp(-1, 1)
        x = (x + 1) / 2
        x = x.squeeze().numpy()
        ax.imshow(x, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


def main():
    train(pt_path=r"data/celeba_gray64_20000.pt", img_size=64, batch_size=32, num_epochs=12, num_timesteps=1_000, lr=1e-4)
    sample('checkpoints/ddpm_checkpoint_gpu64', img_size=64)


if __name__ == '__main__':
    main()
