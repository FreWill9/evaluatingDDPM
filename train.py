import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
from diffusion_utils import linear_beta_schedule, precompute_schedule, forward_diffusion

from models.ho_unet import UNet as HoUNet
from models.simple_unet import SimpleUnet
from models.unet import UNET


def train(data_path, checkpoint_name, batch_size, num_epochs, num_timesteps):
    
    tensors = torch.load(data_path)
    dataset = TensorDataset(tensors)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    criterion = nn.MSELoss()

    betas = linear_beta_schedule(1000).to(device)
    schedule = precompute_schedule(betas)

    # Resume from checkpoint if available
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = os.path.join("checkpoints", f"{checkpoint_name}.pt")
    start_epoch = 0

    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"  → Resuming at epoch {start_epoch}")

    # Training loop
    for epoch in tqdm(range(start_epoch, num_epochs), desc="Training"):
        model.train()
        epoch_loss = 0.0

        for (x_0,) in train_loader:
            x_0 = x_0.to(device)

            # Sample random timesteps uniformly for each image in the batch
            t = torch.randint(0, num_timesteps, (x_0.shape[0],), device=device)

            x_t, noise = forward_diffusion(x_0, t, schedule)
            predicted_noise = model(x_t, t)
            loss = criterion(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        tqdm.write(f"Epoch [{epoch + 1}/{num_epochs}]  Loss: {avg_loss:.6f}")

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "num_timesteps": num_timesteps,
        }, checkpoint_path)
        tqdm.write(f"Checkpoint saved to {checkpoint_path}")
    print("Training complete.")


if __name__ == "__main__":
    data_path = "data/celeba_gray32_5000.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select model architecture
    model = SimpleUnet().to(device)

    checkpoint_name = "latest_simple_unet32"
    batch_size = 128
    num_epochs = 3
    num_timesteps = 1000
    learn_rate = 0.0002

    train(data_path, checkpoint_name, batch_size, num_epochs, num_timesteps)
