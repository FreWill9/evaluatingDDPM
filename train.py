import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
from diffusion_utils import get_beta_schedule, precompute_schedule, forward_diffusion

from models.ho_unet import UNet as HoUNet
from models.simple_unet import SimpleUnet
from models.unet import UNET

def train(model, data_path,  checkpoint_name, batch_size, num_epochs, num_timesteps, 
            learning_rate, schedule_type="linear"):
    
    tensors = torch.load(data_path)
    dataset = TensorDataset(tensors)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # This adds learning rate decay. Computes lr * gamma, every step_size epochs. This halves the lr after around 150 epochs.
    # For stronger decay choose smaller gamma, for no lr-decay choose gamma = 1.
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    criterion = nn.MSELoss()

    # Use Automatic Mixed Precision (AMP) on GPU to speed up training
    use_amp = device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)

    # Resume from checkpoint if available
    checkpoint_folder = "checkpoints_dataset_size"
    os.makedirs(checkpoint_folder, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_folder, f"{checkpoint_name}.pt")
    start_epoch = 0
    loss_history = []

    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if "scheduler_state_dict" in ckpt:
            lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        start_epoch = ckpt["epoch"] + 1
        loss_history = ckpt.get("loss_history", [])
        # Override schedule_type from checkpoint to ensure consistency
        schedule_type = ckpt.get("schedule_type", schedule_type)
        print(f"  → Resuming at epoch {start_epoch}")

    # Build schedule based on checkpoint or specified type (if no checkpoint)
    schedule = precompute_schedule(get_beta_schedule(schedule_type, num_timesteps, device))

    # Training loop
    for epoch in tqdm(range(start_epoch, num_epochs), desc="Training"):
        model.train()
        epoch_loss = 0.0

        for (x_0,) in train_loader:
            x_0 = x_0.to(device)

            # Sample random timesteps uniformly for each image in the batch
            t = torch.randint(0, num_timesteps, (x_0.shape[0],), device=device)

            optimizer.zero_grad()
 
            with autocast(device.type, enabled=use_amp):
                # Forward pass in mixed precision: saves memory and speeds up computation
                x_t, noise = forward_diffusion(x_0, t, schedule)
                predicted_noise = model(x_t, t)
                loss = criterion(predicted_noise, noise)
 
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
 
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        lr_scheduler.step()

        tqdm.write(f"Epoch [{epoch + 1}/{num_epochs}]  Loss: {avg_loss:.6f}")
        loss_history.append(float(avg_loss))

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "loss": avg_loss,
            "loss_history": loss_history,
            "num_timesteps": num_timesteps,
            "schedule_type": schedule_type,
            "num_train_samples": len(dataset),
        }, checkpoint_path)
        tqdm.write(f"Checkpoint saved to {checkpoint_path}")

    print("Training complete.")


if __name__ == "__main__":
    data_path = "data/celeba_gray64_1000.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select model architecture
    model = SimpleUnet().to(device)

    batch_size = 32
    num_epochs = 100
    num_timesteps = 500
    initial_learn_rate = 0.0002

    checkpoint_name = f"simple_{len(torch.load(data_path))}_{num_epochs}eps{num_timesteps}"

    train(model, data_path, checkpoint_name, batch_size, num_epochs, num_timesteps, 
          initial_learn_rate, schedule_type="linear")
