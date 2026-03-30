from time import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
from tqdm import tqdm
import os
from diffusion_utils import get_beta_schedule, precompute_schedule, forward_diffusion

from models.ho_unet import UNet as HoUNet
from models.simple_unet import SimpleUnet
from models.unet import UNET
from models.unet_DS import UNET_DS


def train(model, data_path,  checkpoint_name, batch_size, num_epochs, num_timesteps, 
            learning_rate, schedule_type="linear", deep_supervision=False):
    
    tensors = torch.load(data_path)
    dataset = TensorDataset(tensors)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=1,
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
    checkpoint_folder = "checkpoints"
    os.makedirs(checkpoint_folder, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_folder, f"{checkpoint_name}.pt")
    start_epoch = 0
    loss_history = []
    epoch_time_history = []
    if deep_supervision: aux_loss_history = []

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
        if deep_supervision: aux_loss_history = ckpt.get("aux_loss_history", [])
        epoch_time_history = ckpt.get("epoch_time_history", [])
        # Override schedule_type from checkpoint to ensure consistency
        schedule_type = ckpt.get("schedule_type", schedule_type)
        print(f"  → Resuming at epoch {start_epoch}")

    # Build schedule based on checkpoint or specified type (if no checkpoint)
    schedule = precompute_schedule(get_beta_schedule(schedule_type, num_timesteps, device))

    # Training loop
    for epoch in tqdm(range(start_epoch, num_epochs), desc="Training"):
        epoch_start_time = time()
        model.train()
        epoch_loss = 0.0
        if deep_supervision: epoch_aux_loss = 0.0

        for (x_0,) in train_loader:
            x_0 = x_0.to(device)

            # Sample random timesteps uniformly for each image in the batch
            t = torch.randint(0, num_timesteps, (x_0.shape[0],), device=device)

            optimizer.zero_grad()
 
            with autocast(device.type, enabled=use_amp):
                # Forward pass in mixed precision: saves memory and speeds up computation
                x_t, noise = forward_diffusion(x_0, t, schedule)
                if deep_supervision:
                    predicted_noise, aux_out = model(x_t, t)
                    # Upsample auxiliary output and compute auxiliary loss to enable deep supervision
                    aux_pred = F.interpolate(aux_out, size=noise.shape[-2:], mode="bilinear", align_corners=False)
                    aux_loss = criterion(aux_pred, noise)
                    
                else:
                    predicted_noise = model(x_t, t)
                
                loss = criterion(predicted_noise, noise)

                if deep_supervision:
                    # Weighted sum of losses
                    lambda_aux = 0.01
                    loss = loss + lambda_aux * aux_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
 
            epoch_loss += loss.item()
            if deep_supervision: epoch_aux_loss += aux_loss.item()

        avg_loss = epoch_loss / len(train_loader)
        if deep_supervision: avg_aux_loss = epoch_aux_loss / len(train_loader)
        lr_scheduler.step()

        # Measure time taken for epoch and compute samples per second
        epoch_time = time() - epoch_start_time
        epoch_time_history.append(float(epoch_time))

        tqdm.write(f"Epoch [{epoch + 1}/{num_epochs}]  Loss: {avg_loss:.6f}")
        loss_history.append(float(avg_loss))
        if deep_supervision: aux_loss_history.append(float(avg_aux_loss))

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "loss": avg_loss,
            "loss_history": loss_history,
            "epoch_time_history": epoch_time_history,
            "num_timesteps": num_timesteps,
            "schedule_type": schedule_type,
            "num_train_samples": len(dataset),
        }

        if deep_supervision: checkpoint["aux_loss_history"] = aux_loss_history

        torch.save(checkpoint, checkpoint_path)
        tqdm.write(f"Checkpoint saved to {checkpoint_path}")

    print("Training complete.")


if __name__ == "__main__":
    data_path = "data/celeba_gray64_10000.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 32
    num_epochs = 700
    num_timesteps = 500
    initial_learn_rate = 0.0002

    checkpoint_name = f"DS500_linear"

    # Select model architecture
    model = UNET_DS().to(device)

    train(model, data_path, checkpoint_name, batch_size, num_epochs, num_timesteps, 
          initial_learn_rate, schedule_type="linear", deep_supervision=True)
