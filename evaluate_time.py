import torch
import time
from sampling_utils import sample
from diffusion_utils import get_beta_schedule, precompute_schedule
from models.simple_unet import SimpleUnet
from models.unet import UNET
from models.unet_DS import UNET_DS

def print_training_time(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    time = checkpoint.get("epoch_time_history", [])
    time_sum = sum(time)
    print (f"Time for {checkpoint_path}: {time_sum}")

def print_sample_time(model, schedule, device, img_size, num_timesteps, num_samples):
    """Generate and save a batch of images as PNGs."""
    start_time = time.time()
    sample(model, (num_samples, 1, img_size, img_size), schedule, device, num_timesteps)
    end_time = time.time()
    print(f"Time for {num_samples} samples: {end_time - start_time}")

if __name__ == "__main__":
    #print_training_time("checkpoints_dataset_size/simple_20000_100eps500.pt")

    # Select model architecture and checkpoint for sampling
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNET(time_steps=1000).to(device)
    checkpoint_path = "checkpoints/noDS500_linear_200.pt"

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Read schedule type from checkpoint and precompute values
    schedule_type = ckpt["schedule_type"]
    num_timesteps = ckpt["num_timesteps"]
    betas = get_beta_schedule(schedule_type, num_timesteps, device)
    schedule = precompute_schedule(betas)

    print_sample_time(model, schedule, device, img_size=64, num_timesteps=num_timesteps, num_samples=50)