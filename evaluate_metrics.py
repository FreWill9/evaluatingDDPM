import torch
import torchvision
from torch_fidelity import calculate_metrics
from diffusion_utils import linear_beta_schedule, precompute_schedule
from sampling_utils import sample
from models.ho_unet import UNet as HoUNet
from models.simple_unet import SimpleUnet
from models.unet import UNET
import os

def save_preprocessed_real_images(data_path, save_dir):
    """Loads and saves preprocessed real images from .pt file to PNGs."""
    images = torch.load(data_path)  # [N, 1, img_size, img_size], normalized to [-1, 1]
    # Undo normalization
    images = images * 0.5 + 0.5
    images = images.clamp(0, 1)

    os.makedirs(save_dir, exist_ok=True)
    for i, img in enumerate(images):
        torchvision.utils.save_image(img, os.path.join(save_dir, f"{i:04d}.png"))
    print(f"Saved {len(images)} real images to: {save_dir}")


def save_images(model, schedule, device, img_size, num_timesteps, num_samples, save_dir="outputs"):
    """Generate and save a batch of images as PNGs."""
    os.makedirs(save_dir, exist_ok=True)
    samples = sample(model, (num_samples, 1, img_size, img_size), schedule, device, num_timesteps)
    # Undo normalization
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    for i in range(num_samples):
        torchvision.utils.save_image(samples[i], os.path.join(save_dir, f"{i:04d}.png"))


if __name__ == "__main__":

    save_path_real = r"D:/data/evaluation/real_images/celeba_32_5000"
    save_path_generated = r"D:/data/evaluation/generated_images/simple_unet32_celeba5000"

    # Step 1: Save preprocessed real images as PNGs for metric calculation
    data_path = "data/celeba_gray32_5000.pt"
    if not os.path.exists(save_path_real) or len(os.listdir(save_path_real)) == 0:
        save_preprocessed_real_images(data_path, save_path_real)
    else: 
        print(f"Real images already exist at: {save_path_real}, skipping saving step.")
        
    
    # Step 2: Generate and save images as PNGs from the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select model and load checkpoint
    model = SimpleUnet().to(device)
    checkpoint_path = "checkpoints/latest_simple_unet32.pt"
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    img_size = 32
    betas = linear_beta_schedule(1000).to(device)
    schedule = precompute_schedule(betas)

    num_timesteps = 1000
    num_samples = len(os.listdir(save_path_real))  # Match number of real images
    if not os.path.exists(save_path_generated) or len(os.listdir(save_path_generated)) == 0:
        save_images(model, schedule, device, img_size, num_timesteps=num_timesteps, num_samples=num_samples, save_dir=save_path_generated)
    else:
        print(f"Generated images already exist at: {save_path_generated}, skipping generation step.")

    # Compute FID and IS using torch-fidelity
    print()
    print("Calculating FID and Inception Score...")
    metrics = calculate_metrics(
        input1=save_path_real,
        input2=save_path_generated,
        fid=True,
        isc=True,
        cuda=torch.cuda.is_available(),
    )
    
    print(f"FID: {metrics['frechet_inception_distance']:.4f}")
    print(f"Inception Score: {metrics['inception_score_mean']:.4f} ± {metrics['inception_score_std']:.4f}")

    """
    Findings:
    - For celeba_32_5000 (real): IS = 3.2148 ± 0.0779
        - For simple_unet32_celeba5000 (generated): 3.2148 ± 0.0779, FID = 34.6164
        - For unet32_celeba5000 (generated): TODO
        - For ho_unet32_celeba5000 (generated): TODO
    """