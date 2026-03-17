import numpy as np
from PIL import Image
from sklearn.metrics import pairwise_distances
import torch
import torchvision
from torch_fidelity import calculate_metrics
from diffusion_utils import linear_beta_schedule, cosine_beta_schedule, precompute_schedule
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


def compute_nn_distance(real_dir, gen_dir, subsample=None):
    """
    Computes nearest-neighbor pixel distances between generated and real images.
    Prints gen->real and real->real distances and their ratio (gen->real / real->real).

    Interpretation:
    - If ratio ≈ 1 -> good: Generated images are as close to real images as real images are to each other.
    - If ratio << 1 -> bad: Generated images are closer to real images than real images are to each other (potential memorizing).
    - If ratio >> 1 -> bad: Generated images are much farther from real images than real images are from each other (bad generation quality).
    """
    def load_images_flat(directory):
        paths = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".png")])
        imgs = []
        for p in paths:
            img = np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0
            imgs.append(img.flatten())
        return np.stack(imgs)

    real = load_images_flat(real_dir)
    gen  = load_images_flat(gen_dir)

    # Optional subsampling for faster distance computation
    if subsample is not None:
        real_idx = np.random.choice(len(real), size=min(subsample, len(real)), replace=False)
        gen_idx  = np.random.choice(len(gen),  size=min(subsample, len(gen)),  replace=False)
        real = real[real_idx]
        gen  = gen[gen_idx]
        print(f"  Subsampled to {len(gen)} generated / {len(real)} real images")

    print("\nComputing nearest-neighbor pixel distances...")
    gen_real_dists = pairwise_distances(gen, real, metric="euclidean")
    nn_gen_real = gen_real_dists.min(axis=1)

    real_real_dists = pairwise_distances(real, real, metric="euclidean")
    np.fill_diagonal(real_real_dists, np.inf)
    nn_real_real = real_real_dists.min(axis=1)

    print(f"  Gen->Real  NN distance (mean): {nn_gen_real.mean():.4f}, median: {np.median(nn_gen_real):.4f}")
    print(f"  Real->Real NN distance (mean): {nn_real_real.mean():.4f}, median: {np.median(nn_real_real):.4f}")
    print()
    ratio = nn_gen_real.mean() / nn_real_real.mean()
    print(f"  Ratio (gen->real / real->real): {ratio:.4f}")


if __name__ == "__main__":

    # Select model architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUnet().to(device)

    # Select beta schedule and precompute schedule
    num_timesteps = 1000
    betas = cosine_beta_schedule(num_timesteps).to(device)
    schedule = precompute_schedule(betas)

    # Set path to preprocessed dataset and model checkpoint
    data_name = "digits_gray16_1797"
    data_path = f"data/{data_name}.pt"

    checkpoint_name = "simple_unet_5000eps_digits16"
    checkpoint_path = f"checkpoints/{checkpoint_name}.pt"

    # Set paths for real and generated images for evaluation
    save_path_real = f"D:/data/evaluation/real_images/{data_name}"
    save_path_generated = f"D:/data/evaluation/generated_images/{checkpoint_name}"

    # Set image size and number of samples to generate for evaluation
    img_size = 16
    num_samples = len(os.listdir(save_path_real)) # match number of real images


    # Step 1: Save preprocessed real images as PNGs for metric calculation

    if not os.path.exists(save_path_real) or len(os.listdir(save_path_real)) == 0:
        save_preprocessed_real_images(data_path, save_path_real)
        print(f"Saved preprocessed real images for evaluation at: {save_path_real}")
    else: 
        print(f"Real images already exist at: {save_path_real}, skipping saving step.")

    
    # Step 2: Generate and save images as PNGs from the trained model for metric calculation

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if not os.path.exists(save_path_generated) or len(os.listdir(save_path_generated)) == 0:
        print("Generating and saving generated images for evaluation...")
        save_images(model, schedule, device, img_size, num_timesteps=num_timesteps, num_samples=num_samples, save_dir=save_path_generated)
    else:
        print(f"Generated images already exist at: {save_path_generated}, skipping generation step.")

    # Step 3: Compute FID and IS using torch-fidelity

    print("\nComputing FID and Inception Score using torch-fidelity...")
    metrics = calculate_metrics(
        input1=save_path_real,
        input2=save_path_generated,
        fid=True,
        isc=True,
        cuda=torch.cuda.is_available(),
    )
    
    print(f"FID: {metrics['frechet_inception_distance']:.4f}")
    print(f"Inception Score: {metrics['inception_score_mean']:.4f} ± {metrics['inception_score_std']:.4f}")

    # Compute nearest-neighbor pixel distances
    compute_nn_distance(save_path_real, save_path_generated)

    """
    Findings:
    - For celeba_32_5000: IS (real) = 3.2148 ± 0.0779
        - For simple_unet32_celeba5000 (generated): 3.2148 ± 0.0779, FID = 34.6164
        - For unet32_celeba5000 (generated): TODO
        - For ho_unet32_celeba5000 (generated): TODO
    - For digits_16_1797 (real): IS (real) = 1.7409 ± 0.0336
        - Simple Unet with reduced architecture: down_channels = (32, 64, 128), up_channels = (128, 64, 32)
                - After 100 epochs:
                    - IS: 1.7409 ± 0.0336, FID: 20.8106
                    - Cosine schedule: FID: 111.3080 (maybe some mistake in cosine implementation or during training?)
                - After 5000 epochs:
                    - IS: 1.7409 ± 0.0336, FID: 6.0066
                    - Nearest-neighbor pixel distances:
                        - Gen->Real  NN distance (mean): 0.9834, median: 0.9644
                        - Real->Real NN distance (mean): 1.4271, median: 1.3974
                        - Ratio (gen->real / real->real): 0.6889
    """
