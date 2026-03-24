import random
import numpy as np
from PIL import Image
from sklearn.metrics import pairwise_distances
import torch
import torchvision
from torch_fidelity import calculate_metrics
from tqdm import tqdm
from diffusion_utils import get_beta_schedule, precompute_schedule
from sampling_utils import sample
from models.ho_unet import UNet as HoUNet
from models.simple_unet import SimpleUnet
from models.unet import UNET
import os
import json
import pandas as pd

def save_preprocessed_real_images(data_path, save_dir, num_samples=None):
    """Loads and saves preprocessed real images from .pt file to PNGs."""
    images = torch.load(data_path)  # [N, 1, img_size, img_size], normalized to [-1, 1]
    # Subsample before saving
    if num_samples is not None:
        if num_samples > len(images):
            raise ValueError(f"Requested {num_samples} samples, but dataset has only {len(images)}")
        else:
            indices = random.sample(range(len(images)), num_samples)
            images = images[indices]
    # Undo normalization
    images = images * 0.5 + 0.5
    images = images.clamp(0, 1)

    os.makedirs(save_dir, exist_ok=True)
    for i, img in enumerate(images):
        torchvision.utils.save_image(img, os.path.join(save_dir, f"{i:04d}.png"))
    print(f"Saved {len(images)} real images to: {save_dir}")


def save_images(model, schedule, device, img_size, num_timesteps, num_samples, save_dir):
    """Generate and save a batch of images as PNGs."""
    samples = sample(model, (num_samples, 1, img_size, img_size), schedule, device, num_timesteps)
    # Undo normalization
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    start_index = len(os.listdir(save_dir))
    for i, img in enumerate(samples, start=start_index):
        torchvision.utils.save_image(img, os.path.join(save_dir, f"{i:04d}.png"))


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
    return nn_gen_real.mean(), nn_real_real.mean()


if __name__ == "__main__":

    # Select model architecture
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUnet().to(device)

    # Set paths to preprocessed dataset, holdout dataset, and model checkpoint
    data_path = "data/celeba_gray64_1000.pt" # For NN distance
    data_path_holdout = "data/celeba_gray64_holdout10000.pt" # For metrics FID, KID, IS
    checkpoint_name = "simple_1000_100eps500" # also used for csv entry
    checkpoint_path = f"checkpoints_dataset_size/{checkpoint_name}.pt"

    # Set paths to saving directory for real and generated images for evaluation
    save_path_train = "evaluation_data/real_images/celeba_gray64_1000"
    save_path_generated = f"evaluation_data/generated_images/{checkpoint_name}"

    # Path to holdout set stays the same across evaluations
    save_path_holdout = "evaluation_data/real_images/celeba_gray64_holdout10000"

    # Set image size and number of samples for evaluation
    img_size = 64
    num_samples_for_evaluation = 2000 # Number of samples to generate
    num_samples_per_batch = 50 # Generate and save in batches to avoid memory issues

    # Step 1: Save preprocessed real images as PNGs for metric calculation

    if not os.path.exists(save_path_train) or len(os.listdir(save_path_train)) == 0:
        save_preprocessed_real_images(data_path, save_path_train)
        print(f"Saved preprocessed real images for evaluation at: {save_path_train}")
    else: 
        print(f"Real images already exist at: {save_path_train}, skipping saving step.")

    if not os.path.exists(save_path_holdout) or len(os.listdir(save_path_holdout)) == 0:
        save_preprocessed_real_images(data_path_holdout, save_path_holdout)
        print(f"Saved preprocessed holdout images for evaluation at: {save_path_holdout}")
    else: 
        print(f"Holdout images already exist at: {save_path_holdout}, skipping saving step.")

    
    # Step 2: Generate and save images as PNGs from the trained model for metric calculation

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Read schedule type from checkpoint and precompute values
    schedule_type = ckpt["schedule_type"]
    num_timesteps = ckpt["num_timesteps"]
    betas = get_beta_schedule(schedule_type, num_timesteps, device)
    schedule = precompute_schedule(betas)

    # Generate and save images if not already done
    os.makedirs(save_path_generated, exist_ok=True)
    num_generated_samples = len(os.listdir(save_path_generated))

    if not os.path.exists(save_path_generated) or num_generated_samples < num_samples_for_evaluation:
        print("Generating and saving images for evaluation...")
        # Sample in batches until we have enough samples for evaluation
        with tqdm(total=num_samples_for_evaluation, desc="Generating images") as pbar:
            pbar.update(len(os.listdir(save_path_generated)))
            while len(os.listdir(save_path_generated)) < num_samples_for_evaluation:
                remaining = num_samples_for_evaluation - len(os.listdir(save_path_generated))
                batch = min(remaining, num_samples_per_batch)
                save_images(model, schedule, device, img_size, num_timesteps, batch, save_path_generated)
                pbar.update(batch)
            print(f"Saved generated images for evaluation at: {save_path_generated}")
    else:
        print(f"Generated images already exist at: {save_path_generated}, skipping generation step.")

    # Step 3: Compute FID, KID, and IS using torch-fidelity

    print("\nComputing FID, KID and Inception Score using torch-fidelity...")
    metrics = calculate_metrics(
        input1=save_path_holdout,
        input2=save_path_generated,
        fid=True,
        isc=True,
        kid=True,
        cuda=torch.cuda.is_available(),
    )
    # Compute IS of the real holdout data
    is_holdout = calculate_metrics(
        input1=save_path_holdout,
        isc=True,
        cuda=torch.cuda.is_available(),
    )
    
    print(f"FID: {metrics['frechet_inception_distance']:.4f}")
    print(f"KID: {metrics['kernel_inception_distance_mean']:.4f} ± {metrics['kernel_inception_distance_std']:.4f}")
    print(f"IS (holdout real): {is_holdout['inception_score_mean']:.4f} ± {is_holdout['inception_score_std']:.4f}")
    print(f"IS (generated): {metrics['inception_score_mean']:.4f} ± {metrics['inception_score_std']:.4f}")

    # Compute nearest-neighbor pixel distances
    nn_gen_real_mean, nn_real_real_mean = compute_nn_distance(save_path_holdout, save_path_generated)

    # Save results
    result = {
    "checkpoint_name": checkpoint_name,
    "num_train_samples": ckpt["num_train_samples"],
    "num_epochs": ckpt["epoch"],
    "num_samples_evaluation": num_samples_for_evaluation,
    "fid": metrics['frechet_inception_distance'],
    "kid_mean": metrics['kernel_inception_distance_mean'],
    "kid_std": metrics['kernel_inception_distance_std'],
    "is_holdout_mean": is_holdout['inception_score_mean'],
    "is_holdout_std": is_holdout['inception_score_std'],
    "is_generated_mean": metrics['inception_score_mean'],
    "is_generated_std": metrics['inception_score_std'],
    "nn_gen_real_mean": float(nn_gen_real_mean),
    "nn_real_real_mean": float(nn_real_real_mean),
    }
    
    # Save results to CSV file (append one row)
    csv_path = "metrics.csv"
    df_new = pd.DataFrame([result])
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(csv_path, index=False)

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
                    - Cosine schedule: FID: 16.3952
                    - Sigmoid schedule: FID: 32.7254 (tau=1.0), KID: 0.0356 +- 0.0011
                - After 5000 epochs:
                    - IS: 1.7409 ± 0.0336, FID: 6.0066
                    - Nearest-neighbor pixel distances:
                        - Gen->Real  NN distance (mean): 0.9834, median: 0.9644
                        - Real->Real NN distance (mean): 1.4271, median: 1.3974
                        - Ratio (gen->real / real->real): 0.6889
    - For celeba_64_20000 (real): IS = ?
        - Evaluation of 2000 generated samples with 2000 randomly selected real images:
        - For simple_unet64_celeba20000:
                - 
                - 
        - For unet64_celeba20000 (generated): TODO
    -----------------------------------------------------------------------------------------------------
    Experiments on dataset size:
    - For celeba_64_20000 (real): IS = ?
        - simple_20000_100eps (generated):
    """
