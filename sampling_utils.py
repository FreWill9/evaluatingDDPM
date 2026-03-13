from matplotlib import pyplot as plt
import torch
import torchvision
from diffusion_utils import linear_beta_schedule, precompute_schedule, reverse_diffusion_step
from models.ho_unet import UNet as HoUNet
from models.simple_unet import SimpleUnet
from models.unet import UNET


@torch.no_grad()
def sample(model, shape, num_timesteps, schedule, device):
    """Generate image from pure noise."""
    img = torch.randn(shape, device=device)
    for i in reversed(range(num_timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        img = reverse_diffusion_step(model, img, t, i, schedule)
    return img


def plot_images(model, schedule, device, img_size, num_timesteps, num_samples=1):
    """Generate and plot a grid of images."""
    samples = sample(model, (num_samples, 1, img_size, img_size), num_timesteps, schedule, device)
    # De-normalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    grid = torchvision.utils.make_grid(samples.cpu(), nrow=4)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
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

    plot_images(model, schedule, device, img_size, num_timesteps=1000)
