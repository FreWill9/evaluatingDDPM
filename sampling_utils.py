from matplotlib import pyplot as plt
import torch
import torchvision
from diffusion_utils import linear_beta_schedule, precompute_schedule, reverse_diffusion_step
from models.ho_unet import UNet as HoUNet
from models.simple_unet import SimpleUnet
from models.unet import UNET

@torch.no_grad()
def sample(model, shape, schedule, device, num_timesteps: int = 1000):
    """Generate image from pure noise."""
    img = torch.randn(shape, device=device)
    for i in reversed(range(num_timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        img = reverse_diffusion_step(model, img, t, i, schedule)
    return img

def sample_and_plot(model, schedule, device, img_size, num_timesteps: int = 1000, num_samples: int = 1):
    """Generate and plot a grid of images."""
    samples = sample(model, (num_samples, 1, img_size, img_size), schedule, device, num_timesteps=num_timesteps)
    # De-normalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    grid = torchvision.utils.make_grid(samples.cpu(), nrow=4)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.show()
    plt.close()

def sample_and_save(model, schedule, device, img_size, outdir: str, num_timesteps:int = 1000, num_samples: int = 1):
    """Generate a grid of images and save as svg."""
    samples = sample(model, (num_samples, 1, img_size, img_size), schedule, device, num_timesteps=num_timesteps)
    # De-normalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    grid = torchvision.utils.make_grid(samples.cpu(), nrow=4)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    plt.savefig(f"{outdir}/grid.svg", format="svg", bbox_inches='tight')
    print(f"Saved image successfully to '{outdir}/grid.svg'!")
    plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select model and load checkpoint
    model = UNET().to(device)
    checkpoint_path = "checkpoints/ddpm_checkpoint_gpu64.pt"
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    img_size = 64
    betas = linear_beta_schedule(1000).to(device)
    schedule = precompute_schedule(betas)

    sample_and_save(model, schedule, device, img_size, "outdir", num_samples=12)
