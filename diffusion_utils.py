import math
import numpy as np
import torch
import torch.nn.functional as F


def linear_beta_schedule(num_timesteps, beta_start=0.0001, beta_end=0.02):
    """Linear variance schedule from DDPM paper."""
    return torch.linspace(beta_start, beta_end, num_timesteps)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Copied from https://github.com/openai/improved-diffusion.git.

    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def cosine_beta_schedule(num_timesteps):
    """
    Cosine variance schedule from Improved DDPM paper (Nichol & Dhariwal, 2021).

    Defines alpha_bar(t) as a cosine curve, producing a slower, smoother signal decay compared to the linear schedule. 
    The +0.008 offset prevents beta from being too small near t=0, which would make the first few
    forward steps almost noiseless and waste network capacity.
    """
    betas = betas_for_alpha_bar(
        num_timesteps,
        lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
    )
    return torch.from_numpy(betas).float()


def sigmoid_schedule(t, start, end, tau, clip_min=1e-9):
    """
    Sigmoid schedule copied from https://github.com/google-research/pix2seq and adapted to PyTorch.

    Args:
        t: `float` between 0 and 1.
        start: `float` starting point in x-axis of sigmoid function.
        end: `float` ending point in x-axis of sigmoid function.
        tau: `float` scaling temperature for sigmoid function.
        clip_min: `float` lower bound for output.

    Returns:
        `float` of transformed time between 0 and 1.
    """

    v_start = torch.sigmoid(torch.tensor(start / tau))
    v_end = torch.sigmoid(torch.tensor(end / tau))
    output = (-torch.sigmoid((t * (end - start) + start) / tau) + v_end) / (
            v_end - v_start)
    return torch.clamp(output, clip_min, 1.)


def sigmoid_beta_schedule(num_timesteps, start=-3., end=3., tau=1.0):
    betas = betas_for_alpha_bar(
            num_timesteps,
            lambda t: sigmoid_schedule(
                torch.tensor(t), start, end, tau).item()
    )
    return torch.from_numpy(betas).float()


def precompute_schedule(betas):
    """Precompute all closed-form quantities needed for training & sampling."""
    alphas = 1.0 - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_cumprod": alpha_cumprod,
        "alpha_cumprod_prev": alpha_cumprod_prev,
        "sqrt_alpha_cumprod": torch.sqrt(alpha_cumprod),
        "sqrt_one_minus_alpha_cumprod": torch.sqrt(1.0 - alpha_cumprod),
        "sqrt_recip_alpha": torch.sqrt(1.0 / alphas),
        "posterior_variance": betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod),
    }


def get_beta_schedule(schedule_type, num_timesteps, device):
    """Load and compute a beta schedule by name."""
    schedulers = {
        "linear": linear_beta_schedule,
        "cosine": cosine_beta_schedule,
        "sigmoid": sigmoid_beta_schedule,
    }
    if schedule_type not in schedulers:
        raise ValueError(f"Unknown schedule type '{schedule_type}'. Choose from {list(schedulers)}")
    return schedulers[schedule_type](num_timesteps).to(device)


def extract(schedule_tensor, t, x_shape):
    """Gather the precomputed schedule value at timestep t and reshape accordingly."""
    batch_size = t.shape[0]
    out = schedule_tensor.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def forward_diffusion(x_0, t, schedule, noise=None):
    """Returns the noised image x_t and the noise that was added."""
    if noise is None:
        noise = torch.randn_like(x_0)

    sqrt_alpha_cumprod_t = extract(schedule["sqrt_alpha_cumprod"], t, x_0.shape)
    sqrt_one_minus_alpha_cumprod_t = extract(schedule["sqrt_one_minus_alpha_cumprod"], t, x_0.shape)

    x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
    return x_t, noise


@torch.no_grad()
def reverse_diffusion_step(model, x_t, t, t_index, schedule):
    """Returns the denoised image at the previous timestep, while adding noise.""" 
    betas_t = extract(schedule["betas"], t, x_t.shape)
    sqrt_one_minus_alpha_cumprod_t = extract(schedule["sqrt_one_minus_alpha_cumprod"], t, x_t.shape)
    sqrt_recip_alpha_t = extract(schedule["sqrt_recip_alpha"], t, x_t.shape)

    # Predict the mean
    predicted_noise = model(x_t, t)
    mean = sqrt_recip_alpha_t * (x_t - betas_t / sqrt_one_minus_alpha_cumprod_t * predicted_noise)

    if t_index == 0:
        return mean
    else:
        posterior_var_t = extract(schedule["posterior_variance"], t, x_t.shape)
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(posterior_var_t) * noise
