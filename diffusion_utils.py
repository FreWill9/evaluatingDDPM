import torch
import torch.nn.functional as F


def linear_beta_schedule(num_timesteps, beta_start=0.0001, beta_end=0.02):
    """Linear variance schedule from DDPM paper."""
    return torch.linspace(beta_start, beta_end, num_timesteps)


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
