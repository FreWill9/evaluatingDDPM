import torch
import torch.nn as nn


class DDPM_Scheduler(nn.Module):
    def __init__(self, num_timesteps: int = 1000):
        super().__init__()
        beta = torch.linspace(1e-4, 0.02, num_timesteps)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)          # per-step alpha_t
        self.register_buffer("alpha_bar", alpha_bar)  # cumulative product

    def forward(self, t):
        return self.beta[t], self.alpha[t], self.alpha_bar[t]
