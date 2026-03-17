import torch
import torch.nn as nn
import matplotlib.pyplot as plt


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


def plot_loss(checkpoint_path: str, outdir: str = "loss_plots"):
    checkpoint = torch.load(checkpoint_path)
    loss_history = checkpoint.get("loss_history", [])
    print(loss_history)
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{outdir}/loss-UNET_gpu64.svg", format="svg")
    plt.close()
    print(f"Saved image successfully to '{outdir}/loss-UNET_gpu64.svg'!")


def main():
    plot_loss("checkpoints/UNET_gpu64.pt")


if __name__ == "__main__":
    main()
