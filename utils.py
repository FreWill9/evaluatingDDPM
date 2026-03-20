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
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"{outdir}/loss-UNET_gpu64.svg", format="svg")
    plt.close()
    print(f"Saved loss plot successfully to '{outdir}/loss-UNET_gpu64.svg'!")


def plot_DS_loss(checkpoint_path: str, outdir: str = "loss_plots"):
    checkpoint = torch.load(checkpoint_path)
    loss_history = checkpoint.get("loss_history", [])
    aux_loss_history = checkpoint.get("aux_loss_history", [])
    total_loss_history = checkpoint.get("total_loss_history", [])

    plt.figure(figsize=(8, 5))

    plt.plot(loss_history, label="Main loss")
    plt.plot(aux_loss_history, label="Aux loss")
    plt.plot(total_loss_history, label="Total loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training losses")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{outdir}/loss-UNET64-DS.svg", format="svg")
    plt.close()
    print(f"Saved DS losses plot successfully to '{outdir}/loss-UNET64-DS.svg'!")


def main():
    plot_DS_loss("checkpoints/UNET_gpu64_DS.pt")


if __name__ == "__main__":
    main()
