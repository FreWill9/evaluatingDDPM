import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
from diffusion_utils import get_beta_schedule, precompute_schedule, forward_diffusion


def plot_forward_diffusion(
    sample_image,
    schedule_linear,
    schedule_cosine,
    num_timesteps,
    num_steps=10,
    device="cpu",
):
    style = 'fivethirtyeight'
    plt.style.use(style)
    timesteps = torch.linspace(0, num_timesteps - 1, num_steps, dtype=torch.long).to(device)
    sample_image = sample_image.to(device)

    fig, axes = plt.subplots(2, num_steps, figsize=(num_steps * 2, 5))

    for row, (schedule, label) in enumerate(
        [(schedule_linear, "Linear"), (schedule_cosine, "Cosine")]
    ):
        for col, t in enumerate(timesteps):
            noisy_image, _ = forward_diffusion(sample_image.unsqueeze(0), t.unsqueeze(0), schedule)
            img = noisy_image.squeeze().cpu().numpy()

            axes[row, col].imshow(img, cmap="gray", vmin=-1, vmax=1)
            axes[row, col].axis("off")

            if col == 0:
                axes[row, col].set_title(label, fontsize=20, loc="left", pad=4)
    # Plot timesteps
    for col, t in enumerate(timesteps):
        ax = axes[-1, col]
        ax.annotate(
            f"t={t.item()}",
            xy=(0.5, 0),
            xycoords="axes fraction",
            xytext=(0, -6),
            textcoords="offset points",
            ha="center", va="top",
            fontsize=12,
        )
    plt.tight_layout()
    outdir = "plots"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    out_path = os.path.join(outdir, "fid_vs_schedulers.png")
    plt.savefig(out_path, dpi=300)
    plt.show()


def plot_fid_kid_vs_num_train_samples(csv_path="metrics.csv"):
    # Dual axis plot to show both FID and KID for comparison
    style = 'fivethirtyeight'
    plt.style.use(style)
    figsize = (10, 6)

    df = pd.read_csv(csv_path)

    x = df["num_train_samples"]
    fid = df["fid"]
    kid = df["kid_mean"]

    fig, ax1 = plt.subplots(figsize=figsize)

    label_fontsize = 30
    legend_fontsize = 20
    tick_fontsize = 22

    xticks = [1000, 5000, 10000, 20000]
    ax1.set_xticks(xticks)

    # Plot FID
    color_fid = 'tab:blue'
    ax1.set_xlabel('Training Set Size', fontsize=tick_fontsize)
    ax1.set_ylabel('FID', color=color_fid, fontsize=label_fontsize)
    ax1.plot(x, fid, marker='o', color=color_fid, label='FID')
    ax1.tick_params(axis='y', labelcolor=color_fid, labelsize=tick_fontsize)
    ax1.set_ylim(28, 87) # Set y-axis limits for better visualization
    ax1.tick_params(axis='x', labelsize=tick_fontsize)

    # Add KID to twin axis
    ax2 = ax1.twinx()
    ax2.grid(False)
    color_kid = 'tab:orange'
    ax2.set_ylabel('KID', color=color_kid, fontsize=label_fontsize)
    ax2.plot(x, kid, marker='s', color=color_kid, label='KID (mean)')
    ax2.tick_params(axis='y', labelcolor=color_kid, labelsize=tick_fontsize)

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best', fontsize=legend_fontsize)
    legend.set_zorder(10)

    #plt.title("FID and KID vs Number of Training Samples", fontsize=30)
    plt.tight_layout()

    outdir = "plots"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    out_path = os.path.join(outdir, "fid_kid_vs_datasetsize.png")
    plt.savefig(out_path, dpi=300)
    plt.show()


def plot_fid_vs_schedulers(csv_path="metrics_scheduler.csv"):
    style = 'fivethirtyeight'
    plt.style.use(style)
    figsize = (10, 6)

    df = pd.read_csv(csv_path)
    df_sorted = df.sort_values('num_train_samples')

    df_linear = df_sorted[df_sorted['schedule_type'] == 'linear']
    df_cosine = df_sorted[df_sorted['schedule_type'] == 'cosine']

    fig, ax = plt.subplots(figsize=figsize)

    label_fontsize = 30
    legend_fontsize = 20
    tick_fontsize = 22

    xticks = [1000, 5000, 10000, 20000]
    ax.set_xticks(xticks)

    color_linear = 'tab:blue'
    color_cosine = 'tab:orange'

    ax.plot(df_linear['num_train_samples'], df_linear['fid'],
            marker='o', color=color_linear, label='Linear')
    ax.plot(df_cosine['num_train_samples'], df_cosine['fid'],
            marker='s', color=color_cosine, label='Cosine')

    ax.set_xlabel('Training Set Size', fontsize=label_fontsize)
    ax.set_ylabel('FID', fontsize=label_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.tick_params(axis='x', labelsize=tick_fontsize)
    # ax.set_ylim(28, 87)  # adjust once you know your data range

    legend = ax.legend(loc='best', fontsize=legend_fontsize)
    legend.set_zorder(10)

    plt.tight_layout()

    outdir = "plots"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    out_path = os.path.join(outdir, "fid_vs_schedulers.png")
    plt.savefig(out_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    #plot_fid_kid_vs_num_train_samples()
    #plot_fid_vs_schedulers()

    # Plot forward diffusion process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = "data/celeba_gray64_20000.pt"
    sample_image = torch.load(image_path)[0]
    num_timesteps = 1000
    schedule_linear = precompute_schedule(get_beta_schedule("linear", num_timesteps, device))
    schedule_cosine = precompute_schedule(get_beta_schedule("cosine", num_timesteps, device))
    plot_forward_diffusion(
        sample_image,
        schedule_linear,
        schedule_cosine,
        num_timesteps=num_timesteps,
        num_steps=8,
        device=device,
    )