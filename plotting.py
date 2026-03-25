import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_fid_kid_vs_num_train_samples(csv_path="metrics.csv"):
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


if __name__ == "__main__":
    plot_fid_kid_vs_num_train_samples()