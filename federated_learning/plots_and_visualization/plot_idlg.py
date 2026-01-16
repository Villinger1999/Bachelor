import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from sympy import symbols

csv1 = "results/scores/results_FL_orig_1.csv"
csv2 = "results/scores/results_FL_orig_none_2.csv"

if len(sys.argv) < 2:
    raise SystemExit("Usage: python plot.py <img_idx>")

img_idx = int(sys.argv[1])  
seeds_10 = [123,124,125,126,127,128,129,130,131,132]

out_dir = Path("results/cum_scores/tvr")
out_dir.mkdir(parents=True, exist_ok=True)

def prep_df(df):
    # Ensure numeric dtypes (helps if pandas read them as strings)
    df = df.copy()
    df["img_idx"] = pd.to_numeric(df["img_idx"], errors="coerce")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df["ssim"] = pd.to_numeric(df["ssim"], errors="coerce")
    df["psnr"] = pd.to_numeric(df["psnr"], errors="coerce")
    return df

def select_rows(df, img_idx, seeds):
    out = df[(df["img_idx"] == img_idx) & (df["seed"].isin(seeds))].copy()
    return out.sort_values("seed")

df1 = prep_df(pd.read_csv(csv1))
df2 = prep_df(pd.read_csv(csv2))

d1 = select_rows(df1, img_idx, seeds_10)
d2 = select_rows(df2, img_idx, seeds_10)

# Debug (so you can see what's happening)
print(f"Known img_idx in CSV1 (sample): {sorted(df1['img_idx'].dropna().unique())[:10]}")
print(f"Known img_idx in CSV2 (sample): {sorted(df2['img_idx'].dropna().unique())[:10]}")
print(f"d1 rows: {len(d1)}, d2 rows: {len(d2)}")
print("d1 seeds:", d1["seed"].tolist())
print("d2 seeds:", d2["seed"].tolist())


TVR_DARK  = "#1f77b4"   # with TVR
TVR_LIGHT = "#7fb3d5"   # without TVR
NO_LIGHT = "#f3a6c8"   # without TVR
NO_DARK  = "#e83e8c"   # with TVR


fig, ax_ssim = plt.subplots(figsize=(7,4))
ax_psnr = ax_ssim.twinx()

# SSIM 
l1, = ax_ssim.plot(
    d1["seed"], d1["ssim"],
    color=TVR_DARK, linestyle="-", marker="o",
    label="SSIM (with TVR)"
)
l2, = ax_ssim.plot(
    d2["seed"], d2["ssim"],
    color=NO_DARK, linestyle="-", marker="o",
    label="SSIM (without TVR)"
)
seeds = sorted(set(d1["seed"]).union(d2["seed"]))

ax_ssim.set_xlabel("Seed")
ax_ssim.set_ylabel("SSIM")
# Example fixed limits (adjust if needed)
SSIM_LIM = (0.0, 1.1)
PSNR_LIM = (0, 42)
ax_ssim.set_ylim(*SSIM_LIM)
ax_psnr.set_ylim(*PSNR_LIM)
ax_ssim.set_xticks(seeds)
ax_ssim.tick_params(axis="y")
ax_ssim.grid(True, alpha=0.3)

# PSNR 
l3, = ax_psnr.plot(
    d1["seed"], d1["psnr"],
    color=TVR_LIGHT, linestyle="--", marker="s",
    label="PSNR (with TVR)"
)
l4, = ax_psnr.plot(
    d2["seed"], d2["psnr"],
    color=NO_LIGHT, linestyle="--", marker="s",
    label="PSNR (without TVR)"
)

ax_psnr.set_ylabel("PSNR")
ax_psnr.tick_params(axis="y")

# Combined legend 
lines = [l1, l2, l3, l4]
labels = [l.get_label() for l in lines]
ax_ssim.legend(lines, labels, loc="best", frameon=False)

plt.title(f"SSIM & PSNR vs Seed (img_idx={img_idx})")

plt.savefig(
    out_dir / f"ssim_psnr_twin_img{img_idx}.png",
    dpi=300, bbox_inches="tight"
)
plt.close()

delta = symbols("Delta")

def plot_different(csv, labels, xlabel, ylabel1, ylabel2, title, rotation, out):

    df = pd.read_csv(csv)

    labels = labels
    x = range(len(labels))

    ssim = df["overall_avg_ssim"].astype(float).to_list()
    psnr = df["overall_avg_psnr"].astype(float).to_list()
    
    if "delta_acc_pp" in df:
        model = df["delta_acc_pp"].astype(float).to_list()
    else:
        model = None

    plt.figure()
    bars = plt.bar(
        x,
        ssim,
        color="lightseagreen",
        width=0.6,
        edgecolor="black",
        linewidth=0.9
    )

    plt.xticks(x, labels, rotation=rotation, ha="right")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel1)
    plt.title(f"SSIM for reconstructions {title}")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",         
            ha="center",
            va="bottom",
            fontsize=12
        )

    plt.tight_layout()
    plt.savefig(f"{out}/SSIM.png", dpi=300)
    plt.close()


    plt.figure()
    bars = plt.bar(
        x,
        psnr,
        color="lightseagreen",
        width=0.6,
        edgecolor="black",
        linewidth=0.8
    )

    plt.xticks(x, labels, rotation=rotation, ha="right")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel2)
    plt.title(f"PSNR for reconstructions {title}")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=11
        )
        
    plt.tight_layout()
    plt.savefig(f"{out}/PSNR.png", dpi=300)
    plt.close()

    if model is not None:
        plt.figure()
        bars = plt.bar(
            x,
            model,
            color="lightseagreen",
            width=0.6,
            edgecolor="black",
            linewidth=0.9
        )

        plt.xticks(x, labels, rotation=rotation, ha="right")
        plt.xlabel(xlabel)
        plt.ylabel(f"{delta} acc")
        plt.ylim(0,-3)
        plt.title(f"{delta} model accuracy {title}")
        plt.grid(True, axis="y", linestyle="--", linewidth=0.5)

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=11
            )

        plt.tight_layout()
        plt.savefig(f"{out}/model.png", dpi=300)
        plt.close()


def per_class(csv, out):

    df = pd.read_csv(csv)

    labels = df["label_true"]
    x = range(len(labels))

    ssim = df["avg_ssim"].astype(float).to_list()
    psnr = df["avg_psnr"].astype(float).to_list()

    plt.figure()
    bars = plt.bar(
        x,
        ssim,
        color="lightseagreen",
        width=0.6,
        edgecolor="black",
        linewidth=0.9
    )

    plt.xticks(x, labels, ha="right")
    plt.xlabel("Class label")
    plt.ylabel("SSIM")
    plt.title("Overall SSIM averages per class")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",         
            ha="center",
            va="bottom",
            fontsize=11
        )

    plt.tight_layout()
    plt.savefig(f"{out}/SSIM_perclass.png", dpi=300)
    plt.close()


    plt.figure()
    bars = plt.bar(
        x,
        psnr,
        color="lightseagreen",
        width=0.6,
        edgecolor="black",
        linewidth=0.9
    )

    plt.xticks(x, labels, ha="right")
    plt.xlabel("Class label")
    plt.ylabel("PSNR (dB)")
    plt.title("Overall PSNR averages per class")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=11
        )

    plt.tight_layout()
    plt.savefig(f"{out}/PSNR_perclass", dpi=300)
    plt.close()
    
    
plot_different("results/cum_scores/tvr/tvr.csv", ["None", "1e-5", "1e-6", "5e-7", "4e-7", "3e-7", "2e-7", "1e-7"], rotation=45, xlabel="TVR", ylabel1="SSIM", ylabel2="PSNR", title="for different TVR settings", out="results/cum_scores/tvr")
plot_different(
    "results/cum_scores/scores_sgp/sgp.csv",
    ["0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2"],
    rotation=45,
    xlabel="Percentile",
    ylabel1="SSIM",
    ylabel2="PSNR",
    title="for different SGP threshold",
    out="results/cum_scores/scores_sgp"
)

plot_different(
    "results/cum_scores/scores_clip/clip.csv",
    ["0.999", "0.998", "0.997", "0.996", "0.995", "0.994", "0.993", "0.99"],
    rotation=45,
    xlabel="Percentile",
    ylabel1="SSIM",
    ylabel2="PSNR",
    title="for different clipping thresholds",
    out="results/cum_scores/scores_clip"
)

plot_different(
    "results/cum_scores/norm_clip/normclip.csv",
    ["0.995", "0.99", "0.98", "0.95", "0.93", "0.9", "0.88", "0.85"],
    rotation=45,
    xlabel="Percentile",
    ylabel1="SSIM",
    ylabel2="PSNR",
    title="for different norm clipping thresholds",
    out="results/cum_scores/norm_clip"
)

per_class("results/cum_scores/scores_FL/per_class_averages.csv", "results/cum_scores/scores_FL")
per_class("results/cum_scores/per_class_averages.csv", "scores")
per_class("results/cum_scores/scores_clip/per_class_averages.csv", "results/cum_scores/scores_clip")
per_class("results/cum_scores/scores_sgp/per_class_averages.csv", "results/cum_scores/scores_sgp")
per_class("results/cum_scores/norm_clip/per_class_averages.csv", "results/cum_scores/norm_clip")


