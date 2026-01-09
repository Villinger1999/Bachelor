import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

csv1_path = "results_FL_orig_1.csv"
csv2_path = "results_FL_orig_none_2.csv"

if len(sys.argv) < 2:
    raise SystemExit("Usage: python plot.py <img_idx>")

img_idx = int(sys.argv[1])  
seeds_10 = [123,124,125,126,127,128,129,130,131,132]

out_dir = Path("scores/tvr")
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

df1 = prep_df(pd.read_csv(csv1_path))
df2 = prep_df(pd.read_csv(csv2_path))

d1 = select_rows(df1, img_idx, seeds_10)
d2 = select_rows(df2, img_idx, seeds_10)

# Debug (so you can see what's happening)
print(f"Known img_idx in CSV1 (sample): {sorted(df1['img_idx'].dropna().unique())[:10]}")
print(f"Known img_idx in CSV2 (sample): {sorted(df2['img_idx'].dropna().unique())[:10]}")
print(f"d1 rows: {len(d1)}, d2 rows: {len(d2)}")
print("d1 seeds:", d1["seed"].tolist())
print("d2 seeds:", d2["seed"].tolist())

if d1.empty and d2.empty:
    raise SystemExit("No rows matched your filters. Check img_idx and seeds_10.")

# SSIM plot
plt.figure(figsize=(8,6))
if not d1.empty:
    plt.plot(d1["seed"], d1["ssim"], marker="o", label="with TVR")
if not d2.empty:
    plt.plot(d2["seed"], d2["ssim"], marker="o", label="without TVR")

plt.xlabel("Seed")
plt.ylabel("SSIM")
plt.title(f"SSIM vs Seed (img_idx={img_idx})")
plt.grid(True)
plt.legend()
plt.savefig(out_dir / f"ssim_vs_seed_img{img_idx}.png", dpi=300, bbox_inches="tight")
plt.close()

# PSNR plot
plt.figure(figsize=(6,4))
if not d1.empty:
    plt.plot(d1["seed"], d1["psnr"], marker="o", label="with TVR")
if not d2.empty:
    plt.plot(d2["seed"], d2["psnr"], marker="o", label="without TVR")

plt.xlabel("Seed")
plt.ylabel("PSNR")
plt.title(f"PSNR vs Seed (img_idx={img_idx})")
plt.grid(True)
plt.legend()
plt.savefig(out_dir / f"psnr_vs_seed_img{img_idx}.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved:\n  {out_dir / f'ssim_vs_seed_img{img_idx}.png'}\n  {out_dir / f'psnr_vs_seed_img{img_idx}.png'}")

TVR_DARK  = "#1f77b4"   # with TVR
TVR_LIGHT = "#7fb3d5"   # without TVR
NO_LIGHT = "#f3a6c8"   # without TVR
NO_DARK  = "#e83e8c"   # with TVR

# --------------------
# Twin plot: SSIM + PSNR
# --------------------
fig, ax_ssim = plt.subplots(figsize=(7,4))
ax_psnr = ax_ssim.twinx()

# ---- SSIM (left axis, blue) ----
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
SSIM_LIM = (0.5, 0.95)
PSNR_LIM = (12, 30)
ax_ssim.set_ylim(*SSIM_LIM)
ax_psnr.set_ylim(*PSNR_LIM)
ax_ssim.set_xticks(seeds)
ax_ssim.tick_params(axis="y")
ax_ssim.grid(True, alpha=0.3)

# ---- PSNR (right axis, pink) ----
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

# ---- Combined legend ----
lines = [l1, l2, l3, l4]
labels = [l.get_label() for l in lines]
ax_ssim.legend(lines, labels, loc="best", frameon=False)

plt.title(f"SSIM & PSNR vs Seed (img_idx={img_idx})")

plt.savefig(
    out_dir / f"ssim_psnr_twin_img{img_idx}.png",
    dpi=300, bbox_inches="tight"
)
plt.close()
