from __future__ import annotations
import argparse
import os
import random
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import tensorflow as tf
import matplotlib.pyplot as plt

from classes.models import LeNet
from classes.attacks import iDLG
from classes.helperfunctions import compute_ssim_psnr


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_cifar10_torch():
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    x_train = torch.tensor(x_train.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
    y_train = torch.tensor(y_train.squeeze(), dtype=torch.long)
    return x_train, y_train


def load_model(state_dict_path: str, device: str):
    model = LeNet()
    sd = torch.load(state_dict_path, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def visualize_run(orig_img, dummy, recon, label_pred, label_true, losses, title, save_path) -> None:
    if isinstance(recon, np.ndarray):
        recon_t = torch.from_numpy(recon)
    else:
        recon_t = recon

    recon_t = recon_t.detach().cpu()
    if recon_t.dim() == 3:
        recon_t = recon_t.unsqueeze(0)

    dummy_t = dummy.detach().cpu()
    if dummy_t.dim() == 3:
        dummy_t = dummy_t.unsqueeze(0)

    orig_t = orig_img.detach().cpu()
    if orig_t.dim() == 3:
        orig_t = orig_t.unsqueeze(0)

    def to_hwc(x):
        x0 = x[0]
        return x0.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    axes[0].imshow(to_hwc(dummy_t))
    axes[0].set_title("Dummy init", fontsize=16)
    axes[0].axis("off")

    axes[1].imshow(to_hwc(recon_t))
    axes[1].set_title(f"Reconstruction\npred={int(label_pred)}", fontsize=16)
    axes[1].axis("off")

    axes[2].imshow(to_hwc(orig_t))
    axes[2].set_title(f"Original\ntrue={label_true}", fontsize=16)
    axes[2].axis("off")

    axes[3].plot(losses)
    axes[3].set_title("Loss curve", fontsize=16)
    axes[3].set_xlabel("Iteration", fontsize=16)
    axes[3].set_ylabel("Loss", fontsize=16)
    axes[3].grid(True)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


REQUIRED_COLS = ["img_idx", "seed", "ssim", "psnr", "final_loss", "defense"]


def load_run_rows(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    if "repeat_id" in df.columns:
        df["repeat_id_num"] = pd.to_numeric(df["repeat_id"], errors="coerce")
        df = df[df["repeat_id_num"].notna()].copy()

    # Coerce numeric
    for c in ["img_idx", "seed", "ssim", "psnr", "final_loss"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["img_idx", "seed", "ssim", "psnr", "final_loss"]).copy()

    df["img_idx"] = df["img_idx"].astype(int)
    df["seed"] = df["seed"].astype(int)

    return df


def find_csv_row_for_run(df: pd.DataFrame, img_idx: int, seed: int, defense: str) -> Optional[pd.Series]:
    sub = df[(df["img_idx"] == img_idx) & (df["seed"] == seed) & (df["defense"] == defense)]
    if sub.empty:
        return None
    if len(sub) > 1:
        print("Warning: multiple matching CSV rows found; using the first.")
    return sub.iloc[0]


def rerun_one(
    model,
    orig_img,
    label_true: int,
    device: str,
    defense: str,
    tvr: str,
    percentile: Optional[float],
    iterations: int,
    seed: int,
) -> Dict:
    set_all_seeds(seed)

    label = torch.tensor([label_true], dtype=torch.long, device=device)
    
    tvr = float(tvr)

    attacker = iDLG(
        model=model,
        label=label,
        seed=seed,
        clamp=(0.0, 1.0),
        device=device,
        orig_img=orig_img,
        grads=None,
        defense=defense,
        tvr=tvr,
        percentile=percentile,
        random_dummy=False,
        dummy_var=0.1,
    )

    def_save, dummy, recon, label_pred, history, losses = attacker.attack(iterations=iterations)

    if isinstance(recon, np.ndarray):
        recon_t = torch.from_numpy(recon).to(device=device, dtype=orig_img.dtype)
        if recon_t.dim() == 3:
            recon_t = recon_t.unsqueeze(0)
    else:
        recon_t = recon.to(device)

    ssim, psnr = compute_ssim_psnr(orig_img, recon_t)

    return {
        "seed": seed,
        "dummy": dummy,
        "recon": recon_t.detach().cpu(),
        "label_pred": int(label_pred.item()) if hasattr(label_pred, "item") else int(label_pred),
        "ssim": float(ssim),
        "psnr": float(psnr),
        "losses": losses,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to model .pt state_dict")
    ap.add_argument("--img_idx", type=int, required=True, help="Image index to rerun")
    ap.add_argument("--seed", type=int, required=True, help="Seed to rerun")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--tvr", default="3e-7")

    ap.add_argument("--defense", default="none", choices=["none", "normclipping", "clipping", "sgp"], help="Defense to apply")
    ap.add_argument("--percentile", default=None, type=float, help="clipping quantile or sgp keep_ratio")
    ap.add_argument("--out_dir", default="viz_one")

    ap.add_argument("--csv", default=None, help="Optional CSV to print stored psnr/ssim for this img_idx+seed+defense")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)
    x_train, y_train = load_cifar10_torch()

    img_idx = int(args.img_idx)
    seed = int(args.seed)
    defense = str(args.defense)

    if img_idx < 0 or img_idx >= len(x_train):
        raise ValueError(f"--img_idx out of range: {img_idx} (valid: 0..{len(x_train)-1})")

    orig_img = x_train[img_idx].unsqueeze(0).to(device)
    label_true = int(y_train[img_idx].item())

    csv_row = None
    if args.csv is not None:
        df = load_run_rows(args.csv)
        csv_row = find_csv_row_for_run(df, img_idx=img_idx, seed=seed, defense=defense)

    out = rerun_one(
        model=model,
        orig_img=orig_img,
        label_true=label_true,
        device=device,
        defense=defense,
        tvr=args.tvr,
        percentile=args.percentile,
        iterations=args.iterations,
        seed=seed,
    )

    pct_str = "None" if args.percentile is None else str(args.percentile)
    title = (
        f"img={img_idx} | seed={seed} | TVR={args.tvr} | label={label_true} | "
        f"def={defense}({pct_str}) | PSNR={out['psnr']:.3f} | SSIM={out['ssim']:.3f}"
    )

    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, f"img{img_idx}_seed{seed}_tvr{args.tvr}_it{args.iterations}.png")
    visualize_run(
        orig_img,
        out["dummy"],
        out["recon"],
        out["label_pred"],
        label_true,
        out["losses"],
        title,
        save_path,
    )

    if csv_row is not None:
        print(
            "csv:  "
            f"psnr={float(csv_row['psnr']):.6f}, "
            f"ssim={float(csv_row['ssim']):.6f}, "
            f"final_loss={float(csv_row['final_loss']):.6f}"
        )
    elif args.csv is not None:
        print("csv:  no matching row found for this img_idx+seed+defense")

    print(f"rerun: psnr={out['psnr']:.6f}, ssim={out['ssim']:.6f}")
    print(f"Saved plot to: {save_path}")


if __name__ == "__main__":
    main()
