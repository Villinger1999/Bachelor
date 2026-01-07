import argparse
import os
import csv
import re
import random
import numpy as np
import torch
import tensorflow as tf
import matplotlib.pyplot as plt

import lpips  # NEW

from classes.models import LeNet
from classes.attacks import iDLG
from classes.helperfunctions import compute_ssim_psnr


# ----------------------------
# Determinism
# ----------------------------
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optional (GPU determinism):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Load CIFAR-10 (train only)
# ----------------------------
def load_cifar10_train():
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    x_train = torch.tensor(x_train.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
    y_train = torch.tensor(y_train.squeeze(), dtype=torch.long)
    return x_train, y_train


# ----------------------------
# Load model
# ----------------------------
def load_model(state_dict_path: str, device: str):
    model = LeNet()
    sd = torch.load(state_dict_path, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


# ----------------------------
# Parse "tensor([6])" -> 6
# ----------------------------
def parse_label_pred(s: str) -> int:
    m = re.search(r"(-?\d+)", s)
    return int(m.group(1)) if m else -1


# ----------------------------
# Read CSV (infer repeat_id by row order, skip AVG row)
# Supports BOTH formats:
# 1) old: label_pred,label_correct,ssim,psnr,final_loss,...
# 2) new: label_pred,label_correct,ssim,psnr,lpips,final_loss,...
# ----------------------------
def read_runs(csv_path: str):
    runs = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        # Detect if LPIPS column exists based on header (preferred)
        has_lpips = False
        if header:
            header_lower = [h.strip().lower() for h in header]
            has_lpips = "lpips" in header_lower

        for row in reader:
            if not row:
                continue
            if row[0] == "AVG":
                continue

            # expected:
            # old: label_pred,label_correct,ssim,psnr,final_loss
            # new: label_pred,label_correct,ssim,psnr,lpips,final_loss
            label_pred = parse_label_pred(row[0])
            label_correct = int(float(row[1]))
            ssim = float(row[2])
            psnr = float(row[3])

            if has_lpips:
                lp = float(row[4])
                final_loss = float(row[5])
            else:
                lp = None
                final_loss = float(row[4])

            runs.append({
                "repeat_id": len(runs),  # inferred by order
                "label_pred": label_pred,
                "label_correct": label_correct,
                "ssim": ssim,
                "psnr": psnr,
                "lpips": lp,
                "final_loss": final_loss,
            })

    return runs


# ----------------------------
# Pick one representative run in a bin
# Strategy: choose run closest to the bin midpoint (stable selection).
# ----------------------------
def pick_one_in_bin(runs, metric: str, lo: float, hi: float):
    in_bin = [r for r in runs if (r[metric] >= lo and r[metric] < hi)]
    if not in_bin:
        return None

    mid = (lo + hi) / 2.0
    in_bin.sort(key=lambda r: abs(r[metric] - mid))
    return in_bin[0]


# ----------------------------
# LPIPS helper
# ----------------------------
@torch.no_grad()
def compute_lpips(orig_img: torch.Tensor, recon_img: torch.Tensor, lpips_fn) -> float:
    """
    orig_img, recon_img: [1,3,H,W] in [0,1]
    LPIPS expects [-1,1]
    returns scalar float (lower is better)
    """
    x = orig_img * 2.0 - 1.0
    y = recon_img * 2.0 - 1.0
    val = lpips_fn(x, y)  # shape [1,1,1,1]
    return float(val.item())


# ----------------------------
# Visualization: dummy / recon / orig / loss curve
# Adds LPIPS to the title line in the recon panel.
# ----------------------------
def visualize(orig_img, dummy, recon, label_pred, label_true, losses, title, save_path, lpips_val=None):
    # Ensure tensors on CPU
    orig = orig_img.detach().cpu()
    dum  = dummy.detach().cpu()
    rec  = recon.detach().cpu() if torch.is_tensor(recon) else torch.from_numpy(recon).detach().cpu()

    if orig.dim() == 3: orig = orig.unsqueeze(0)
    if dum.dim() == 3:  dum  = dum.unsqueeze(0)
    if rec.dim() == 3:  rec  = rec.unsqueeze(0)

    def to_hwc(x):
        x = x[0]  # (C,H,W)
        return x.permute(1, 2, 0).numpy()

    orig_hwc = to_hwc(orig)
    dum_hwc  = to_hwc(dum)
    rec_hwc  = to_hwc(rec)

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    axes[0].imshow(dum_hwc)
    axes[0].set_title("Dummy init")
    axes[0].axis("off")

    axes[1].imshow(rec_hwc)
    if lpips_val is None:
        axes[1].set_title(f"Reconstruction\npred={label_pred}")
    else:
        axes[1].set_title(f"Reconstruction\npred={label_pred}\nLPIPS={lpips_val:.4f}")
    axes[1].axis("off")

    axes[2].imshow(orig_hwc)
    axes[2].set_title(f"Original\ntrue={label_true}")
    axes[2].axis("off")

    axes[3].plot(losses)
    axes[3].set_title("Loss curve")
    axes[3].set_xlabel("Iteration")
    axes[3].set_ylabel("Loss")
    axes[3].grid(True)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Re-run exactly one repeat_id deterministically
# ----------------------------
def rerun_idlg(model, orig_img, label_true, device, base_seed, img_idx, repeat_id,
              defense, percentile, iterations, lpips_fn=None):
    seed = base_seed + 1000 * img_idx + repeat_id
    set_all_seeds(seed)

    label = torch.tensor([label_true], dtype=torch.long, device=device)

    attacker = iDLG(
        model=model,
        label=label,
        seed=seed,
        clamp=(0.0, 1.0),
        device=device,
        orig_img=orig_img,
        grads=None,            # assumes orig gradients scenario
        defense=defense,
        percentile=percentile,
        random_dummy=True,
        dummy_var=0.0,
    )

    def_save, dummy, recon, label_pred, history, losses = attacker.attack(iterations=iterations)

    # recon might already be tensor on CPU or GPU depending on your iDLG; make it tensor (1,C,H,W)
    if torch.is_tensor(recon):
        recon_t = recon
        if recon_t.dim() == 3:
            recon_t = recon_t.unsqueeze(0)
    else:
        recon_t = torch.from_numpy(np.array(recon))
        if recon_t.dim() == 3:
            recon_t = recon_t.unsqueeze(0)

    recon_t = recon_t.to(device=device, dtype=orig_img.dtype)

    ssim, psnr = compute_ssim_psnr(orig_img, recon_t)

    lp = None
    if lpips_fn is not None:
        lp = compute_lpips(orig_img, recon_t, lpips_fn)

    return {
        "seed": seed,
        "repeat_id": repeat_id,
        "dummy": dummy,
        "recon": recon_t.detach().cpu(),
        "label_pred": int(label_pred.item()) if hasattr(label_pred, "item") else int(label_pred),
        "ssim": float(ssim),
        "psnr": float(psnr),
        "lpips": None if lp is None else float(lp),
        "losses": losses,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--img_idx", type=int, required=True)
    ap.add_argument("--base_seed", type=int, required=True)
    ap.add_argument("--iterations", type=int, default=100)

    ap.add_argument("--defense", default="none", choices=["none", "clipping", "sgp"])
    ap.add_argument("--percentile", type=float, default=None)
    ap.add_argument("--out_dir", default="viz_bins")

    # NEW: allow choosing LPIPS backbone
    ap.add_argument("--lpips_net", default="vgg", choices=["vgg", "alex", "squeeze"])

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)

    # NEW: LPIPS model (for display + reporting)
    lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device).eval()

    x_train, y_train = load_cifar10_train()
    orig_img = x_train[args.img_idx].unsqueeze(0).to(device)
    label_true = int(y_train[args.img_idx].item())

    runs = read_runs(args.csv)

    # bins you requested (interpreted correctly)
    ssim_bins = [(0.95, 1.00), (0.90, 0.95), (0.85, 0.90), (0.80, 0.85), (0.75, 0.80), (0.70, 0.75)]
    psnr_bins = [(20, 25), (25, 30), (30, 35), (35, 40)]

    picked = []

    for lo, hi in ssim_bins:
        r = pick_one_in_bin(runs, "ssim", lo, hi)
        if r is None:
            print(f"[WARN] No run found in SSIM bin [{lo}, {hi})")
            continue
        picked.append(("SSIM", lo, hi, r["repeat_id"], r["ssim"], r["psnr"], r.get("lpips", None)))

    for lo, hi in psnr_bins:
        r = pick_one_in_bin(runs, "psnr", lo, hi)
        if r is None:
            print(f"[WARN] No run found in PSNR bin [{lo}, {hi})")
            continue
        picked.append(("PSNR", lo, hi, r["repeat_id"], r["ssim"], r["psnr"], r.get("lpips", None)))

    # re-run and save
    for metric, lo, hi, rep_id, ssim_csv, psnr_csv, lpips_csv in picked:
        out = rerun_idlg(
            model=model,
            orig_img=orig_img,
            label_true=label_true,
            device=device,
            base_seed=args.base_seed,
            img_idx=args.img_idx,
            repeat_id=rep_id,
            defense=args.defense,
            percentile=args.percentile,
            iterations=args.iterations,
            lpips_fn=lpips_fn,  # NEW
        )

        # Prefer re-computed LPIPS; fall back to CSV value if for some reason missing
        lp_show = out["lpips"] if out["lpips"] is not None else lpips_csv

        title = (
            f"{metric} bin [{lo},{hi}) | img={args.img_idx} | rep={rep_id} | seed={out['seed']} | "
            f"def={args.defense}({args.percentile}) | PSNR={out['psnr']:.3f} | SSIM={out['ssim']:.3f}"
        )
        if lp_show is not None:
            title += f" | LPIPS={lp_show:.4f}"

        fname = f"{metric}_{lo}_{hi}_img{args.img_idx}_rep{rep_id}.png".replace(".", "p")
        save_path = os.path.join(args.out_dir, fname)

        visualize(
            orig_img=orig_img,
            dummy=out["dummy"],
            recon=out["recon"],
            label_pred=out["label_pred"],
            label_true=label_true,
            losses=out["losses"],
            title=title,
            save_path=save_path,
            lpips_val=lp_show,  # NEW
        )

        print(f"Saved: {save_path}")

    print(f"Done. Output dir: {args.out_dir}")


if __name__ == "__main__":
    main()
