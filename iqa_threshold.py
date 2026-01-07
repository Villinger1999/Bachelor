import argparse
import os
import csv
import re
import random
import numpy as np
import torch
import tensorflow as tf
import matplotlib.pyplot as plt

import lpips

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------
# CIFAR-10 train
# ----------------------------
def load_cifar10_train():
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    x_train = torch.tensor(x_train.transpose(0, 3, 1, 2), dtype=torch.float32) / 255.0
    y_train = torch.tensor(y_train.squeeze(), dtype=torch.long)
    return x_train, y_train


# ----------------------------
# Model
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


def looks_like_float(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


# ----------------------------
# Read CSV and infer repeat_id from row order (skip AVG row)
# Robustly skips scenario/header rows like:
#   normal_model_orig_grads, model_path, model_acc, defense, img_idx, label_true
# ----------------------------
def read_runs(csv_path: str):
    runs = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        has_lpips = False
        if header:
            header_lower = [h.strip().lower() for h in header]
            has_lpips = "lpips" in header_lower

        for row in reader:
            if not row:
                continue

            # Skip AVG summary row
            if row[0].strip() == "AVG":
                continue

            # Skip scenario/meta rows (common in your other script)
            # They usually have non-numeric columns where SSIM/PSNR should be.
            # We expect row[2] and row[3] to be floats for valid run rows.
            if len(row) < 5:
                continue
            if not (looks_like_float(row[2]) and looks_like_float(row[3])):
                continue

            # Now parse as run row
            label_pred = parse_label_pred(row[0])
            label_correct = int(float(row[1]))
            ssim = float(row[2])
            psnr = float(row[3])

            if has_lpips:
                lp = float(row[4])
                final_loss = float(row[5]) if len(row) > 5 and looks_like_float(row[5]) else float("nan")
            else:
                lp = None
                final_loss = float(row[4]) if looks_like_float(row[4]) else float("nan")

            runs.append({
                "repeat_id": len(runs),  # inferred by order of VALID run rows
                "label_pred": label_pred,
                "label_correct": label_correct,
                "ssim": ssim,
                "psnr": psnr,
                "lpips": lp,
                "final_loss": final_loss,
            })
    return runs


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
    val = lpips_fn(x, y)  # [1,1,1,1]
    return float(val.item())


# ----------------------------
# Visualization (4 panels)
# ----------------------------
def visualize(orig_img, dummy, recon, label_pred, label_true, losses, title, save_path, lpips_val=None):
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
# Save tensors (orig/dummy/recon) for each rerun
# ----------------------------
# def save_tensor_triplet(out_dir, stem, orig_img, dummy, recon):
#     os.makedirs(out_dir, exist_ok=True)
#     payload = {
#         "orig": orig_img.detach().cpu(),
#         "dummy": dummy.detach().cpu(),
#         "recon": recon.detach().cpu() if torch.is_tensor(recon) else torch.from_numpy(np.array(recon)).detach().cpu(),
#     }
#     torch.save(payload, os.path.join(out_dir, f"{stem}.pt"))


# ----------------------------
# Re-run iDLG deterministically for a given repeat_id
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
        grads=None,
        defense=defense,
        percentile=percentile,
        random_dummy=True,
        dummy_var=0.0,
    )

    def_save, dummy, recon, label_pred, history, losses = attacker.attack(iterations=iterations)

    # recon -> tensor (1,C,H,W)
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


def safe_fname(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--img_idx", type=int, required=True)
    ap.add_argument("--base_seed", type=int, required=True)

    ap.add_argument("--iterations", type=int, default=100)

    ap.add_argument("--min_ssim", type=float, default=0.75)
    ap.add_argument("--min_psnr", type=float, default=22.0)
    ap.add_argument("--max_ssim", type=float, default=1.0)
    ap.add_argument("--max_psnr", type=float, default=1e9)

    ap.add_argument("--defense", default="none", choices=["none", "clipping", "sgp"])
    ap.add_argument("--percentile", type=float, default=None)

    ap.add_argument("--out_dir", default="viz_thresholds")
    ap.add_argument("--lpips_net", default="vgg", choices=["vgg", "alex", "squeeze"])

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)

    lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device).eval()

    x_train, y_train = load_cifar10_train()
    orig_img = x_train[args.img_idx].unsqueeze(0).to(device)
    label_true = int(y_train[args.img_idx].item())

    runs = read_runs(args.csv)
    if not runs:
        print("No valid run rows found in CSV. (It may contain only AVG/meta rows.)")
        return

    # Print global maxima from CSV
    max_ssim_csv = max(r["ssim"] for r in runs)
    max_psnr_csv = max(r["psnr"] for r in runs)
    print(f"CSV maxima (runs only): max SSIM={max_ssim_csv:.6f}, max PSNR={max_psnr_csv:.6f}")

    # Select ALL runs within min/max
    selected = [
        r for r in runs
        if (args.min_ssim <= r["ssim"] <= args.max_ssim) or (args.min_psnr <= r["psnr"] <= args.max_psnr)
    ]
    # Optional sorting
    selected.sort(key=lambda r: (r["psnr"], r["ssim"]), reverse=True)

    print(
        f"Selected {len(selected)} runs with "
        f"SSIM in [{args.min_ssim}, {args.max_ssim}] and "
        f"PSNR in [{args.min_psnr}, {args.max_psnr}]"
    )

    if not selected:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    img_dir = os.path.join(args.out_dir, "png")
    # pt_dir = os.path.join(args.out_dir, "pt")
    os.makedirs(img_dir, exist_ok=True)
    # os.makedirs(pt_dir, exist_ok=True)

    # Save an index file for convenience
    index_path = os.path.join(args.out_dir, "selected_runs.txt")
    with open(index_path, "w") as f:
        for r in selected:
            f.write(f"repeat_id={r['repeat_id']}, ssim={r['ssim']:.6f}, psnr={r['psnr']:.6f}\n")

    # Track maxima among selected (CSV)
    max_sel_ssim_csv = max(r["ssim"] for r in selected)
    max_sel_psnr_csv = max(r["psnr"] for r in selected)
    print(f"Selected (CSV) maxima: max SSIM={max_sel_ssim_csv:.6f}, max PSNR={max_sel_psnr_csv:.6f}")

    # Re-run and plot EVERY selected
    best_out = None
    for k, r in enumerate(selected):
        rep_id = r["repeat_id"]
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
            lpips_fn=lpips_fn,
        )

        if best_out is None or (out["psnr"], out["ssim"]) > (best_out["psnr"], best_out["ssim"]):
            best_out = out

        title = (
            f"img={args.img_idx} rep={rep_id} seed={out['seed']} | "
            f"def={args.defense}({args.percentile}) | PSNR={out['psnr']:.3f} | SSIM={out['ssim']:.3f}"
        )
        if out["lpips"] is not None:
            title += f" | LPIPS={out['lpips']:.4f}"

        stem = safe_fname(
            f"rep{rep_id}_seed{out['seed']}_psnr{out['psnr']:.3f}_ssim{out['ssim']:.3f}_lpips{out['lpips']:.4f}"
        )

        png_path = os.path.join(img_dir, f"{stem}.png")
        # pt_path_stem = os.path.join(pt_dir, stem)  # save_tensor_triplet adds .pt

        visualize(
            orig_img=orig_img,
            dummy=out["dummy"],
            recon=out["recon"],
            label_pred=out["label_pred"],
            label_true=label_true,
            losses=out["losses"],
            title=title,
            save_path=png_path,
            lpips_val=out["lpips"],
        )

        # save_tensor_triplet(pt_dir, stem, orig_img, out["dummy"], out["recon"])

        if (k + 1) % 10 == 0:
            print(f"Saved {k+1}/{len(selected)}")

    # Print maxima among re-runs
    if best_out is not None:
        print(
            f"Selected (re-run) maxima: max PSNR={best_out['psnr']:.6f}, max SSIM={best_out['ssim']:.6f}, "
            f"LPIPS(best)={best_out['lpips']:.6f} "
            f"(rep_id={best_out['repeat_id']}, seed={best_out['seed']})"
        )

    print(f"Done. Output saved in: {args.out_dir}")
    print(f"PNGs: {img_dir}")
    # print(f"Tensors: {pt_dir}")
    print(f"Index file: {index_path}")


if __name__ == "__main__":
    main()
