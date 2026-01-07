import argparse
import os
import csv
import numpy as np
import torch
import tensorflow as tf
import random
import matplotlib.pyplot as plt

from classes.models import LeNet
from classes.attacks import iDLG

# If your compute_ssim_psnr expects tensors, use it.
# Otherwise we compute here using skimage.
from classes.helperfunctions import compute_ssim_psnr


# ----------------------------
# Reproducibility
# ----------------------------
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Data / model loading
# ----------------------------
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


# ----------------------------
# Visualization (dummy, recon, orig, loss)
# ----------------------------
def visualize_run(orig_img, dummy, recon, label_pred, label_true, losses, title, save_path):
    """
    orig_img: (1,C,H,W) torch
    dummy:    (1,C,H,W) torch
    recon:    torch (1,C,H,W) or (C,H,W) or numpy
    label_pred: torch scalar or int
    label_true: int
    losses: list[float]
    """
    # recon to tensor on cpu
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

    # convert to HWC
    def to_hwc(x):
        x = x[0]  # (C,H,W)
        return x.permute(1, 2, 0).numpy()

    dummy_hwc = to_hwc(dummy_t)
    recon_hwc = to_hwc(recon_t)
    orig_hwc  = to_hwc(orig_t)

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    axes[0].imshow(dummy_hwc)
    axes[0].set_title("Dummy init")
    axes[0].axis("off")

    axes[1].imshow(recon_hwc)
    axes[1].set_title(f"Reconstruction\npred={int(label_pred)}")
    axes[1].axis("off")

    axes[2].imshow(orig_hwc)
    axes[2].set_title(f"Original\ntrue={label_true}")
    axes[2].axis("off")

    axes[3].plot(losses)
    axes[3].set_title("Loss curve")
    axes[3].set_xlabel("Iteration")
    axes[3].set_ylabel("Loss")
    axes[3].grid(True)

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# CSV parsing
# ----------------------------
def read_runs_from_csv(csv_path):
    """
    Your csv rows look like:
    label_pred,label_correct,ssim,psnr,final_loss,...
    We need repeat_id/seed to do it perfectly, but you said you can compute seed from repeat_id.
    If your CSV does NOT store repeat_id, you must either:
      - store it going forward, OR
      - infer order: line index = repeat_id (works if strictly one row per repeat in order).
    This function returns a list of dicts with repeat_id inferred from row order.
    """
    runs = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # skip "AVG" summary rows or headers
            if row[0] in ("label_pred", "AVG"):
                continue

            # row format example:
            # tensor([6]),1,0.9936,36.66,0.00368,...
            # label_pred may be "tensor([6])" -> parse int from it
            label_pred_str = row[0]
            try:
                # extract digits inside tensor([x])
                lp = int(label_pred_str.replace("tensor([", "").replace("])", "").strip())
            except:
                # fallback
                try:
                    lp = int(label_pred_str)
                except:
                    lp = -1

            label_correct = int(float(row[1]))
            ssim = float(row[2])
            psnr = float(row[3])
            final_loss = float(row[4])

            runs.append({
                "repeat_id": len(runs),  # infer by order
                "label_pred": lp,
                "label_correct": label_correct,
                "ssim": ssim,
                "psnr": psnr,
                "final_loss": final_loss,
            })
    return runs


def pick_extremes(runs):
    # best/worst psnr and ssim
    best_psnr = max(runs, key=lambda r: r["psnr"])
    worst_psnr = min(runs, key=lambda r: r["psnr"])
    best_ssim = max(runs, key=lambda r: r["ssim"])
    worst_ssim = min(runs, key=lambda r: r["ssim"])
    return {
        "best_psnr": best_psnr,
        "worst_psnr": worst_psnr,
        "best_ssim": best_ssim,
        "worst_ssim": worst_ssim,
    }


# ----------------------------
# Re-run a single repeat deterministically
# ----------------------------
def rerun_one(model, orig_img, label_true, device, base_seed, img_idx, repeat_id,
              defense, percentile, iterations):
    # IMPORTANT: you used K=1000
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
        grads=None,               # assuming orig_grads scenario
        defense=defense,
        percentile=percentile,
        random_dummy=True,
        dummy_var=0.0,
    )

    def_save, dummy, recon, label_pred, history, losses = attacker.attack(iterations=iterations)

    # ensure recon is tensor for compute_ssim_psnr
    if isinstance(recon, np.ndarray):
        recon_t = torch.from_numpy(recon).to(device=device, dtype=orig_img.dtype)
        if recon_t.dim() == 3:
            recon_t = recon_t.unsqueeze(0)
    else:
        recon_t = recon.to(device)

    ssim, psnr = compute_ssim_psnr(orig_img, recon_t)

    return {
        "seed": seed,
        "repeat_id": repeat_id,
        "dummy": dummy,
        "recon": recon_t.detach().cpu(),
        "label_pred": int(label_pred.item()) if hasattr(label_pred, "item") else int(label_pred),
        "ssim": float(ssim),
        "psnr": float(psnr),
        "losses": losses,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV file containing per-run rows in order")
    ap.add_argument("--model", required=True, help="path to model .pt state_dict")
    ap.add_argument("--img_idx", type=int, default=0)
    ap.add_argument("--base_seed", type=int, default=123)
    ap.add_argument("--iterations", type=int, default=100)

    ap.add_argument("--defense", default="none", choices=["none","clipping","sgp"])
    ap.add_argument("--percentile", default=None, type=float, help="clipping quantile or sgp keep_ratio")
    ap.add_argument("--out_dir", default="viz_extremes")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x_train, y_train = load_cifar10_torch()
    orig_img = x_train[args.img_idx].unsqueeze(0).to(device)
    label_true = int(y_train[args.img_idx].item())

    model = load_model(args.model, device)

    runs = read_runs_from_csv(args.csv)
    extremes = pick_extremes(runs)

    # Re-run only the 4 interesting repeats
    for tag, r in extremes.items():
        rep_id = r["repeat_id"]
        out = rerun_one(
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
        )

        title = (
            f"{tag} | img={args.img_idx} | rep={rep_id} | seed={out['seed']} | "
            f"def={args.defense}({args.percentile}) | PSNR={out['psnr']:.3f} | SSIM={out['ssim']:.3f}"
        )
        save_path = os.path.join(args.out_dir, f"{tag}_img{args.img_idx}_rep{rep_id}.png")
        visualize_run(orig_img, out["dummy"], out["recon"], out["label_pred"], label_true, out["losses"], title, save_path)

    print(f"Saved 4 plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
