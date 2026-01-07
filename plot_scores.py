import argparse
import csv
import re
import numpy as np
import matplotlib.pyplot as plt


def read_psnr_ssim(csv_path: str):
    psnr, ssim = [], []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)

        for row in reader:
            if not row:
                continue
            if row[0] == "AVG":
                continue

            # Expected columns:
            # label_pred,label_correct,ssim,psnr,final_loss,...
            try:
                s = float(row[2])
                p = float(row[3])
            except Exception:
                continue

            if np.isfinite(s) and np.isfinite(p):
                ssim.append(s)
                psnr.append(p)

    return np.array(psnr), np.array(ssim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="psnr_ssim_scatter.png")
    ap.add_argument("--title", default=None)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    psnr, ssim = read_psnr_ssim(args.csv)
    if len(psnr) == 0:
        print("No valid PSNR/SSIM rows found.")
        return

    # Correlation (Pearson)
    corr = np.corrcoef(psnr, ssim)[0, 1]

    plt.figure(figsize=(7, 5))
    plt.scatter(psnr, ssim, alpha=0.7)
    plt.xlabel("PSNR (dB)")
    plt.ylabel("SSIM")
    plt.xlim(0,40)
    plt.ylim(0,1.2)
    t = args.title or f"PSNR vs SSIM (n={len(psnr)}), Pearson r={corr:.3f}"
    plt.title(t)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    if args.show:
        plt.show()
    plt.close()

    print(f"Saved: {args.out}")
    print(f"n={len(psnr)}, Pearson r={corr:.6f}")


if __name__ == "__main__":
    main()
