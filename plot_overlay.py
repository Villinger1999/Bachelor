import argparse
import csv
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

            try:
                s = float(row[2])  # ssim
                p = float(row[3])  # psnr
            except Exception:
                continue

            if np.isfinite(s) and np.isfinite(p):
                ssim.append(s)
                psnr.append(p)

    return np.array(psnr), np.array(ssim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True,
                    help="List of CSVs to overlay")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Optional labels (same count as csvs). If omitted, filenames are used.")
    ap.add_argument("--out", default="psnr_ssim_multi.png")
    ap.add_argument("--title", default="PSNR vs SSIM comparison")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    labels = args.labels
    if labels is None:
        labels = args.csvs
    if len(labels) != len(args.csvs):
        raise ValueError("If --labels is provided, it must have same length as --csvs")

    plt.figure(figsize=(8, 6))

    for csv_path, label in zip(args.csvs, labels):
        psnr, ssim = read_psnr_ssim(csv_path)
        if len(psnr) == 0:
            print(f"WARNING: no points in {csv_path}")
            continue

        corr = np.corrcoef(psnr, ssim)[0, 1]
        plt.scatter(psnr, ssim, alpha=0.6, label=f"{label} (n={len(psnr)}, r={corr:.2f})")

    plt.xlabel("PSNR (dB)")
    plt.ylabel("SSIM")
    plt.title(args.title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    if args.show:
        plt.show()
    plt.close()

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
