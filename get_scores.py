from __future__ import annotations
import argparse
import glob
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

# Metrics present in your per-run rows
METRIC_COLS = ["ssim", "psnr", "final_loss"]

# Keys to avoid mixing different experiments
BASE_KEYS = ["scenario", "model_path", "defense"]

# Image-level grouping (for per-image averages)
IMAGE_KEYS = BASE_KEYS + ["img_idx", "label_true"]

# Class-level grouping (for per-class best/worst + averages)
CLASS_KEYS = BASE_KEYS + ["label_true"]


def _expand_inputs(csvs: Optional[List[str]], pattern: Optional[str]) -> List[str]:
    paths: List[str] = []
    if csvs:
        paths.extend(csvs)
    if pattern:
        paths.extend(glob.glob(pattern))
    # de-dupe while preserving order
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def load_and_concat(csv_paths: Iterable[str]) -> pd.DataFrame:
    dfs = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df["source_csv"] = str(p)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No CSV files found / provided.")

    df = pd.concat(dfs, ignore_index=True)

    # Numeric conversions (robust to blanks / AVG rows)
    for c in ["img_idx", "label_true", "repeat_id", "seed", "label_pred", "label_correct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in METRIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep only actual run rows (AVG rows have repeat_id like "AVG" -> NaN after coercion)
    df_runs = df[df["repeat_id"].notna()].copy()
    return df_runs


def per_image_averages(df_runs: pd.DataFrame) -> pd.DataFrame:
    """(1) Average PSNR/SSIM/Loss per image (img_idx + label_true, plus scenario/model/defense)."""
    agg = df_runs.groupby(IMAGE_KEYS, dropna=False)[METRIC_COLS].mean().reset_index()
    agg = agg.rename(columns={"ssim": "avg_ssim", "psnr": "avg_psnr", "final_loss": "avg_loss"})
    counts = df_runs.groupby(IMAGE_KEYS, dropna=False)["repeat_id"].count().reset_index(name="n_runs")
    agg = agg.merge(counts, on=IMAGE_KEYS, how="left")

    # Sort "image-related" outputs by image
    agg = agg.sort_values(IMAGE_KEYS).reset_index(drop=True)
    return agg


def per_class_averages(df_runs: pd.DataFrame) -> pd.DataFrame:
    """Extra: Average PSNR/SSIM/Loss per class/label_true (plus scenario/model/defense)."""
    agg = df_runs.groupby(CLASS_KEYS, dropna=False)[METRIC_COLS].mean().reset_index()
    agg = agg.rename(columns={"ssim": "avg_ssim", "psnr": "avg_psnr", "final_loss": "avg_loss"})
    counts = df_runs.groupby(CLASS_KEYS, dropna=False)["repeat_id"].count().reset_index(name="n_runs")
    agg = agg.merge(counts, on=CLASS_KEYS, how="left")

    # Sort "class-related" outputs by class
    agg = agg.sort_values(CLASS_KEYS).reset_index(drop=True)
    return agg


def overall_averages(df_runs: pd.DataFrame) -> pd.Series:
    """(2) Overall averages across ALL runs."""
    s = df_runs[METRIC_COLS].mean(numeric_only=True)
    return s.rename(
        {"ssim": "overall_avg_ssim", "psnr": "overall_avg_psnr", "final_loss": "overall_avg_loss"}
    )


def best_and_worst_per_class(df_runs: pd.DataFrame) -> pd.DataFrame:
    """
    (3) Best & worst runs per CLASS (label_true, plus scenario/model/defense) for each metric:
      - PSNR: best=max, worst=min
      - SSIM: best=max, worst=min
      - Loss: best=min, worst=max
    """

    def pick_row(g: pd.DataFrame, metric: str, best: bool) -> dict:
        if metric in ["psnr", "ssim"]:
            idx = g[metric].idxmax() if best else g[metric].idxmin()
        elif metric == "final_loss":
            idx = g[metric].idxmin() if best else g[metric].idxmax()
        else:
            raise ValueError(metric)

        row = g.loc[idx]
        return {
            f"{metric}_{'best' if best else 'worst'}_value": row[metric],
            f"{metric}_{'best' if best else 'worst'}_img_idx": row.get("img_idx", pd.NA),
            f"{metric}_{'best' if best else 'worst'}_repeat_id": row.get("repeat_id", pd.NA),
            f"{metric}_{'best' if best else 'worst'}_seed": row.get("seed", pd.NA),
            f"{metric}_{'best' if best else 'worst'}_source_csv": row.get("source_csv", pd.NA),
        }

    out_rows = []
    for keys, g in df_runs.groupby(CLASS_KEYS, dropna=False):
        record = dict(zip(CLASS_KEYS, keys))

        for metric in METRIC_COLS:
            gg = g[g[metric].notna()]
            if gg.empty:
                record.update(
                    {
                        f"{metric}_best_value": pd.NA,
                        f"{metric}_best_img_idx": pd.NA,
                        f"{metric}_best_repeat_id": pd.NA,
                        f"{metric}_best_seed": pd.NA,
                        f"{metric}_best_source_csv": pd.NA,
                        f"{metric}_worst_value": pd.NA,
                        f"{metric}_worst_img_idx": pd.NA,
                        f"{metric}_worst_repeat_id": pd.NA,
                        f"{metric}_worst_seed": pd.NA,
                        f"{metric}_worst_source_csv": pd.NA,
                    }
                )
            else:
                record.update(pick_row(gg, metric, best=True))
                record.update(pick_row(gg, metric, best=False))

        out_rows.append(record)

    res = pd.DataFrame(out_rows)

    # Friendlier column names for loss
    res = res.rename(
        columns={
            "final_loss_best_value": "loss_best_value",
            "final_loss_best_img_idx": "loss_best_img_idx",
            "final_loss_best_repeat_id": "loss_best_repeat_id",
            "final_loss_best_seed": "loss_best_seed",
            "final_loss_best_source_csv": "loss_best_source_csv",
            "final_loss_worst_value": "loss_worst_value",
            "final_loss_worst_img_idx": "loss_worst_img_idx",
            "final_loss_worst_repeat_id": "loss_worst_repeat_id",
            "final_loss_worst_seed": "loss_worst_seed",
            "final_loss_worst_source_csv": "loss_worst_source_csv",
        }
    )

    # Sort "class-related" outputs by class
    res = res.sort_values(CLASS_KEYS).reset_index(drop=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="*", help="CSV files to analyze")
    ap.add_argument("--glob", dest="glob_pattern", help='Glob pattern, e.g. "results/*.csv"')
    ap.add_argument("--outdir", default="analysis_out", help="Directory for output CSVs")
    args = ap.parse_args()

    paths = _expand_inputs(args.csvs, args.glob_pattern)
    if not paths:
        raise SystemExit("No CSVs provided. Use --csvs ... or --glob 'path/*.csv'.")

    df_runs = load_and_concat(paths)

    img_avgs = per_image_averages(df_runs)
    class_avgs = per_class_averages(df_runs)
    overall = overall_averages(df_runs)
    class_extremes = best_and_worst_per_class(df_runs)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # per-image averages (sorted by image)
    img_avgs_path = outdir / "per_image_averages.csv"
    img_avgs.to_csv(img_avgs_path, index=False)

    # per-class averages (sorted by class)
    class_avgs_path = outdir / "per_class_averages.csv"
    class_avgs.to_csv(class_avgs_path, index=False)

    # overall averages
    overall_path = outdir / "overall_averages.csv"
    overall.to_frame(name="value").reset_index(names=["metric"]).to_csv(overall_path, index=False)

    # best/worst per class (sorted by class)
    extremes_path = outdir / "best_worst_per_class.csv"
    class_extremes.to_csv(extremes_path, index=False)

    print("\n=== Overall averages across ALL runs ===")
    print(overall.to_string())

    print(f"\nSaved:\n  {img_avgs_path}\n  {class_avgs_path}\n  {overall_path}\n  {extremes_path}\n")


if __name__ == "__main__":
    main()
