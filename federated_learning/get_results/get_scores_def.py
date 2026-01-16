from __future__ import annotations
import argparse
import glob
from pathlib import Path
from typing import List, Optional

import pandas as pd

METRIC_COLS = ["ssim", "psnr", "final_loss","label_correct"]
KEYS = ["scenario", "model_path", "defense"]


def _expand_inputs(csvs: Optional[List[str]], pattern: Optional[str]) -> List[str]:
    paths: List[str] = []
    if csvs:
        paths.extend(csvs)
    if pattern:
        paths.extend(glob.glob(pattern))
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _first_nonnull(series: pd.Series):
    s = series.dropna()
    return s.iloc[0] if len(s) else pd.NA


def summarize_one_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Coerce repeat_id
    if "repeat_id" in df.columns:
        df["repeat_id"] = pd.to_numeric(df["repeat_id"], errors="coerce")

    # Coerce numeric columns
    for c in METRIC_COLS + ["model_acc_before", "model_acc_after", "threshold"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Use only run rows for metric averages
    run_rows = df[df["repeat_id"].notna()].copy() if "repeat_id" in df.columns else df.copy()

    # Mean metrics
    metrics = (
        run_rows.groupby(KEYS, dropna=False)[METRIC_COLS]
        .mean()
        .rename(
            columns={
                "ssim": "overall_avg_ssim",
                "psnr": "overall_avg_psnr",
                "final_loss": "overall_avg_loss",
                "label_correct": "label_correct"
            }
        )
        .reset_index()
    )

    # Extract threshold + accuracies 
    extra = (
        df.groupby(KEYS, dropna=False)
        .agg(
            threshold=("threshold", _first_nonnull),
            acc_before=("model_acc_before", _first_nonnull),
            acc_after=("model_acc_after", _first_nonnull),
        )
        .reset_index()
    )

    out = metrics.merge(extra, on=KEYS, how="left")

    # Delta accuracy in percentage points
    out["delta_acc_pp"] = (out["acc_after"] - out["acc_before"]) * 100
    out["delta_acc_pp"] = out["delta_acc_pp"].round(2)

    # Rounding for readability
    out["overall_avg_ssim"] = out["overall_avg_ssim"].round(6)
    out["overall_avg_psnr"] = out["overall_avg_psnr"].round(6)
    out["overall_avg_loss"] = out["overall_avg_loss"].round(9)
    

    # Sort by threshold 
    out = out.sort_values("threshold", ascending=False).reset_index(drop=True)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="*", help="CSV files to analyze (processed one-by-one)")
    ap.add_argument("--glob", dest="glob_pattern", help='Glob pattern, e.g. "results/*.csv"')
    ap.add_argument("--out", default=None, help="Optional combined output CSV path")
    args = ap.parse_args()

    paths = _expand_inputs(args.csvs, args.glob_pattern)
    if not paths:
        raise SystemExit("No CSVs provided.")

    all_summaries = []

    for p in paths:
        summary = summarize_one_csv(p)

        # Add source file column
        summary.insert(0, "source_csv", Path(p).name)

        # Console output 
        print("\n" + "=" * 80)
        print(f"CSV: {p}")
        print("=" * 80)
        print(summary.to_string(index=False))

        all_summaries.append(summary)

    # Combine all summaries into one DataFrame 
    combined = pd.concat(all_summaries, ignore_index=True)

    # Save single CSV if requested 
    if args.out:
        out_path = Path(args.out)
        combined.to_csv(out_path, index=False)
        print(f"\nSaved combined results to: {out_path}")

if __name__ == "__main__":
    main()