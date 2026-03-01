"""
scripts/run_predictions.py

PE model prediction runner (risk at horizons), using pred_grid.

Correct approach:
- Build a prediction exposure grid up to each horizon t0 (subject × interval grid with y=interval width,
  truncated at t0), then integrate hazard to get subject-level cumhaz and risk.

Inputs:
- --model:     fitted model JSON
- --base_csv:  non-long CSV (one row per subject) with id + covariates (+ time/event optional)
- --long_csv:  observed long CSV (train/test) used to derive breaks and restrict cohort IDs
- --out_prefix: prefix for output files

Outputs:
- {out_prefix}_subject_risk.csv
- {out_prefix}_summary_by_group.csv (if --group_col provided)
- {out_prefix}_risk_hist_t{t0}.png (optional)
- {out_prefix}_risk_by_group_box_t{t0}.png (optional, if group_col)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pe.predict import predict_long
from utils.pred_grid import build_pred_long_from_long


def parse_floats_csv(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def subject_risk_at_horizon(
    base_sub: pd.DataFrame,
    long_df_for_breaks: pd.DataFrame,
    model: dict,
    t0: float,
    *,
    id_col: str = "id",
) -> pd.DataFrame:
    """
    Returns per-subject predictions at horizon t0:
      id, _pred_cumhaz_t{t0}, _pred_risk_t{t0}
    plus covariates from base_sub.
    """
    pred_long, _breaks = build_pred_long_from_long(base_sub, long_df_for_breaks, t0=t0)

    pred, _ = predict_long(pred_long, model, return_X=False)
    if not isinstance(pred, pd.DataFrame) or "hazard" not in pred.columns:
        raise TypeError("predict_long must return a DataFrame with a 'hazard' column.")

    haz = pred["hazard"].to_numpy(dtype=float)
    y = pred_long["y"].to_numpy(dtype=float)
    inc = y * haz

    tmp = pd.DataFrame({id_col: pred_long[id_col].to_numpy(dtype=int), "_inc": inc})
    cumhaz = tmp.groupby(id_col)["_inc"].sum()
    risk = 1.0 - np.exp(-cumhaz)

    out = base_sub.copy()
    out[f"_pred_cumhaz_t{int(t0)}"] = out[id_col].map(cumhaz).to_numpy(dtype=float)
    out[f"_pred_risk_t{int(t0)}"] = out[id_col].map(risk).to_numpy(dtype=float)
    return out


def plot_risk_hist(
    subj: pd.DataFrame,
    t0: float,
    out_png: Path,
    *,
    risk_col: str,
    title: str,
) -> None:
    plt.figure()
    plt.hist(subj[risk_col].to_numpy(dtype=float), bins=30)
    plt.xlabel("Predicted risk")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_risk_by_group_box(
    subj: pd.DataFrame,
    t0: float,
    out_png: Path,
    *,
    risk_col: str,
    group_col: str,
    title: str,
) -> None:
    # Box plot by group (simple, readable)
    groups = []
    labels = []
    for gval, gdf in subj.groupby(group_col):
        groups.append(gdf[risk_col].to_numpy(dtype=float))
        labels.append(str(gval))

    plt.figure()
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.xlabel(group_col)
    plt.ylabel("Predicted risk")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to fitted model JSON.")
    ap.add_argument("--base_csv", required=True, help="Non-long CSV: one row per subject with id + covariates.")
    ap.add_argument("--long_csv", required=True, help="Observed long CSV (train/test) used to derive breaks + cohort.")
    ap.add_argument("--out_prefix", required=True, help="Output prefix (path without extension).")
    ap.add_argument("--horizons", default="365,725,1825", help="Comma-separated horizons in days.")
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--group_col", default=None, help="Optional grouping column for summaries/plots (e.g. stage).")
    ap.add_argument("--make_plots", default=True, action="store_true", help="If set, write simple risk distribution plots.")
    args = ap.parse_args()

    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    base = pd.read_csv(args.base_csv)
    long_df = pd.read_csv(args.long_csv)

    if args.id_col not in base.columns:
        raise ValueError(f"base_csv missing '{args.id_col}'")
    if args.id_col not in long_df.columns:
        raise ValueError(f"long_csv missing '{args.id_col}'")

    base[args.id_col] = pd.to_numeric(base[args.id_col], errors="raise").astype(int)
    long_df[args.id_col] = pd.to_numeric(long_df[args.id_col], errors="raise").astype(int)

    # Restrict to cohort IDs present in long_df (train/test split)
    cohort_ids = np.sort(long_df[args.id_col].unique())
    base_sub = base[base[args.id_col].isin(cohort_ids)].copy()
    print(f"Unique ids: base_sub={base_sub[args.id_col].nunique():,}, long={len(cohort_ids):,}")

    horizons = parse_floats_csv(args.horizons)

    # Build per-horizon predictions and merge into a single subject-level table
    subj = base_sub.copy()
    for t0 in horizons:
        tmp = subject_risk_at_horizon(subj, long_df, model, t0, id_col=args.id_col)
        # tmp includes all columns; merge only new prediction columns
        new_cols = [c for c in tmp.columns if c.startswith("_pred_") and c.endswith(f"t{int(t0)}")]
        for c in new_cols:
            subj[c] = tmp[c].to_numpy()

        # Optional plots
        if args.make_plots:
            out_prefix = Path(args.out_prefix)
            out_prefix.parent.mkdir(parents=True, exist_ok=True)

            risk_col = f"_pred_risk_t{int(t0)}"
            out_hist = out_prefix.parent / f"{out_prefix.name}_risk_hist_t{int(t0)}.png"
            plot_risk_hist(
                subj=subj,
                t0=t0,
                out_png=out_hist,
                risk_col=risk_col,
                title=f"Predicted risk distribution (t={int(t0)} days)",
            )
            print(f"Wrote: {out_hist}")

            if args.group_col is not None:
                if args.group_col not in subj.columns:
                    raise ValueError(f"group_col='{args.group_col}' not found in base_csv columns.")
                out_box = out_prefix.parent / f"{out_prefix.name}_risk_by_{args.group_col}_box_t{int(t0)}.png"
                plot_risk_by_group_box(
                    subj=subj,
                    t0=t0,
                    out_png=out_box,
                    risk_col=risk_col,
                    group_col=args.group_col,
                    title=f"Predicted risk by {args.group_col} (t={int(t0)} days)",
                )
                print(f"Wrote: {out_box}")

    # Write subject-level predictions
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    out_csv = out_prefix.parent / f"{out_prefix.name}_subject_risk.csv"
    subj.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")

    # Optional group summary table
    if args.group_col is not None:
        if args.group_col not in subj.columns:
            raise ValueError(f"group_col='{args.group_col}' not found in base_csv columns.")

        rows = []
        for gval, gdf in subj.groupby(args.group_col):
            row = {args.group_col: gval, "n": int(len(gdf))}
            for t0 in horizons:
                rc = f"_pred_risk_t{int(t0)}"
                row[f"mean_risk_t{int(t0)}"] = float(gdf[rc].mean())
                row[f"p10_risk_t{int(t0)}"] = float(gdf[rc].quantile(0.10))
                row[f"p50_risk_t{int(t0)}"] = float(gdf[rc].quantile(0.50))
                row[f"p90_risk_t{int(t0)}"] = float(gdf[rc].quantile(0.90))
            rows.append(row)

        summ = pd.DataFrame(rows).sort_values(args.group_col).reset_index(drop=True)
        out_summ = out_prefix.parent / f"{out_prefix.name}_summary_by_{args.group_col}.csv"
        summ.to_csv(out_summ, index=False)
        print(f"Wrote: {out_summ}")

    print("Predictions complete.")


if __name__ == "__main__":
    main()