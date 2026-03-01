"""
scripts/run_calibration_risk.py

Risk calibration for the Piecewise Exponential (PE) model at fixed horizons.

Key correction vs the old approach:
- DO NOT compute predicted risk from the *observed* long dataframe (which is truncated at censor/event time).
- Instead, build a *prediction exposure grid* up to each horizon t0 (subject × interval grid with y = interval width
  truncated at t0), then integrate hazard to get subject-level cumulative hazard and risk.

Inputs:
- --model:     fitted model JSON
- --base_csv:  non-long CSV (one row per subject) with id, time, event, and covariates
- --long_csv:  observed long CSV (train or test) used only to (a) restrict cohort IDs and (b) derive breaks
- --out_dir:   output directory to write CSV + plots
Optional:
- --group_col: categorical grouping column (e.g., stage) for stratified calibration plots

Outputs:
- calibration_risk_bins_t{t0}.csv
- calibration_risk_bins_t{t0}.png
- If group_col provided:
    calibration_risk_bins_{group_col}_t{t0}.csv
    calibration_risk_bins_{group_col}_t{t0}.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pe.predict import predict_long
from utils.pred_grid import build_pred_long_from_long


def km_risk_at_horizon(times: np.ndarray, events: np.ndarray, t0: float) -> float:
    """
    Kaplan–Meier risk at horizon t0: 1 - S_hat(t0).
    times/events are arrays for a cohort/subgroup.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)

    # event times up to t0
    event_times = np.unique(times[(events == 1) & (times <= t0)])
    if event_times.size == 0:
        return 0.0  # no events by t0 => risk 0

    S = 1.0
    for tj in event_times:
        n_risk = np.sum(times >= tj)
        d = np.sum((times == tj) & (events == 1))
        if n_risk <= 0:
            continue
        S *= (1.0 - d / n_risk)

    return float(1.0 - S)


def quantile_bins(risk: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Return integer bin labels [0..B-1] using qcut, dropping duplicate edges.
    """
    r = pd.Series(np.asarray(risk, dtype=float))
    # If constant or near-constant, qcut can fail; handle gracefully.
    if float(r.max() - r.min()) < 1e-12:
        return np.zeros(len(r), dtype=int)

    # duplicates="drop" avoids failures when ties prevent exact quantiles
    bins = pd.qcut(r, q=n_bins, labels=False, duplicates="drop")
    return bins.to_numpy(dtype=int)


def build_subject_risk(
    base_sub: pd.DataFrame,
    long_df_for_breaks: pd.DataFrame,
    model: dict,
    t0: float,
    *,
    id_col: str,
    time_col: str,
    event_col: str,
) -> pd.DataFrame:
    """
    Build per-subject predicted risk at horizon t0 using a prediction grid (not observed long).

    Returns a dataframe with:
      id, time, event, _pred_risk, _pred_cumhaz
      plus all covariates from base_sub
    """
    # Build prediction grid up to t0 (subject × intervals up to t0) using breaks derived from long_df_for_breaks
    pred_long, _breaks = build_pred_long_from_long(base_sub, long_df_for_breaks, t0=t0)

    # Predict per-row hazard (per day)
    pred, _ = predict_long(pred_long, model, return_X=False)
    if not isinstance(pred, pd.DataFrame) or "hazard" not in pred.columns:
        raise TypeError("predict_long must return a DataFrame with a 'hazard' column for this script.")

    haz = pred["hazard"].to_numpy(dtype=float)
    y = pred_long["y"].to_numpy(dtype=float)

    inc = y * haz
    tmp = pd.DataFrame({id_col: pred_long[id_col].to_numpy(dtype=int), "_inc": inc})
    cumhaz = tmp.groupby(id_col)["_inc"].sum()
    risk = 1.0 - np.exp(-cumhaz)

    out = base_sub.copy()
    out["_pred_cumhaz"] = out[id_col].map(cumhaz).to_numpy(dtype=float)
    out["_pred_risk"] = out[id_col].map(risk).to_numpy(dtype=float)

    # Ensure time/event exist
    if time_col not in out.columns or event_col not in out.columns:
        raise ValueError(f"base_csv must include '{time_col}' and '{event_col}'")
    return out


def calibration_table(
    subj_df: pd.DataFrame,
    t0: float,
    *,
    id_col: str,
    time_col: str,
    event_col: str,
    n_bins: int,
) -> pd.DataFrame:
    """
    Build calibration table by predicted-risk quantile bins.
    Columns:
      bin, n, mean_pred, km_risk, diff, ratio
    """
    df = subj_df.copy()
    bins = quantile_bins(df["_pred_risk"].to_numpy(dtype=float), n_bins=n_bins)
    df["bin"] = bins

    rows = []
    for b in sorted(df["bin"].unique()):
        g = df[df["bin"] == b]
        km = km_risk_at_horizon(g[time_col].to_numpy(), g[event_col].to_numpy(), t0)
        mean_pred = float(g["_pred_risk"].mean())
        rows.append(
            {
                "bin": int(b),
                "n": int(len(g)),
                "mean_pred": mean_pred,
                "km_risk": float(km),
                "diff": float(km - mean_pred),
                "ratio_obs_over_pred": float(km / mean_pred) if mean_pred > 0 else np.nan,
                "pred_q_lo": float(g["_pred_risk"].quantile(0.0)),
                "pred_q_hi": float(g["_pred_risk"].quantile(1.0)),
            }
        )

    out = pd.DataFrame(rows).sort_values("bin").reset_index(drop=True)
    return out


def plot_calibration(
    tab: pd.DataFrame,
    out_png: Path,
    *,
    title: str,
) -> None:
    """
    Plot observed (KM) vs expected (mean predicted) by bin.
    """
    x = tab["mean_pred"].to_numpy(dtype=float)
    y = tab["km_risk"].to_numpy(dtype=float)

    plt.figure()
    plt.plot(x, y, marker="o")
    # 45-degree reference
    lo = float(np.nanmin([x.min(), y.min()]))
    hi = float(np.nanmax([x.max(), y.max()]))
    pad = 0.02 * (hi - lo + 1e-12)
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad])

    plt.xlabel("Mean predicted risk (bin)")
    plt.ylabel("Observed KM risk (bin)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to fitted model JSON.")
    ap.add_argument("--base_csv", required=True, help="Non-long CSV with id,time,event and covariates.")
    ap.add_argument("--long_csv", required=True, help="Observed long CSV (train/test) used to derive breaks and cohort.")
    ap.add_argument("--out_dir", required=True, help="Output directory for calibration CSV/plots.")
    ap.add_argument("--horizons", default="365,725,1825", help="Comma-separated horizons in days.")
    ap.add_argument("--n_bins", type=int, default=10, help="Number of quantile bins.")
    ap.add_argument("--group_col", default=None, help="Optional grouping column (e.g., stage) for stratified plots.")
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--time_col", default="time")
    ap.add_argument("--event_col", default="event")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    base = pd.read_csv(args.base_csv)
    long_df = pd.read_csv(args.long_csv)

    # Cohort restriction: only IDs present in long_csv (train/test split)
    if args.id_col not in base.columns:
        raise ValueError(f"base_csv missing '{args.id_col}'")
    if args.id_col not in long_df.columns:
        raise ValueError(f"long_csv missing '{args.id_col}'")

    cohort_ids = np.sort(pd.to_numeric(long_df[args.id_col], errors="raise").astype(int).unique())
    base[args.id_col] = pd.to_numeric(base[args.id_col], errors="raise").astype(int)
    base_sub = base[base[args.id_col].isin(cohort_ids)].copy()

    print(f"Unique ids: base_sub={base_sub[args.id_col].nunique():,}, long={len(cohort_ids):,}")

    horizons = [float(x) for x in args.horizons.split(",") if x.strip()]
    for t0 in horizons:
        # Build per-subject predicted risk using prediction grid up to t0
        subj = build_subject_risk(
            base_sub=base_sub,
            long_df_for_breaks=long_df,
            model=model,
            t0=t0,
            id_col=args.id_col,
            time_col=args.time_col,
            event_col=args.event_col,
        )

        # Overall calibration table + plot
        tab = calibration_table(
            subj_df=subj,
            t0=t0,
            id_col=args.id_col,
            time_col=args.time_col,
            event_col=args.event_col,
            n_bins=args.n_bins,
        )

        out_csv = out_dir / f"calibration_risk_bins_t{int(t0)}.csv"
        tab.to_csv(out_csv, index=False)
        print(f"Wrote: {out_csv}")

        out_png = out_dir / f"calibration_risk_bins_t{int(t0)}.png"
        plot_calibration(tab, out_png, title=f"Risk calibration by bins (t={int(t0)} days)")
        print(f"Wrote: {out_png}")

        # Stratified calibration
        if args.group_col is not None:
            if args.group_col not in subj.columns:
                raise ValueError(f"group_col='{args.group_col}' not found in base_csv columns.")

            tabs = []
            # Make one plot with one curve per group (binning within group)
            plt.figure()
            for gval, gdf in subj.groupby(args.group_col):
                tab_g = calibration_table(
                    subj_df=gdf,
                    t0=t0,
                    id_col=args.id_col,
                    time_col=args.time_col,
                    event_col=args.event_col,
                    n_bins=args.n_bins,
                )
                tab_g[args.group_col] = gval
                tabs.append(tab_g)

                plt.plot(tab_g["mean_pred"].to_numpy(float), tab_g["km_risk"].to_numpy(float), marker="o", label=str(gval))

            # 45-degree reference
            all_x = np.concatenate([tg["mean_pred"].to_numpy(float) for tg in tabs]) if tabs else np.array([0.0, 1.0])
            all_y = np.concatenate([tg["km_risk"].to_numpy(float) for tg in tabs]) if tabs else np.array([0.0, 1.0])
            lo = float(np.nanmin([all_x.min(), all_y.min()]))
            hi = float(np.nanmax([all_x.max(), all_y.max()]))
            pad = 0.02 * (hi - lo + 1e-12)
            plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad])

            plt.xlabel("Mean predicted risk (bin)")
            plt.ylabel("Observed KM risk (bin)")
            plt.title(f"Risk calibration by bins, stratified by {args.group_col} (t={int(t0)} days)")
            plt.legend()
            plt.tight_layout()

            out_png_g = out_dir / f"calibration_risk_bins_{args.group_col}_t{int(t0)}.png"
            plt.savefig(out_png_g, dpi=150)
            plt.close()
            print(f"Wrote: {out_png_g}")

            out_csv_g = out_dir / f"calibration_risk_bins_{args.group_col}_t{int(t0)}.csv"
            pd.concat(tabs, ignore_index=True).to_csv(out_csv_g, index=False)
            print(f"Wrote: {out_csv_g}")

    print("Calibration (risk) complete.")

if __name__ == "__main__":
    main()