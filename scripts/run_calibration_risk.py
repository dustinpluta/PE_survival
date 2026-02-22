from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pe.predict import predict_long  # uses model formula + patsy


def parse_floats_csv(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def ensure_id_in_input(base_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    If id_col not in input CSV, create it as 0..n-1 using row order.
    This must match the row-order used to create __row_id__ during expand_long.
    """
    if id_col in base_df.columns:
        return base_df
    base_df = base_df.copy()
    base_df[id_col] = np.arange(len(base_df), dtype=int)
    return base_df


def km_survival_at(t0: float, times: np.ndarray, events: np.ndarray) -> float:
    """
    Kaplan–Meier survival estimate S(t0).
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)

    event_times = np.unique(times[(events == 1) & (times <= t0)])
    if event_times.size == 0:
        return 1.0

    S = 1.0
    for tj in event_times:
        n_risk = np.sum(times >= tj)
        d = np.sum((times == tj) & (events == 1))
        if n_risk <= 0:
            break
        S *= (1.0 - d / n_risk)
    return float(S)


def horizon_to_k(breaks: np.ndarray, t0: float) -> int:
    """
    Map horizon t0 to the interval index k0 such that breaks[k0+1] == t0.
    Requires t0 to be present in breaks.
    """
    idx = np.where(np.isclose(breaks, t0))[0]
    if idx.size == 0:
        raise ValueError(f"horizon t0={t0} must be contained in breaks.")
    j = int(idx[0])
    if j == 0:
        raise ValueError("horizon cannot equal breaks[0].")
    return j - 1


def quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Robust quantile binning using rank to avoid qcut duplicate-edge failures.
    Returns integer bin labels 0..n_bins-1.
    """
    r = pd.Series(x).rank(method="first")
    bins = pd.qcut(r, q=n_bins, labels=False)
    return np.asarray(bins, dtype=int)

def compute_subject_cumhaz_up_to_k(
    long_df: pd.DataFrame,
    model: dict,
    k_col: str,
    id_col: str,
    k0: int,
) -> pd.DataFrame:
    """
    Compute subject-level cumulative hazard up to and including interval k0:
      H_i(t0) = sum_{k<=k0} y_ik * hazard_ik
    where hazard_ik = exp(eta_ik) from predict_long.

    Returns df with [id_col, cumhaz].
    """
    cfg = model["config"]
    y_col = cfg["exposure_col"]

    pred, _ = predict_long(long_df, model, return_X=False)
    hazard = pred["hazard"].to_numpy(dtype=float)

    tmp = long_df[[id_col, k_col, y_col]].copy()
    tmp["_hazard"] = hazard
    tmp["_increment"] = tmp[y_col].astype(float) * tmp["_hazard"]

    tmp[k_col] = tmp[k_col].astype(int)
    tmp = tmp.loc[tmp[k_col] <= int(k0), [id_col, "_increment"]]

    out = (
        tmp.groupby(id_col, as_index=False)["_increment"]
        .sum()
        .rename(columns={"_increment": "cumhaz"})
    )
    return out


def build_calibration_table(
    subj_df: pd.DataFrame,
    t0: float,
    n_bins: int,
    time_col: str,
    event_col: str,
    strata_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    subj_df must contain: _pred_risk, time_col, event_col (and optional strata_col).
    """
    df = subj_df.copy()
    df["bin"] = quantile_bins(df["_pred_risk"].to_numpy(dtype=float), n_bins=n_bins)

    group_cols = ["bin"] if strata_col is None else [strata_col, "bin"]
    rows = []

    for keys, sub in df.groupby(group_cols, dropna=False):
        times = sub[time_col].to_numpy(dtype=float)
        events = sub[event_col].to_numpy(dtype=int)

        S = km_survival_at(t0, times, events)
        obs_risk = 1.0 - S
        pred_risk = float(sub["_pred_risk"].mean())

        row = {
            "horizon": float(t0),
            "bin": int(sub["bin"].iloc[0]),
            "n": int(len(sub)),
            "pred_risk_mean": pred_risk,
            "obs_risk_km": float(obs_risk),
        }
        if strata_col is not None:
            row[strata_col] = keys[0]
        rows.append(row)

    res = pd.DataFrame(rows)
    sort_cols = ([] if strata_col is None else [strata_col]) + ["bin"]
    return res.sort_values(sort_cols).reset_index(drop=True)


def plot_calibration(res: pd.DataFrame, out_png: Path, t0: float, strata_col: Optional[str]) -> None:
    plt.figure()

    if strata_col is not None:
        for sval, sub in res.groupby(strata_col, dropna=False):
            plt.plot(
                sub["pred_risk_mean"].to_numpy(dtype=float),
                sub["obs_risk_km"].to_numpy(dtype=float),
                marker="o",
                linestyle="-",
                linewidth=1.5,
                label=str(sval),
            )
        plt.legend(ncols=2, fontsize=8)
    else:
        plt.plot(
            res["pred_risk_mean"].to_numpy(dtype=float),
            res["obs_risk_km"].to_numpy(dtype=float),
            marker="o",
            linestyle="-",
            linewidth=1.5,
        )

    lo = float(min(res["pred_risk_mean"].min(), res["obs_risk_km"].min()))
    hi = float(max(res["pred_risk_mean"].max(), res["obs_risk_km"].max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)

    plt.title(f"Risk calibration at t={int(t0)} days (KM vs predicted)")
    plt.xlabel("Predicted risk (mean within bin)")
    plt.ylabel("Observed risk (KM within bin)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Risk calibration (KM vs predicted risk) from PE model.")
    ap.add_argument("--model", required=True, help="Saved model JSON")
    ap.add_argument("--long_csv", required=True, help="Long-format CSV (expanded)")
    ap.add_argument("--input_csv", required=True, help="Non-long CSV with time/event (+ optional strata)")
    ap.add_argument("--out_dir", required=True, help="Output directory")

    ap.add_argument("--id_col", default="__row_id__", help="ID col in long; created in input if missing")
    ap.add_argument("--k_col", default="k", help="Interval index column in long CSV")
    ap.add_argument("--time_col", default="time", help="Time column in input CSV")
    ap.add_argument("--event_col", default="event", help="Event indicator column in input CSV (0/1)")

    ap.add_argument("--breaks", required=True, help="Comma-separated breakpoints; horizons must be included")
    ap.add_argument("--horizons", default="365,1825", help="Comma-separated horizons (must be in breaks)")
    ap.add_argument("--n_bins", type=int, default=10, help="Number of risk bins (10 = deciles)")
    ap.add_argument("--strata_col", default="", help="Optional strata column in input CSV (e.g., stage)")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== run_calibration_risk ===")
    print(f"model     : {args.model}")
    print(f"long_csv  : {args.long_csv}")
    print(f"input_csv : {args.input_csv}")
    print(f"out_dir   : {out_dir}")

    with open(args.model, "r", encoding="utf-8") as f:
        model = json.load(f)

    long_df = pd.read_csv(args.long_csv)
    base_df = pd.read_csv(args.input_csv)

    print(f"Loaded long_df: rows={len(long_df):,}, cols={len(long_df.columns)}")
    print(f"Loaded base_df: rows={len(base_df):,}, cols={len(base_df.columns)}")

    # Validate columns
    if args.id_col not in long_df.columns:
        raise ValueError(f"long_csv missing id_col '{args.id_col}'")
    if args.k_col not in long_df.columns:
        raise ValueError(f"long_csv missing k_col '{args.k_col}'")

    for c in [args.time_col, args.event_col]:
        if c not in base_df.columns:
            raise ValueError(f"input_csv missing required column '{c}'")

    strata_col = args.strata_col.strip() or None
    if strata_col is not None and strata_col not in base_df.columns:
        raise ValueError(f"strata_col '{strata_col}' not found in input_csv")

    # Ensure ID exists in input
    base_df = ensure_id_in_input(base_df, args.id_col)

    breaks = np.asarray(parse_floats_csv(args.breaks), dtype=float)
    horizons = np.asarray(parse_floats_csv(args.horizons), dtype=float)

    print(f"Parsed breaks: {len(breaks)} points")
    print(f"Horizons: {list(horizons)}")

    # Main loop over horizons
    for t0 in horizons:
        k0 = horizon_to_k(breaks, float(t0))
        print(f"\n--- Horizon t={t0} -> k0={k0} ---")

        ch = compute_subject_cumhaz_up_to_k(
            long_df=long_df,
            model=model,
            k_col=args.k_col,
            id_col=args.id_col,
            k0=int(k0),
        )
        print(f"Computed subject cumhaz up to k0: rows={len(ch):,}")

        cols = [args.id_col, args.time_col, args.event_col] + ([] if strata_col is None else [strata_col])
        subj = base_df[cols].merge(ch, on=args.id_col, how="inner")
        print(f"After merge: rows={len(subj):,}")
        if subj.empty:
            raise ValueError(
                "Empty merge between input_csv and cumhaz. "
                "Most likely the input row order does not match __row_id__ assignment in expand_long."
            )

        subj["_pred_surv"] = np.exp(-subj["cumhaz"].to_numpy(dtype=float))
        subj["_pred_risk"] = 1.0 - subj["_pred_surv"]

        print(
            f"Pred risk summary: min={subj['_pred_risk'].min():.6g}, "
            f"p50={subj['_pred_risk'].median():.6g}, max={subj['_pred_risk'].max():.6g}"
        )

        res = build_calibration_table(
            subj_df=subj,
            t0=float(t0),
            n_bins=int(args.n_bins),
            time_col=args.time_col,
            event_col=args.event_col,
            strata_col=strata_col,
        )

        out_csv = out_dir / f"calibration_risk_bins_t{int(t0)}.csv"
        res.to_csv(out_csv, index=False)
        print(f"Wrote: {out_csv} (rows={len(res)})")

        out_png = out_dir / f"calibration_risk_bins_t{int(t0)}.png"
        plot_calibration(res, out_png, t0=float(t0), strata_col=strata_col)
        print(f"Wrote: {out_png}")

    print("\nCalibration complete.")


if __name__ == "__main__":
    main()