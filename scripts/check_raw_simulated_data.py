from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_floats_csv(s: str) -> np.ndarray:
    return np.asarray([float(x) for x in str(s).split(",") if str(x).strip()], dtype=float)


def km_curve(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns step-function KM survival curve evaluated at event times:
      t_unique, S_hat(t_unique)
    """
    times = np.asarray(times, float)
    events = np.asarray(events, int)

    # Unique event times (exclude censored)
    t_event = np.unique(times[events == 1])
    if t_event.size == 0:
        return np.array([0.0]), np.array([1.0])

    S = 1.0
    S_vals = []
    for tj in t_event:
        n_risk = np.sum(times >= tj)
        d = np.sum((times == tj) & (events == 1))
        if n_risk <= 0:
            continue
        S *= (1.0 - d / n_risk)
        S_vals.append(S)

    return t_event, np.asarray(S_vals, float)


def nelson_aalen(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Nelson–Aalen cumulative hazard evaluated at event times:
      t_unique, H_hat(t_unique)
    """
    times = np.asarray(times, float)
    events = np.asarray(events, int)

    t_event = np.unique(times[events == 1])
    if t_event.size == 0:
        return np.array([0.0]), np.array([0.0])

    H = 0.0
    H_vals = []
    for tj in t_event:
        n_risk = np.sum(times >= tj)
        d = np.sum((times == tj) & (events == 1))
        if n_risk <= 0:
            continue
        H += d / n_risk
        H_vals.append(H)

    return t_event, np.asarray(H_vals, float)


def crude_piecewise_hazard(
    df: pd.DataFrame,
    breaks: np.ndarray,
    *,
    time_col: str,
    event_col: str,
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Crude PE hazard estimate per interval:
      lambda_hat_k = (events in interval) / (person-time in interval)

    Person-time uses clipping at observed time (standard crude estimate).
    """
    br = np.asarray(breaks, float)
    K = len(br) - 1

    out_rows = []
    groups = ["ALL"] if group_col is None else sorted(df[group_col].astype(str).unique())

    for g in groups:
        dfg = df if group_col is None else df[df[group_col].astype(str) == g]

        t = dfg[time_col].to_numpy(float)
        e = dfg[event_col].to_numpy(int)

        for k in range(K):
            a, b = br[k], br[k + 1]
            # exposure in [a,b): sum_i max(0, min(t_i, b) - a) but only if t_i > a
            y = np.maximum(0.0, np.minimum(t, b) - a)

            # events in interval: event occurs if e_i=1 and t_i in [a,b)
            d = np.sum((e == 1) & (t >= a) & (t < b))

            pt = float(y.sum())
            lam_hat = float(d / pt) if pt > 0 else np.nan

            out_rows.append(
                {
                    "group": g,
                    "k": k,
                    "t_left": float(a),
                    "t_right": float(b),
                    "width": float(b - a),
                    "events": int(d),
                    "pt": pt,
                    "hazard_hat": lam_hat,
                }
            )

    return pd.DataFrame(out_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Raw simulated survival data QA diagnostics.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", default="models/qa_raw")
    ap.add_argument("--time_col", default="time")
    ap.add_argument("--event_col", default="event")
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--group_col", default="stage", help="Optional grouping col for stratified plots")
    ap.add_argument("--breaks", default="", help="Comma-separated breaks (days) for crude piecewise hazards")
    ap.add_argument("--horizons", default="365,725,1825", help="Horizons (days) for KM risk sanity checks")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    print(f"Loaded: {args.csv}")
    print(f"Rows: {len(df):,}  Cols: {len(df.columns)}")

    # ---- Basic column checks ----
    for c in [args.id_col, args.time_col, args.event_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'")

    # ID uniqueness (base should be 1 row per subject)
    n_unique = df[args.id_col].nunique(dropna=False)
    if n_unique != len(df):
        print(f"[WARN] id not unique: unique={n_unique:,} rows={len(df):,}")
    else:
        print(f"[OK] id unique: {n_unique:,}")

    # Event coding sanity
    ev_vals = sorted(pd.unique(df[args.event_col]))
    print("event unique values:", ev_vals)
    if not set(ev_vals).issubset({0, 1}):
        print("[WARN] event is not strictly 0/1")

    # Time sanity
    t = pd.to_numeric(df[args.time_col], errors="coerce")
    if t.isna().any():
        raise ValueError("time contains NA/non-numeric values")
    if (t < 0).any():
        raise ValueError("time contains negative values")

    print("time summary:")
    print(t.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))

    # Censoring fraction
    e = pd.to_numeric(df[args.event_col], errors="coerce").astype(int)
    event_rate = float(e.mean())
    censor_frac = float(1.0 - event_rate)
    print(f"Event rate: {event_rate:.4f}")
    print(f"Censor frac: {censor_frac:.4f}")

    # ---- KM + NA curves overall ----
    tt_km, S_km = km_curve(t.to_numpy(float), e.to_numpy(int))
    tt_na, H_na = nelson_aalen(t.to_numpy(float), e.to_numpy(int))

    plt.figure()
    plt.step(np.r_[0.0, tt_km], np.r_[1.0, S_km], where="post")
    plt.xlabel("time")
    plt.ylabel("S(t)  (Kaplan–Meier)")
    plt.title("KM survival (overall)")
    plt.tight_layout()
    p = out_dir / "km_overall.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print("Wrote:", p)

    plt.figure()
    plt.step(np.r_[0.0, tt_na], np.r_[0.0, H_na], where="post")
    plt.xlabel("time")
    plt.ylabel("H(t)  (Nelson–Aalen)")
    plt.title("Cumulative hazard (overall)")
    plt.tight_layout()
    p = out_dir / "na_cumhaz_overall.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print("Wrote:", p)

    # ---- KM risk at horizons ----
    horizons = parse_floats_csv(args.horizons)
    for h in horizons:
        # KM survival at h: last S(tj) for tj <= h
        if tt_km.size == 0:
            S_h = 1.0
        else:
            mask = tt_km <= h
            S_h = float(S_km[mask][-1]) if np.any(mask) else 1.0
        print(f"KM risk at t={int(h)}: {1.0 - S_h:.6f}")

    # ---- Grouped KM curves (optional) ----
    group_col = args.group_col.strip()
    if group_col and group_col in df.columns:
        plt.figure()
        for g, sub in df.groupby(group_col, dropna=False):
            tg = pd.to_numeric(sub[args.time_col], errors="coerce").to_numpy(float)
            eg = pd.to_numeric(sub[args.event_col], errors="coerce").astype(int).to_numpy()
            tt, S = km_curve(tg, eg)
            plt.step(np.r_[0.0, tt], np.r_[1.0, S], where="post", label=str(g))
        plt.xlabel("time")
        plt.ylabel("S(t)")
        plt.title(f"KM survival by {group_col}")
        plt.legend(ncols=2, fontsize=8)
        plt.tight_layout()
        p = out_dir / f"km_by_{group_col}.png"
        plt.savefig(p, dpi=150)
        plt.close()
        print("Wrote:", p)

    # ---- Event vs censor time histograms ----
    plt.figure()
    plt.hist(t[e == 1], bins=50, alpha=0.7, label="events")
    plt.hist(t[e == 0], bins=50, alpha=0.7, label="censored")
    plt.xlabel("time")
    plt.ylabel("count")
    plt.title("Observed time distribution: events vs censored")
    plt.legend()
    plt.tight_layout()
    p = out_dir / "time_hist_event_vs_censor.png"
    plt.savefig(p, dpi=150)
    plt.close()
    print("Wrote:", p)

    # ---- Crude piecewise hazard (if breaks provided) ----
    if args.breaks.strip():
        breaks = parse_floats_csv(args.breaks)
        if breaks.size < 2 or not np.all(np.diff(breaks) > 0):
            raise ValueError("breaks must be strictly increasing")
        crude = crude_piecewise_hazard(
            df,
            breaks,
            time_col=args.time_col,
            event_col=args.event_col,
            group_col=(group_col if group_col in df.columns else None),
        )
        out_csv = out_dir / "crude_piecewise_hazard.csv"
        crude.to_csv(out_csv, index=False)
        print("Wrote:", out_csv)

        # plot overall crude hazard
        overall = crude[crude["group"] == "ALL"].copy()
        mids = 0.5 * (overall["t_left"].to_numpy(float) + overall["t_right"].to_numpy(float))
        plt.figure()
        plt.plot(mids, overall["hazard_hat"].to_numpy(float), marker="o")
        plt.xlabel("interval midpoint")
        plt.ylabel("crude hazard (events / person-time)")
        plt.title("Crude piecewise hazard (overall)")
        plt.tight_layout()
        p = out_dir / "crude_piecewise_hazard_overall.png"
        plt.savefig(p, dpi=150)
        plt.close()
        print("Wrote:", p)

    # ---- Covariate sanity ----
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in {args.id_col, args.time_col, args.event_col}]
    if num_cols:
        summ = df[num_cols].describe().T
        out_csv = out_dir / "numeric_covariate_summary.csv"
        summ.to_csv(out_csv)
        print("Wrote:", out_csv)

    cat_cols = [c for c in df.columns if c not in {args.id_col, args.time_col, args.event_col} and df[c].dtype == object]
    if cat_cols:
        for c in cat_cols:
            vc = df[c].value_counts(dropna=False)
            out_csv = out_dir / f"categorical_{c}_counts.csv"
            vc.to_csv(out_csv)
            print("Wrote:", out_csv)


if __name__ == "__main__":
    main()