# scripts/run_calibration.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pe.predict import predict_long


def _parse_cutpoints(s: str) -> list[float]:
    # e.g. "365,730,1095" -> [365.0, 730.0, 1095.0]
    if not s.strip():
        return []
    return [float(x) for x in s.split(",") if x.strip()]


def _k_midpoints_from_breaks(breaks: list[float]) -> np.ndarray:
    b = np.asarray(breaks, dtype=float)
    if b.ndim != 1 or len(b) < 2:
        raise ValueError("model config 'breaks' must be a list with length >= 2")
    return 0.5 * (b[:-1] + b[1:])


def _assign_time_window(k: np.ndarray, k_mids: np.ndarray, cutpoints: list[float]) -> np.ndarray:
    """
    Map each row's interval k to a window label using midpoint time.
    Windows defined by cutpoints:
      (-inf, c1], (c1, c2], ..., (c_last, +inf)
    Labels: "W0", "W1", ...
    """
    t = k_mids[k]
    cps = np.asarray(cutpoints, dtype=float)
    # np.digitize returns bin index in 0..len(cps)
    w = np.digitize(t, cps, right=True)
    return np.asarray([f"W{j}" for j in w], dtype=object)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run PE calibration from saved model JSON + long CSV")
    ap.add_argument("--model", required=True, help="Path to saved model JSON")
    ap.add_argument("--long_csv", required=True, help="Long-format CSV")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--breaks",
        default="",
        help="Comma-separated breakpoints in days (needed for time-window calibration if not stored in model JSON).",
    )

    # Optional group calibration
    ap.add_argument("--group_col", default="", help="Optional group column for stratified calibration (e.g., stage)")

    # Time-window calibration controls
    ap.add_argument(
        "--time_cuts",
        default="365",
        help="Comma-separated day cutpoints for time windows (default: 365). Example: '180,365,730'.",
    )
    ap.add_argument(
        "--id_col",
        default="__row_id__",
        help="Subject id column in long data (default: __row_id__).",
    )

    args = ap.parse_args()

    model_path = Path(args.model)
    long_path = Path(args.long_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(model_path, "r", encoding="utf-8") as f:
        model = json.load(f)

    long_df = pd.read_csv(long_path)

    cfg = model["config"]
    interval_col = cfg["interval_col"]
    event_col = cfg["event_col"]
    exposure_col = cfg["exposure_col"]
    eps = float(cfg.get("eps", 1e-12))

    # Validate required columns
    for c in [args.id_col, interval_col, event_col, exposure_col]:
        if c not in long_df.columns:
            raise ValueError(f"long_df missing required column '{c}'. Columns: {list(long_df.columns)}")

    # Predict expected counts per long row
    pred, _ = predict_long(long_df, model, return_X=False)
    mu = pred["mu"].to_numpy(dtype=float)

    # -----------------------------
    # A) Calibration by interval k
    # -----------------------------
    df = long_df[[interval_col, event_col, exposure_col]].copy()
    df["_mu"] = mu

    cal_k = (
        df.groupby(interval_col, as_index=False)
        .agg(
            observed=(event_col, "sum"),
            expected=("_mu", "sum"),
            exposure=(exposure_col, "sum"),
        )
        .sort_values(interval_col)
        .reset_index(drop=True)
    )
    cal_k["ratio_obs_exp"] = cal_k["observed"] / np.maximum(cal_k["expected"], eps)
    cal_k["obs_minus_exp"] = cal_k["observed"] - cal_k["expected"]

    cal_k_path = out_dir / "calibration_by_interval.csv"
    cal_k.to_csv(cal_k_path, index=False)

    obs = cal_k["observed"].to_numpy(dtype=float)
    exp = cal_k["expected"].to_numpy(dtype=float)
    max_abs_diff = float(np.max(np.abs(obs - exp))) if len(obs) else float("nan")
    allclose = bool(np.allclose(obs, exp, rtol=1e-10, atol=1e-10))

    print(f"Wrote: {cal_k_path}")
    print(f"Observed sum={obs.sum():.6f}, Expected sum={exp.sum():.6f}")
    print(f"max|obs-exp|={max_abs_diff:.6f}, allclose={allclose}")

    # Plot observed vs expected by interval (will often overlap by construction)
    k_vals = cal_k[interval_col].to_numpy()
    plt.figure()
    plt.plot(k_vals, obs, marker="o", linestyle="-", linewidth=2, label="Observed", zorder=3)
    plt.plot(k_vals, exp, marker="s", linestyle="--", linewidth=2, label="Expected", zorder=2)
    plt.grid(True, axis="y", alpha=0.3)
    plt.title("Events by interval: observed vs expected")
    plt.xlabel(f"Interval ({interval_col})")
    plt.ylabel("Events")
    plt.legend()
    plt.tight_layout()
    p_obs_exp_k = out_dir / "events_obs_vs_exp_by_interval.png"
    plt.savefig(p_obs_exp_k)
    plt.close()
    print(f"Wrote: {p_obs_exp_k}")

    plt.figure()
    plt.plot(k_vals, cal_k["ratio_obs_exp"].to_numpy(dtype=float), marker="o", linestyle="-", linewidth=2)
    plt.grid(True, axis="y", alpha=0.3)
    plt.title("Observed / Expected by interval")
    plt.xlabel(f"Interval ({interval_col})")
    plt.ylabel("Observed / Expected")
    plt.tight_layout()
    p_ratio_k = out_dir / "ratio_obs_over_exp_by_interval.png"
    plt.savefig(p_ratio_k)
    plt.close()
    print(f"Wrote: {p_ratio_k}")

    # -----------------------------------------
    # B) Time-window calibration (informative)
    # -----------------------------------------
    breaks_list = None
    if "breaks" in cfg and cfg["breaks"]:
        breaks_list = cfg["breaks"]
    elif args.breaks.strip():
        breaks_list = _parse_cutpoints(args.breaks)  # returns list[float]
    else:
        raise ValueError(
            "Need breakpoints to do time-window calibration. "
            "Provide --breaks or store config['breaks'] in the model JSON."
        )

    k_mids = _k_midpoints_from_breaks(breaks_list)
    # ensure k is int within range
    k_int = long_df[interval_col].to_numpy()
    if not np.issubdtype(k_int.dtype, np.integer):
        k_int = k_int.astype(int)

    if np.min(k_int) < 0 or np.max(k_int) >= len(k_mids):
        raise ValueError(f"Interval values out of range for breaks: k in [{k_int.min()}, {k_int.max()}], "
                         f"but have {len(k_mids)} intervals")

    cutpoints = _parse_cutpoints(args.time_cuts)
    window = _assign_time_window(k_int, k_mids, cutpoints)

    dfw = long_df[[event_col, exposure_col]].copy()
    dfw["_mu"] = mu
    dfw["window"] = window

    cal_w = (
        dfw.groupby("window", as_index=False)
        .agg(
            observed=(event_col, "sum"),
            expected=("_mu", "sum"),
            exposure=(exposure_col, "sum"),
        )
        .sort_values("window")
        .reset_index(drop=True)
    )
    cal_w["ratio_obs_exp"] = cal_w["observed"] / np.maximum(cal_w["expected"], eps)
    cal_w["obs_minus_exp"] = cal_w["observed"] - cal_w["expected"]

    cal_w_path = out_dir / "calibration_by_timewindow.csv"
    cal_w.to_csv(cal_w_path, index=False)
    print(f"Wrote: {cal_w_path}")

    # Plot observed vs expected by window (this should show two curves unless perfect)
    w = cal_w["window"].to_numpy()
    obs_w = cal_w["observed"].to_numpy(dtype=float)
    exp_w = cal_w["expected"].to_numpy(dtype=float)

    plt.figure()
    plt.plot(w, obs_w, marker="o", linestyle="-", linewidth=2, label="Observed", zorder=3)
    plt.plot(w, exp_w, marker="s", linestyle="--", linewidth=2, label="Expected", zorder=2)
    plt.grid(True, axis="y", alpha=0.3)
    plt.title("Events by time window: observed vs expected")
    plt.xlabel("Time window")
    plt.ylabel("Events")
    plt.legend()
    plt.tight_layout()
    p_obs_exp_w = out_dir / "events_obs_vs_exp_by_timewindow.png"
    plt.savefig(p_obs_exp_w)
    plt.close()
    print(f"Wrote: {p_obs_exp_w}")

    plt.figure()
    plt.plot(w, cal_w["ratio_obs_exp"].to_numpy(dtype=float), marker="o", linestyle="-", linewidth=2)
    plt.grid(True, axis="y", alpha=0.3)
    plt.title("Observed / Expected by time window")
    plt.xlabel("Time window")
    plt.ylabel("Observed / Expected")
    plt.tight_layout()
    p_ratio_w = out_dir / "ratio_obs_over_exp_by_timewindow.png"
    plt.savefig(p_ratio_w)
    plt.close()
    print(f"Wrote: {p_ratio_w}")

    # -----------------------------------------
    # C) Stratified time-window calibration
    # -----------------------------------------
    if args.group_col.strip():
        group_col = args.group_col.strip()
        if group_col not in long_df.columns:
            raise ValueError(f"group_col '{group_col}' not found in long_df")

        dfwg = long_df[[group_col, event_col, exposure_col]].copy()
        dfwg["_mu"] = mu
        dfwg["window"] = window

        cal_wg = (
            dfwg.groupby(["window", group_col], as_index=False)
            .agg(
                observed=(event_col, "sum"),
                expected=("_mu", "sum"),
                exposure=(exposure_col, "sum"),
            )
            .sort_values(["window", group_col])
            .reset_index(drop=True)
        )
        cal_wg["ratio_obs_exp"] = cal_wg["observed"] / np.maximum(cal_wg["expected"], eps)

        cal_wg_path = out_dir / f"calibration_by_timewindow_and_{group_col}.csv"
        cal_wg.to_csv(cal_wg_path, index=False)
        print(f"Wrote: {cal_wg_path}")

        plt.figure()
        for gval, sub in cal_wg.groupby(group_col):
            plt.plot(
                sub["window"].to_numpy(),
                sub["ratio_obs_exp"].to_numpy(dtype=float),
                marker="o",
                linestyle="-",
                linewidth=1.5,
                label=str(gval),
            )
        plt.grid(True, axis="y", alpha=0.3)
        plt.title(f"Observed / Expected by time window and {group_col}")
        plt.xlabel("Time window")
        plt.ylabel("Observed / Expected")
        plt.legend(ncols=2, fontsize=8)
        plt.tight_layout()
        p_wg = out_dir / f"ratio_obs_over_exp_by_timewindow_and_{group_col}.png"
        plt.savefig(p_wg)
        plt.close()
        print(f"Wrote: {p_wg}")

        # Keep your existing per-(k,group) outputs too (optional but you already had them)
        df_g = long_df[[interval_col, group_col, event_col, exposure_col]].copy()
        df_g["_mu"] = mu

        cal_kg = (
            df_g.groupby([interval_col, group_col], as_index=False)
            .agg(
                observed=(event_col, "sum"),
                expected=("_mu", "sum"),
                exposure=(exposure_col, "sum"),
            )
            .sort_values([interval_col, group_col])
            .reset_index(drop=True)
        )
        cal_kg["ratio_obs_exp"] = cal_kg["observed"] / np.maximum(cal_kg["expected"], eps)

        cal_kg_path = out_dir / f"calibration_by_interval_and_{group_col}.csv"
        cal_kg.to_csv(cal_kg_path, index=False)
        print(f"Wrote: {cal_kg_path}")

        plt.figure()
        for gval, sub in cal_kg.groupby(group_col):
            plt.plot(
                sub[interval_col].to_numpy(),
                sub["ratio_obs_exp"].to_numpy(dtype=float),
                marker="o",
                linestyle="-",
                linewidth=1.0,
                label=str(gval),
            )
        plt.grid(True, axis="y", alpha=0.3)
        plt.title(f"Observed / Expected by interval and {group_col}")
        plt.xlabel(f"Interval ({interval_col})")
        plt.ylabel("Observed / Expected")
        plt.legend(ncols=2, fontsize=8)
        plt.tight_layout()
        p_kg = out_dir / f"ratio_obs_over_exp_by_interval_and_{group_col}.png"
        plt.savefig(p_kg)
        plt.close()
        print(f"Wrote: {p_kg}")

    print("Calibration complete.")


if __name__ == "__main__":
    main()