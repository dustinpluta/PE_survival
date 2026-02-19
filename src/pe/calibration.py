# src/pe/calibration.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pe.predict import predict_long


@dataclass(frozen=True)
class PECalibrationConfig:
    interval_col: str = "k"
    event_col: str = "d"
    exposure_col: str = "y"

    # Plotting / stability
    eps: float = 1e-12


def _validate_long_for_calibration(long_df: pd.DataFrame, cfg: PECalibrationConfig) -> None:
    need = [cfg.interval_col, cfg.event_col, cfg.exposure_col]
    missing = [c for c in need if c not in long_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in long_df: {missing}")

    vals = set(pd.unique(long_df[cfg.event_col].dropna()))
    if not vals.issubset({0, 1}):
        raise ValueError(f"Event column must be 0/1; found {sorted(vals)}")


def calibration_by_interval(
    long_df: pd.DataFrame,
    model: Dict[str, Any],
    cfg: Optional[PECalibrationConfig] = None,
) -> pd.DataFrame:
    """
    Returns per-interval calibration table with:
      k, observed, expected, exposure_sum, obs_minus_exp, ratio_obs_exp
    """
    cfg = cfg or PECalibrationConfig()
    _validate_long_for_calibration(long_df, cfg)

    pred, _ = predict_long(long_df, model, return_X=False)
    mu = pred["mu"].to_numpy(dtype=float)

    df = long_df[[cfg.interval_col, cfg.event_col, cfg.exposure_col]].copy()
    df["_mu"] = mu

    out = (
        df.groupby(cfg.interval_col, as_index=False)
        .agg(
            observed=(cfg.event_col, "sum"),
            expected=("_mu", "sum"),
            exposure_sum=(cfg.exposure_col, "sum"),
        )
        .sort_values(cfg.interval_col)
        .reset_index(drop=True)
    )

    out["obs_minus_exp"] = out["observed"] - out["expected"]
    out["ratio_obs_exp"] = out["observed"] / np.maximum(out["expected"], cfg.eps)

    return out


def calibration_by_interval_and_group(
    long_df: pd.DataFrame,
    model: Dict[str, Any],
    group_col: str,
    cfg: Optional[PECalibrationConfig] = None,
) -> pd.DataFrame:
    """
    Returns calibration by (k, group) with:
      k, group, observed, expected, exposure_sum, obs_minus_exp, ratio_obs_exp
    """
    cfg = cfg or PECalibrationConfig()
    _validate_long_for_calibration(long_df, cfg)

    if group_col not in long_df.columns:
        raise ValueError(f"group_col '{group_col}' not found in long_df")

    pred, _ = predict_long(long_df, model, return_X=False)
    mu = pred["mu"].to_numpy(dtype=float)

    df = long_df[[cfg.interval_col, group_col, cfg.event_col, cfg.exposure_col]].copy()
    df["_mu"] = mu

    out = (
        df.groupby([cfg.interval_col, group_col], as_index=False)
        .agg(
            observed=(cfg.event_col, "sum"),
            expected=("_mu", "sum"),
            exposure_sum=(cfg.exposure_col, "sum"),
        )
        .sort_values([cfg.interval_col, group_col])
        .reset_index(drop=True)
    )

    out["obs_minus_exp"] = out["observed"] - out["expected"]
    out["ratio_obs_exp"] = out["observed"] / np.maximum(out["expected"], cfg.eps)

    return out


def plot_interval_calibration(
    cal: pd.DataFrame,
    out_path: str | Path,
    title: str = "Calibration by interval",
) -> None:
    """
    Saves a simple observed vs expected plot across interval k.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    k = cal["k"].to_numpy()
    obs = cal["observed"].to_numpy(dtype=float)
    exp = cal["expected"].to_numpy(dtype=float)

    plt.figure()
    plt.plot(k, obs, marker="o")
    plt.plot(k, exp, marker="o")
    plt.title(title)
    plt.xlabel("Interval k")
    plt.ylabel("Events (observed / expected)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_interval_ratio(
    cal: pd.DataFrame,
    out_path: str | Path,
    title: str = "Observed / Expected by interval",
) -> None:
    """
    Saves a ratio plot across k (ideal ~1).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    k = cal["k"].to_numpy()
    ratio = cal["ratio_obs_exp"].to_numpy(dtype=float)

    plt.figure()
    plt.plot(k, ratio, marker="o")
    plt.title(title)
    plt.xlabel("Interval k")
    plt.ylabel("Observed / Expected")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_group_interval_ratio(
    cal_kg: pd.DataFrame,
    group_col: str,
    out_path: str | Path,
    title: str = "Observed / Expected by interval and group",
    max_groups: int = 12,
) -> None:
    """
    One line per group: ratio_obs_exp vs k. Truncates to max_groups if many.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    groups = list(pd.unique(cal_kg[group_col]))
    if len(groups) > max_groups:
        groups = groups[:max_groups]

    plt.figure()
    for g in groups:
        sub = cal_kg[cal_kg[group_col] == g]
        plt.plot(sub["k"].to_numpy(), sub["ratio_obs_exp"].to_numpy(dtype=float), marker="o")

    plt.title(title)
    plt.xlabel("Interval k")
    plt.ylabel("Observed / Expected")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
