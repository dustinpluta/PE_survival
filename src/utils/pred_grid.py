"""
pred_grid.py

Utilities to build a *prediction exposure grid* for PE risk prediction / calibration.

Key idea:
- Fitting uses the observed long dataframe (exposure truncated at censor/event time).
- Risk prediction at horizon t0 must integrate hazard up to t0 for *all* subjects, regardless of censoring.
  That requires a subject × interval grid with y equal to interval width (truncated in last interval).

This module builds that grid.

Typical usage:
    base_df = pd.read_csv("data/base.csv")   # one row per subject
    long_df = pd.read_csv("data/long.csv")   # used only to derive breaks (optional)
    breaks = derive_breaks_from_long(long_df)
    pred_long = build_pred_long(base_df, breaks, t0=365.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PredGridConfig:
    id_col: str = "id"
    time_col: str = "time"     # optional; not required for prediction grid
    event_col: str = "event"   # optional; not required for prediction grid
    k_col: str = "k"
    t_left_col: str = "t_left"
    t_right_col: str = "t_right"
    y_col: str = "y"
    d_col: str = "d"


def derive_breaks_from_long(
    long_df: pd.DataFrame,
    *,
    cfg: PredGridConfig = PredGridConfig(),
    ensure_strict_increasing: bool = True,
) -> np.ndarray:
    """
    Derive calendar breaks from a long dataframe that has t_left and t_right columns.

    Returns:
        breaks: sorted unique endpoints, float array of length K+1

    Notes:
    - This assumes long_df was produced by expand_long with fixed calendar intervals.
    """
    for c in [cfg.t_left_col, cfg.t_right_col]:
        if c not in long_df.columns:
            raise ValueError(f"long_df missing required column '{c}'")

    t_left = pd.to_numeric(long_df[cfg.t_left_col], errors="raise").to_numpy(dtype=float)
    t_right = pd.to_numeric(long_df[cfg.t_right_col], errors="raise").to_numpy(dtype=float)

    breaks = np.unique(np.concatenate([t_left, t_right]))
    breaks = np.sort(breaks.astype(float))

    if ensure_strict_increasing:
        if len(breaks) < 2:
            raise ValueError("Derived breaks has length < 2.")
        if not np.all(np.diff(breaks) > 0):
            raise ValueError("Derived breaks are not strictly increasing.")

    return breaks


def _interval_table_from_breaks(breaks: np.ndarray) -> pd.DataFrame:
    b = np.asarray(breaks, dtype=float)
    if b.ndim != 1 or len(b) < 2:
        raise ValueError("breaks must be a 1D array of length >= 2")
    if not np.all(np.diff(b) > 0):
        raise ValueError("breaks must be strictly increasing")

    K = len(b) - 1
    out = pd.DataFrame(
        {
            "k": np.arange(K, dtype=int),
            "t_left": b[:-1],
            "t_right": b[1:],
        }
    )
    out["width"] = out["t_right"] - out["t_left"]
    return out


def build_pred_long(
    base_df: pd.DataFrame,
    breaks: Sequence[float],
    *,
    t0: Optional[float] = None,
    covariate_cols: Optional[Sequence[str]] = None,
    cfg: PredGridConfig = PredGridConfig(),
) -> pd.DataFrame:
    """
    Build prediction long dataframe for integrating hazard up to horizon t0.

    Args:
        base_df:
            One row per subject. Must include cfg.id_col and any covariates used in the model.
            time/event columns are ignored if present.
        breaks:
            Calendar breakpoints (days), length K+1.
        t0:
            Horizon (days). If None, defaults to breaks[-1].
        covariate_cols:
            If provided, restrict covariates copied from base_df to this list.
            Otherwise, uses all columns except (time_col, event_col) and id_col is always included.
        cfg:
            Column naming config.

    Returns:
        pred_long:
            DataFrame with columns:
              id, k, t_left, t_right, y, d(=0), + covariates
            where y is the interval width truncated at t0 in the final intersecting interval.

    Important:
        This grid is NOT the observed exposure. It is the exposure needed for risk prediction.
    """
    if cfg.id_col not in base_df.columns:
        raise ValueError(f"base_df missing required id column '{cfg.id_col}'")

    b = np.asarray(breaks, dtype=float)
    if t0 is None:
        t0 = float(b[-1])
    else:
        t0 = float(t0)
    if t0 <= 0:
        raise ValueError("t0 must be > 0")
    if t0 > float(b[-1]) + 1e-12:
        raise ValueError(f"t0={t0} exceeds followup_end={b[-1]} from breaks")

    intervals = _interval_table_from_breaks(b)
    # keep intervals that intersect [0, t0]
    keep = intervals[intervals["t_left"] < t0].copy()
    keep["t_right_h"] = np.minimum(keep["t_right"].to_numpy(float), t0)
    keep["y"] = keep["t_right_h"] - keep["t_left"]
    keep = keep[keep["y"] > 0].copy()

    # Select covariates to carry forward (constant by subject)
    if covariate_cols is None:
        drop = {cfg.time_col, cfg.event_col}
        cov_cols = [c for c in base_df.columns if c not in drop and c != cfg.id_col]
    else:
        cov_cols = list(covariate_cols)
        for c in cov_cols:
            if c not in base_df.columns:
                raise ValueError(f"covariate_cols includes '{c}' not in base_df")

    subj = base_df[[cfg.id_col] + cov_cols].copy()

    # Cartesian product: subjects × intervals
    # Requires pandas >= 1.2 for how="cross"
    pred_long = subj.merge(
        keep[[ "k", "t_left", "t_right_h", "y" ]],
        how="cross",
    ).rename(columns={"t_right_h": cfg.t_right_col})

    pred_long = pred_long.rename(
        columns={
            "k": cfg.k_col,
            "t_left": cfg.t_left_col,
            "y": cfg.y_col,
        }
    )

    # ensure schema columns exist
    pred_long[cfg.d_col] = 0

    # final ordering: id, k, t_left, t_right, y, d, covariates...
    front = [cfg.id_col, cfg.k_col, cfg.t_left_col, cfg.t_right_col, cfg.y_col, cfg.d_col]
    rest = [c for c in pred_long.columns if c not in front]
    pred_long = pred_long[front + rest]

    return pred_long


def build_pred_long_from_long(
    base_df: pd.DataFrame,
    long_df: pd.DataFrame,
    *,
    t0: Optional[float] = None,
    covariate_cols: Optional[Sequence[str]] = None,
    cfg: PredGridConfig = PredGridConfig(),
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Convenience wrapper:
      - derive breaks from long_df
      - build prediction grid from base_df and breaks

    Returns:
      pred_long, breaks
    """
    breaks = derive_breaks_from_long(long_df, cfg=cfg)
    pred_long = build_pred_long(base_df, breaks, t0=t0, covariate_cols=covariate_cols, cfg=cfg)
    return pred_long, breaks