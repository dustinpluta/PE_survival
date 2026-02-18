# src/pe/expand_long.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExpandLongConfig:
    time_col: str = "time"
    event_col: str = "event"
    id_col: str = "__row_id__"
    keep_cols: Optional[Sequence[str]] = None
    clip_to_followup: bool = True
    eps: float = 1e-12


# -------------------------
# Step 0: validation helpers
# -------------------------
def validate_breaks(breaks: np.ndarray) -> np.ndarray:
    br = np.asarray(breaks, float)
    if br.ndim != 1 or br.size < 2:
        raise ValueError("breaks must be a 1D array of length >= 2")
    if not np.all(np.diff(br) > 0):
        raise ValueError("breaks must be strictly increasing")
    if br[0] != 0:
        raise ValueError("breaks must start at 0 for this project")
    return br


def validate_survival_df(df: pd.DataFrame, time_col: str, event_col: str) -> None:
    if time_col not in df.columns:
        raise ValueError(f"Missing time column '{time_col}'")
    if event_col not in df.columns:
        raise ValueError(f"Missing event column '{event_col}'")

    t = df[time_col].to_numpy(float)
    if np.any(~np.isfinite(t)):
        raise ValueError("Non-finite values in time column")
    if np.any(t < 0):
        raise ValueError("Negative times found")

    vals = set(pd.unique(df[event_col].dropna()))
    if not vals.issubset({0, 1}):
        raise ValueError(f"Event column must be coded 0/1; found values={sorted(vals)}")


def choose_keep_cols(df: pd.DataFrame, cfg: ExpandLongConfig) -> list[str]:
    if cfg.keep_cols is None:
        keep = [c for c in df.columns if c not in (cfg.time_col, cfg.event_col)]
    else:
        missing = [c for c in cfg.keep_cols if c not in df.columns]
        if missing:
            raise ValueError(f"keep_cols contains missing columns: {missing}")
        keep = list(cfg.keep_cols)

    # Never carry id_col as a covariate
    keep = [c for c in keep if c != cfg.id_col]
    return keep


# -----------------------------------------
# Step 1: build a clean "base" subject table
# -----------------------------------------
def make_base(df: pd.DataFrame, breaks: np.ndarray, cfg: Optional[ExpandLongConfig] = None) -> pd.DataFrame:
    """
    Returns a clean subject-level table with:
      - cfg.id_col = 0..n-1 (always)
      - time_col optionally clipped to follow-up
      - event_col as int 0/1
      - kept covariates

    No long expansion yet.
    """
    cfg = cfg or ExpandLongConfig()
    br = validate_breaks(breaks)
    validate_survival_df(df, cfg.time_col, cfg.event_col)

    base = df.reset_index(drop=True).copy()
    base[cfg.id_col] = np.arange(len(base), dtype=int)

    if cfg.clip_to_followup:
        base[cfg.time_col] = np.minimum(base[cfg.time_col].to_numpy(float), float(br[-1]))

    base[cfg.event_col] = base[cfg.event_col].astype(int)

    keep = choose_keep_cols(base, cfg)
    cols = [cfg.id_col, cfg.time_col, cfg.event_col] + keep
    base = base.loc[:, cols]

    # Guard: no duplicate columns
    if base.columns.duplicated().any():
        dups = base.columns[base.columns.duplicated()].tolist()
        raise RuntimeError(f"make_base produced duplicate columns: {dups}")

    return base


# -----------------------------------------
# Step 2: compute exposure y_ik (no events d)
# -----------------------------------------
def expand_exposure(base: pd.DataFrame, breaks: np.ndarray, cfg: Optional[ExpandLongConfig] = None) -> pd.DataFrame:
    """
    Returns long df with exposure y but d is all zeros (not assigned yet).
    """
    cfg = cfg or ExpandLongConfig()
    br = validate_breaks(breaks)

    ids = base[cfg.id_col].to_numpy(int)
    T = base[cfg.time_col].to_numpy(float)

    n = len(base)
    K = len(br) - 1
    left = br[:-1]
    right = br[1:]
    width = right - left

    out_id = np.repeat(ids, K)
    out_k = np.tile(np.arange(K, dtype=int), n)
    out_left = np.tile(left, n)
    out_right = np.tile(right, n)
    out_width = np.tile(width, n)

    T_rep = np.repeat(T, K)
    y = np.maximum(0.0, np.minimum(T_rep, out_right) - out_left)

    long_df = pd.DataFrame(
        {
            cfg.id_col: out_id,
            "k": out_k,
            "t_left": out_left,
            "t_right": out_right,
            "width": out_width,
            "y": y,
            "d": np.zeros(n * K, dtype=int),
        }
    )

    # Attach covariates by repeating base rows
    cov_cols = [c for c in base.columns if c not in (cfg.time_col, cfg.event_col, cfg.id_col)]
    cov_rep = base.loc[base.index.repeat(K), cov_cols].reset_index(drop=True)
    long_df = pd.concat([long_df.reset_index(drop=True), cov_rep], axis=1)

    if long_df.columns.duplicated().any():
        dups = long_df.columns[long_df.columns.duplicated()].tolist()
        raise RuntimeError(f"expand_exposure produced duplicate columns: {dups}")

    return long_df


# -------------------------------------------------------
# Step 3: assign events d_ik using "positive exposure" rule
# -------------------------------------------------------
def assign_events(long_df: pd.DataFrame, base: pd.DataFrame, breaks: np.ndarray, cfg: Optional[ExpandLongConfig] = None) -> pd.DataFrame:
    """
    Assign exactly one d=1 per event subject, choosing an interval with y>0 when possible.

    Rule:
      - Start with k_right = searchsorted(br[1:], T, side='right') clipped to [0,K-1]
      - If the chosen interval has y==0, shift to the previous interval (k-1) if possible.
      - If still y==0 (T==0 case), keep k=0 and retain that row later.
    """
    cfg = cfg or ExpandLongConfig()
    br = validate_breaks(breaks)

    out = long_df.copy()
    out["d"] = 0

    K = len(br) - 1
    T = base[cfg.time_col].to_numpy(float)
    E = base[cfg.event_col].to_numpy(int)

    # candidate interval
    k = np.searchsorted(br[1:], T, side="right")
    k = np.clip(k, 0, K - 1).astype(int)

    # check exposure in that interval: y_ik = max(0, min(T, br[k+1]) - br[k])
    a = br[k]
    b = br[k + 1]
    yk = np.maximum(0.0, np.minimum(T, b) - a)

    # if yk==0, shift left
    needs = yk <= cfg.eps
    if np.any(needs):
        k2 = k.copy()
        k2[needs] = np.maximum(k2[needs] - 1, 0)

        a2 = br[k2]
        b2 = br[k2 + 1]
        yk2 = np.maximum(0.0, np.minimum(T, b2) - a2)

        # if still 0 (T==0), force k=0
        k2[yk2 <= cfg.eps] = 0
        k = k2

    # set d=1 for event subjects at their chosen (id,k)
    ev_idx = np.where(E == 1)[0]
    if ev_idx.size > 0:
        ev_ids = base.loc[ev_idx, cfg.id_col].to_numpy(int)
        ev_k = k[ev_idx]

        # map (id,k) -> row mask
        # (fast vector approach)
        key = ev_ids * K + ev_k
        out_key = out[cfg.id_col].to_numpy(int) * K + out["k"].to_numpy(int)
        mark = np.isin(out_key, key)
        out.loc[mark, "d"] = 1

    return out


# ------------------------------------------------
# Final: one-shot API using the steps + row-dropping
# ------------------------------------------------
def expand_to_long(df: pd.DataFrame, breaks: np.ndarray, cfg: Optional[ExpandLongConfig] = None) -> pd.DataFrame:
    cfg = cfg or ExpandLongConfig()
    base = make_base(df, breaks, cfg)
    long_df = expand_exposure(base, breaks, cfg)
    long_df = assign_events(long_df, base, breaks, cfg)

    # Drop y==0 rows except keep event rows
    keep = (long_df["y"] > cfg.eps) | (long_df["d"] == 1)
    long_df = long_df.loc[keep].reset_index(drop=True)

    return long_df
