# src/pe/diagnostics.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pe.predict import predict_long, predict_subject_cumhaz


@dataclass(frozen=True)
class PEDiagnosticsConfig:
    id_col: str = "__row_id__"
    # numerical guardrails
    eps: float = 1e-12
    # plotting
    max_points: int = 20000  # downsample long rows for some plots if huge


def _subject_level_from_long(long_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Reconstruct subject-level:
      T_i = sum_k y_ik
      event_i = sum_k d_ik
    """
    if id_col not in long_df.columns:
        raise ValueError(f"long_df missing id_col '{id_col}'")

    # Required columns
    for c in ["y", "d"]:
        if c not in long_df.columns:
            raise ValueError(f"long_df missing required column '{c}'")

    sub = (
        long_df.groupby(id_col, as_index=False)
        .agg(T=("y", "sum"), event=("d", "sum"))
    )
    sub["event"] = sub["event"].astype(int)
    return sub


def _km_curve(times: np.ndarray, events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kaplan–Meier estimate of S(t) for right-censored data.
    Returns arrays (t_unique, S_hat_at_t).
    """
    times = np.asarray(times, float)
    events = np.asarray(events, int)

    # sort by time
    order = np.argsort(times)
    times = times[order]
    events = events[order]

    # unique event/censor times
    uniq = np.unique(times)

    n = len(times)
    at_risk = n
    S = 1.0

    t_out = []
    S_out = []

    idx0 = 0
    for t in uniq:
        # all subjects with this time
        mask = (times == t)
        d = int(np.sum(events[mask] == 1))
        c = int(np.sum(events[mask] == 0))

        # KM step only depends on events at t
        if at_risk > 0 and d > 0:
            S *= (1.0 - d / at_risk)

        t_out.append(float(t))
        S_out.append(float(S))

        # update risk set: remove both events and censors at t
        at_risk -= (d + c)
        idx0 += int(np.sum(mask))

    return np.asarray(t_out), np.asarray(S_out)


def compute_diagnostics(
    long_df: pd.DataFrame,
    model: Dict[str, Any],
    cfg: PEDiagnosticsConfig | None = None,
) -> Dict[str, Any]:
    """
    Core diagnostics computed from (saved model, long data).
    Returns a JSON-serializable dict.
    """
    cfg = cfg or PEDiagnosticsConfig()

    # Row-level predictions
    pred, _ = predict_long(long_df, model, return_X=False)

    # Deviance / Pearson-like residual summaries (row-level)
    y = long_df[model["config"]["exposure_col"]].to_numpy(dtype=float)
    d = long_df[model["config"]["event_col"]].to_numpy(dtype=float)
    mu = pred["mu"].to_numpy(dtype=float)

    mu_safe = np.maximum(mu, cfg.eps)

    # Pearson residual: (d - mu)/sqrt(mu)
    pearson = (d - mu_safe) / np.sqrt(mu_safe)

    # Deviance residual for Poisson (signed)
    # dev = 2 * [ d*log(d/mu) - (d-mu) ], with convention d*log(d/mu)=0 if d=0
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(d > 0, d * np.log(d / mu_safe), 0.0)
    dev = 2.0 * (term - (d - mu_safe))
    dev = np.sign(d - mu_safe) * np.sqrt(np.maximum(dev, 0.0))

    def _summ(x: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.mean(x)),
            "sd": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
            "q05": float(np.quantile(x, 0.05)),
            "q25": float(np.quantile(x, 0.25)),
            "q50": float(np.quantile(x, 0.50)),
            "q75": float(np.quantile(x, 0.75)),
            "q95": float(np.quantile(x, 0.95)),
        }

    # Subject-level Cox–Snell residuals r_i = H_i(T_i|x_i) = sum_k y_ik * hazard_ik
    cumhaz_df = predict_subject_cumhaz(long_df, model, id_col=cfg.id_col)
    sub = _subject_level_from_long(long_df, cfg.id_col)
    sub = sub.merge(cumhaz_df, on=cfg.id_col, how="left")
    if sub["cumhaz"].isna().any():
        raise RuntimeError("cumhaz missing for some subjects after merge (id mismatch).")

    r = sub["cumhaz"].to_numpy(dtype=float)
    event = sub["event"].to_numpy(dtype=int)

    # Cox–Snell summary: mean among events is often used as a quick scalar check
    r_events = r[event == 1]
    cs = {
        "n_subjects": int(len(sub)),
        "n_events": int(np.sum(event)),
        "mean_r_events": float(np.mean(r_events)) if len(r_events) else float("nan"),
        "var_r_events": float(np.var(r_events, ddof=1)) if len(r_events) > 1 else float("nan"),
        "mean_r_all": float(np.mean(r)),
    }

    out: Dict[str, Any] = {
        "row_level": {
            "n_rows": int(len(long_df)),
            "pearson": _summ(pearson),
            "deviance": _summ(dev),
        },
        "cox_snell": cs,
    }
    return out


def plot_residual_histograms(
    long_df: pd.DataFrame,
    model: Dict[str, Any],
    out_path: str | Path,
    cfg: PEDiagnosticsConfig | None = None,
) -> None:
    cfg = cfg or PEDiagnosticsConfig()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pred, _ = predict_long(long_df, model, return_X=False)
    d = long_df[model["config"]["event_col"]].to_numpy(dtype=float)
    mu = pred["mu"].to_numpy(dtype=float)
    mu_safe = np.maximum(mu, cfg.eps)
    pearson = (d - mu_safe) / np.sqrt(mu_safe)

    # downsample if huge
    x = pearson
    if len(x) > cfg.max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(x), size=cfg.max_points, replace=False)
        x = x[idx]

    plt.figure()
    plt.hist(x, bins=60)
    plt.title("Pearson residuals (Poisson PE long data)")
    plt.xlabel("Pearson residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_cox_snell(
    long_df: pd.DataFrame,
    model: Dict[str, Any],
    out_path: str | Path,
    cfg: PEDiagnosticsConfig | None = None,
) -> None:
    """
    Cox–Snell diagnostic: KM of residuals r_i, compare -log(S_hat(r)) vs r.
    If model fits well, points should lie near y=x.
    """
    cfg = cfg or PEDiagnosticsConfig()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cumhaz_df = predict_subject_cumhaz(long_df, model, id_col=cfg.id_col)
    sub = _subject_level_from_long(long_df, cfg.id_col).merge(cumhaz_df, on=cfg.id_col, how="left")

    r = sub["cumhaz"].to_numpy(dtype=float)
    event = sub["event"].to_numpy(dtype=int)

    t, S = _km_curve(r, event)
    S_safe = np.maximum(S, cfg.eps)
    y = -np.log(S_safe)

    plt.figure()
    plt.plot(t, y, marker="o", linestyle="none")
    # reference line y=x
    mx = float(np.max(t)) if len(t) else 1.0
    grid = np.linspace(0.0, mx, 200)
    plt.plot(grid, grid)
    plt.title("Cox–Snell residual check: -log(KM) vs residual")
    plt.xlabel("Cox–Snell residual r")
    plt.ylabel("-log( Ŝ(r) )")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
