from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class DiagnosticsConfig:
    id_col: str = "id"
    k_col: str = "k"
    y_col: str = "y"
    d_col: str = "d"
    # for Cox–Snell KM calc, we need per-subject event indicator; if long has only d, we compute event=sum(d)>0
    # optional stratification for observed/expected:
    group_col: Optional[str] = None


def _check_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def compute_row_predictions(long_df: pd.DataFrame, model: dict) -> pd.DataFrame:
    """
    Use the model's predict_long to compute per-row predictions for the observed long data.
    Expects predict_long to return a DataFrame with at least:
      - hazard (per unit time)
      - mu (expected count) OR (we compute mu = y * hazard if mu absent)
      - eta (optional)
    """
    from pe.predict import predict_long  # local import to avoid circular issues

    pred, _X = predict_long(long_df, model, return_X=False)
    if not isinstance(pred, pd.DataFrame):
        raise TypeError("predict_long must return a pandas DataFrame as its first return value")

    if "hazard" not in pred.columns:
        raise ValueError("predict_long output missing required column 'hazard'")

    out = pred.copy()
    if "mu" not in out.columns:
        # mu = y * hazard
        if "y" not in long_df.columns:
            raise ValueError("long_df missing 'y' needed to compute mu = y * hazard")
        out["mu"] = long_df["y"].to_numpy(dtype=float) * out["hazard"].to_numpy(dtype=float)

    return out


def pearson_residuals(long_df: pd.DataFrame, pred: pd.DataFrame, *, cfg: DiagnosticsConfig) -> np.ndarray:
    """
    Pearson residuals for Poisson counts: (d - mu) / sqrt(mu)
    """
    _check_cols(long_df, [cfg.d_col])
    if "mu" not in pred.columns:
        raise ValueError("pred missing 'mu'")

    d = long_df[cfg.d_col].to_numpy(dtype=float)
    mu = pred["mu"].to_numpy(dtype=float)
    mu_safe = np.clip(mu, 1e-12, np.inf)
    return (d - mu_safe) / np.sqrt(mu_safe)


def deviance_residuals(long_df: pd.DataFrame, pred: pd.DataFrame, *, cfg: DiagnosticsConfig) -> np.ndarray:
    """
    Poisson deviance residuals (signed).
    """
    _check_cols(long_df, [cfg.d_col])
    if "mu" not in pred.columns:
        raise ValueError("pred missing 'mu'")

    d = long_df[cfg.d_col].to_numpy(dtype=float)
    mu = pred["mu"].to_numpy(dtype=float)
    mu_safe = np.clip(mu, 1e-12, np.inf)

    # deviance contribution: 2 * [ d*log(d/mu) - (d - mu) ], with convention d=0 => 2*mu
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(d > 0, d * np.log(d / mu_safe) - (d - mu_safe), mu_safe)
    dev = 2.0 * term
    # sign from (d - mu)
    return np.sign(d - mu_safe) * np.sqrt(np.clip(dev, 0.0, np.inf))


def observed_expected_by_interval(
    long_df: pd.DataFrame, pred: pd.DataFrame, *, cfg: DiagnosticsConfig
) -> pd.DataFrame:
    """
    Interval-level observed vs expected:
      obs_k = sum_i d_ik
      exp_k = sum_i mu_ik
      ratio = obs/exp
    Optionally stratify by cfg.group_col if present.
    """
    _check_cols(long_df, [cfg.k_col, cfg.d_col])
    if "mu" not in pred.columns:
        raise ValueError("pred missing 'mu'")

    df = pd.DataFrame(
        {
            cfg.k_col: long_df[cfg.k_col].to_numpy(dtype=int),
            "d": long_df[cfg.d_col].to_numpy(dtype=float),
            "mu": pred["mu"].to_numpy(dtype=float),
        }
    )
    if cfg.group_col is not None:
        _check_cols(long_df, [cfg.group_col])
        df[cfg.group_col] = long_df[cfg.group_col].astype(str).to_numpy()

        tab = (
            df.groupby([cfg.k_col, cfg.group_col], as_index=False)
            .agg(obs=("d", "sum"), exp=("mu", "sum"))
            .sort_values([cfg.k_col, cfg.group_col])
        )
    else:
        tab = (
            df.groupby(cfg.k_col, as_index=False)
            .agg(obs=("d", "sum"), exp=("mu", "sum"))
            .sort_values(cfg.k_col)
        )

    tab["ratio_obs_over_exp"] = np.where(tab["exp"] > 0, tab["obs"] / tab["exp"], np.nan)
    tab["diff_obs_minus_exp"] = tab["obs"] - tab["exp"]
    return tab


def _km_survival(times: np.ndarray, events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Kaplan–Meier survival estimate evaluated at unique event times.
    Returns:
      t_event, S_hat(t_event)
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)

    event_times = np.unique(times[events == 1])
    if event_times.size == 0:
        return np.array([0.0]), np.array([1.0])

    S = 1.0
    S_vals = []
    for tj in event_times:
        n_risk = np.sum(times >= tj)
        d = np.sum((times == tj) & (events == 1))
        if n_risk > 0:
            S *= (1.0 - d / n_risk)
        S_vals.append(S)

    return event_times, np.asarray(S_vals, dtype=float)


def cox_snell_residuals(long_df: pd.DataFrame, pred: pd.DataFrame, *, cfg: DiagnosticsConfig) -> pd.DataFrame:
    """
    Cox–Snell residuals r_i = \hat H_i(T_i) computed from observed long rows:
      r_i = sum_k y_ik * hazard_ik   (over observed intervals for subject i)
    where hazard_ik is the model hazard for that interval and subject.

    Returns:
      per-subject dataframe with columns:
        id, r, event
    """
    _check_cols(long_df, [cfg.id_col, cfg.y_col, cfg.d_col])
    if "hazard" not in pred.columns:
        raise ValueError("pred missing 'hazard'")

    df = pd.DataFrame(
        {
            cfg.id_col: long_df[cfg.id_col].to_numpy(dtype=int),
            "y": long_df[cfg.y_col].to_numpy(dtype=float),
            "hazard": pred["hazard"].to_numpy(dtype=float),
            "d": long_df[cfg.d_col].to_numpy(dtype=int),
        }
    )

    agg = df.groupby(cfg.id_col, as_index=False).agg(
        r=("y", lambda x: 0.0),  # placeholder; replaced below
        event=("d", "sum"),
    )
    # compute r via groupby sum of y*hazard without creating huge intermediate copies
    df["_inc"] = df["y"] * df["hazard"]
    r = df.groupby(cfg.id_col)["_inc"].sum()
    agg["r"] = agg[cfg.id_col].map(r).to_numpy(dtype=float)
    agg["event"] = (agg["event"] > 0).astype(int)
    return agg[[cfg.id_col, "r", "event"]]


def plot_pearson_hist(resid: np.ndarray, out_png: str, *, title: str = "Pearson residuals") -> None:
    plt.figure()
    plt.hist(np.asarray(resid, dtype=float), bins=60)
    plt.xlabel("Pearson residual")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_cox_snell(cs: pd.DataFrame, out_png: str, *, title: str = "Cox–Snell residual check") -> None:
    """
    Plot -log(KM(r)) vs r; should lie near y=x if the model fits well.
    """
    r = cs["r"].to_numpy(dtype=float)
    e = cs["event"].to_numpy(dtype=int)

    t_ev, S = _km_survival(r, e)
    S_safe = np.clip(S, 1e-12, 1.0)
    y = -np.log(S_safe)
    x = t_ev

    plt.figure()
    plt.scatter(x, y, s=18)
    # reference line y=x over plotted range
    mx = float(np.nanmax(x)) if x.size else 1.0
    plt.plot([0.0, mx], [0.0, mx])
    plt.xlabel("Cox–Snell residual r")
    plt.ylabel("-log(KM S(r))")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_obs_exp(tab: pd.DataFrame, out_png: str, *, cfg: DiagnosticsConfig, title: str) -> None:
    """
    Plot observed vs expected by interval (two curves).
    If cfg.group_col is set, plots separate curves for each group.
    """
    plt.figure()

    if cfg.group_col is None:
        x = tab[cfg.k_col].to_numpy(dtype=int)
        plt.plot(x, tab["obs"].to_numpy(dtype=float), marker="o", label="Observed")
        plt.plot(x, tab["exp"].to_numpy(dtype=float), marker="o", label="Expected")
        plt.xlabel("Interval k")
        plt.ylabel("Events (sum over long rows)")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        return

    # stratified
    for gval, gdf in tab.groupby(cfg.group_col):
        x = gdf[cfg.k_col].to_numpy(dtype=int)
        plt.plot(x, gdf["ratio_obs_over_exp"].to_numpy(dtype=float), marker="o", label=str(gval))

    plt.xlabel("Interval k")
    plt.ylabel("Observed / Expected")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_ratio_by_interval(tab: pd.DataFrame, out_png: str, *, cfg: DiagnosticsConfig, title: str) -> None:
    """
    Plot ratio obs/exp by interval (single curve).
    For stratified tab, this plots one curve per group.
    """
    plt.figure()

    if cfg.group_col is None:
        x = tab[cfg.k_col].to_numpy(dtype=int)
        plt.plot(x, tab["ratio_obs_over_exp"].to_numpy(dtype=float), marker="o")
        plt.axhline(1.0)
        plt.xlabel("Interval k")
        plt.ylabel("Observed / Expected")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        return

    for gval, gdf in tab.groupby(cfg.group_col):
        x = gdf[cfg.k_col].to_numpy(dtype=int)
        plt.plot(x, gdf["ratio_obs_over_exp"].to_numpy(dtype=float), marker="o", label=str(gval))
    plt.axhline(1.0)
    plt.xlabel("Interval k")
    plt.ylabel("Observed / Expected")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()