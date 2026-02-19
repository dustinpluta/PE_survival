# src/pe/fit_minimal.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from math import exp
from scipy.stats import norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from dataclasses import asdict, dataclass
from pathlib import Path
import json

# Increment when you change the returned model dictionary schema
SCHEMA_VERSION = "pe_fit.v3"

@dataclass(frozen=True)
class PEFitMinimalConfig:
    event_col: str = "d"
    exposure_col: str = "y"
    interval_col: str = "k"

    # y can be 0 for rare T==0 events; we floor for log-offset stability
    eps: float = 1e-12


def _validate_long_df(df: pd.DataFrame, cfg: PEFitMinimalConfig) -> None:
    need = [cfg.event_col, cfg.exposure_col, cfg.interval_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in long df: {missing}")

    y = df[cfg.exposure_col].to_numpy(float)
    if np.any(~np.isfinite(y)) or np.any(y < 0):
        raise ValueError("Exposure y must be finite and nonnegative")

    dvals = set(pd.unique(df[cfg.event_col].dropna()))
    if not dvals.issubset({0, 1}):
        raise ValueError(f"Event d must be coded 0/1; found {sorted(dvals)}")

def _validate_covariates(df: pd.DataFrame, covariates: Sequence[str], categorical: Sequence[str]) -> None:
    covariates = list(covariates)
    missing = [c for c in covariates if c not in df.columns]
    if missing:
        raise ValueError(f"Missing covariates in long df: {missing}")

    categorical = list(categorical)
    for c in categorical:
        if c not in covariates:
            raise ValueError(f"Categorical '{c}' must be included in covariates list")

def _extract_baseline_theta_lambda(
    params: Dict[str, float],
    interval_col: str,
    interval_levels: List[Any],
) -> Dict[str, Any]:
    """
    Reconstruct per-interval baseline log-hazards theta_k and hazards lambda_k
    from statsmodels formula params using C(interval_col).

    Assumes standard treatment coding:
      - Intercept corresponds to reference level (typically smallest level)
      - Other levels appear as C(k)[T.<level>]
    """
    if "Intercept" not in params:
        raise RuntimeError("Expected 'Intercept' in params (formula fit should include an intercept).")

    intercept = float(params["Intercept"])

    # Reference level is the first level in the sorted interval_levels list
    ref = interval_levels[0]

    theta_by_level: Dict[Any, float] = {}
    theta_by_level[ref] = intercept

    # Fill in remaining levels
    for lvl in interval_levels[1:]:
        term = f"C({interval_col})[T.{lvl}]"
        theta_by_level[lvl] = intercept + float(params.get(term, 0.0))

    # Convert to ordered vectors aligned with interval_levels
    theta = [theta_by_level[lvl] for lvl in interval_levels]
    lam = [float(np.exp(t)) for t in theta]
    return {
        "reference_level": ref,
        "theta_by_level": {str(k): float(v) for k, v in theta_by_level.items()},
        "theta": theta,
        "lambda": lam,
    }

def _build_inference_table(
    params: Dict[str, float],
    bse: Dict[str, float],
    interval_col: str,
) -> List[Dict[str, Any]]:
    """
    Build inference table for covariate terms only (exclude baseline C(k) terms).
    Returns list of row dicts.
    """
    rows: List[Dict[str, Any]] = []

    for term, beta in params.items():

        # Skip baseline terms
        if term == "Intercept":
            continue
        if term.startswith(f"C({interval_col})"):
            continue

        se = float(bse.get(term, float("nan")))
        z = float(beta / se) if se > 0 else float("nan")
        p = float(2 * (1 - norm.cdf(abs(z)))) if se > 0 else float("nan")

        hr = float(exp(beta))
        ci_low = float(exp(beta - 1.96 * se)) if se > 0 else float("nan")
        ci_high = float(exp(beta + 1.96 * se)) if se > 0 else float("nan")

        rows.append(
            {
                "term": term,
                "beta": float(beta),
                "se": se,
                "z": z,
                "p_value": p,
                "hazard_ratio": hr,
                "ci_lower_95": ci_low,
                "ci_upper_95": ci_high,
            }
        )
    return rows

    import json

def fit_pe_minimal(
    long_df: pd.DataFrame,
    covariates: Sequence[str],
    categorical: Optional[Sequence[str]] = None,
    cfg: Optional[PEFitMinimalConfig] = None,
) -> Dict[str, Any]:
    """
    Minimal PE fit:
      d ~ Poisson( y * exp( theta_k + x'beta ) )
    using GLM with log link:
      log E[d] = log(y) + C(k) + covariates
    """
    cfg = cfg or PEFitMinimalConfig()
    _validate_long_df(long_df, cfg)

    covariates = list(covariates)
    categorical_list = list(categorical or [])
    _validate_covariates(long_df, covariates=covariates, categorical=categorical_list)

    df = long_df.copy()

    # stable offset
    y = df[cfg.exposure_col].astype(float).to_numpy()
    df["_log_y"] = np.log(np.maximum(y, cfg.eps))

    missing = [c for c in covariates if c not in df.columns]
    if missing:
        raise ValueError(f"Missing covariates in long df: {missing}")

    categorical = set(categorical or [])
    for c in categorical:
        if c not in covariates:
            raise ValueError(f"Categorical '{c}' must be included in covariates list")

    # Build formula: d ~ C(k) + ...
    terms: List[str] = [f"C({cfg.interval_col})"]
    categorical_set = set(categorical_list)
    for v in covariates:
        terms.append(f"C({v})" if v in categorical_set else v)

    formula = f"{cfg.event_col} ~ " + " + ".join(terms)
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Poisson(),
        offset=df["_log_y"],
    )
    res = model.fit()

    # Column names of the fitted design matrix (includes intercept and dummy columns).
    exog_names = list(res.model.exog_names)
    endog_name = str(res.model.endog_names)

    # Record levels observed in training (useful for validation at predict-time).
    interval_levels = sorted(pd.unique(df[cfg.interval_col]).tolist())

    categorical_levels = {}
    for c in categorical:
        # keep stable ordering
        categorical_levels[c] = sorted(pd.unique(df[c]).tolist())

    # Core outputs
    params = res.params.to_dict()
    bse = res.bse.to_dict()

    inference_table = _build_inference_table(
    params=params,
    bse=bse,
    interval_col=cfg.interval_col,
    )

    baseline = _extract_baseline_theta_lambda(
        params=params,
        interval_col=cfg.interval_col,
        interval_levels=interval_levels,
    )
    # Standardized artifact
    out: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "config": {
            **asdict(cfg),
            "covariates": covariates,
            "categorical": sorted(list(categorical_set)),
            "formula": formula,
        },
        "fit_stats": {
            "n_rows": int(len(df)),
            "n_events": int(df[cfg.event_col].sum()),
            "aic": float(res.aic),
            "deviance": float(res.deviance),
            "null_deviance": float(getattr(res, "null_deviance", np.nan)),
            "df_model": int(getattr(res, "df_model", np.nan)),
            "df_resid": int(getattr(res, "df_resid", np.nan)),
            "converged": bool(getattr(res, "converged", True)),
        },
        "params": params,
        "inference": {
            "bse": bse,
            "covariate_table": inference_table,
        },
        "design_info": {
            "exog_names": exog_names,
            "endog_name": endog_name,
            "interval_col": cfg.interval_col,
            "interval_levels": interval_levels,
            "categorical_levels": categorical_levels,
        },
        "baseline": {
            **baseline,
            "K": int(len(interval_levels)),
        },
        "summary": res.summary().as_text(),
    }
    return out
