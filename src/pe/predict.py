# src/pe/predict.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import patsy


def _rhs_from_formula(formula: str) -> str:
    """
    Convert 'd ~ C(k) + x1 + C(stage)' -> 'C(k) + x1 + C(stage)'.
    """
    if "~" not in formula:
        raise ValueError(f"Formula does not contain '~': {formula}")
    return formula.split("~", 1)[1].strip()


def _validate_model_dict(model: Dict[str, Any]) -> None:
    for key in ["config", "design_info", "params"]:
        if key not in model:
            raise ValueError(f"Model dict missing required key '{key}'")

    di = model["design_info"]
    if "exog_names" not in di or not isinstance(di["exog_names"], list):
        raise ValueError("model['design_info']['exog_names'] must exist and be a list")

    cfg = model["config"]
    for k in ["event_col", "exposure_col", "interval_col", "eps", "formula"]:
        if k not in cfg:
            raise ValueError(f"model['config'] missing '{k}'")


def _validate_new_data(long_df: pd.DataFrame, model: Dict[str, Any]) -> None:
    cfg = model["config"]
    need = [cfg["exposure_col"], cfg["interval_col"]]
    missing = [c for c in need if c not in long_df.columns]
    if missing:
        raise ValueError(f"New long_df missing required columns: {missing}")

    # Optional: validate categorical levels (warn/error on unseen)
    cat_levels = model["design_info"].get("categorical_levels", {}) or {}
    for c, levels in cat_levels.items():
        if c in long_df.columns:
            new_levels = set(pd.unique(long_df[c].dropna()).tolist())
            train_levels = set(levels)
            unseen = sorted(list(new_levels - train_levels))
            if unseen:
                raise ValueError(
                    f"Unseen levels in categorical '{c}': {unseen}. "
                    f"Training levels were: {sorted(list(train_levels))}"
                )


def _build_X(long_df: pd.DataFrame, model: Dict[str, Any]) -> pd.DataFrame:
    """
    Build design matrix from RHS of saved formula using patsy.
    Then align to training exog_names (adding missing columns as zeros).
    """
    cfg = model["config"]
    rhs = _rhs_from_formula(cfg["formula"])

    X = patsy.dmatrix(rhs, long_df, return_type="dataframe")

    # Align to training columns
    exog_names: List[str] = model["design_info"]["exog_names"]

    # Add missing columns as zeros
    missing_cols = [c for c in exog_names if c not in X.columns]
    for c in missing_cols:
        X[c] = 0.0

    # Drop unexpected columns (can happen if a categorical has extra levels, but we already check)
    X = X[exog_names]

    # Ensure numeric dtype
    X = X.astype(float)

    return X


def predict_long(
    long_df: pd.DataFrame,
    model: Dict[str, Any],
    return_X: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Predict on long-format dataframe (output of expand_long).

    Returns:
      pred_df with columns:
        - eta: linear predictor excluding offset
        - hazard: exp(eta)
        - mu: y * exp(eta)
      and optionally X (design matrix) if return_X=True.
    """
    _validate_model_dict(model)
    _validate_new_data(long_df, model)

    cfg = model["config"]
    y_col = cfg["exposure_col"]
    eps = float(cfg["eps"])

    X = _build_X(long_df, model)

    params = model["params"]
    # params vector aligned with X columns
    beta = np.array([float(params[name]) for name in X.columns], dtype=float)

    eta = X.to_numpy() @ beta
    hazard = np.exp(eta)

    y = long_df[y_col].to_numpy(dtype=float)
    y_safe = np.maximum(y, eps)
    mu = y_safe * hazard

    pred = pd.DataFrame(
        {
            "eta": eta,
            "hazard": hazard,
            "mu": mu,
        },
        index=long_df.index,
    )

    return (pred, X) if return_X else (pred, None)


def predict_subject_cumhaz(
    long_df: pd.DataFrame,
    model: Dict[str, Any],
    id_col: str = "__row_id__",
) -> pd.DataFrame:
    """
    Compute subject-level cumulative hazard H_i = sum_k y_ik * hazard_ik
    using long-format rows.

    Returns a dataframe with:
      - id_col
      - cumhaz
    """
    if id_col not in long_df.columns:
        raise ValueError(f"long_df must contain id_col='{id_col}' for subject aggregation")

    pred, _ = predict_long(long_df, model, return_X=False)

    tmp = long_df[[id_col, model["config"]["exposure_col"]]].copy()
    tmp["hazard"] = pred["hazard"].to_numpy()
    tmp["increment"] = tmp[model["config"]["exposure_col"]].astype(float) * tmp["hazard"]

    out = tmp.groupby(id_col, as_index=False)["increment"].sum().rename(columns={"increment": "cumhaz"})
    return out
