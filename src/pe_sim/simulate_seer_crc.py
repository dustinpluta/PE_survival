from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .seer_crc_covariates import SEERCRCSimSpec, draw_seer_crc_covariates, categorical_log_hr_offset
from .censoring import tune_exponential_censor_rate

# NEXT FILE WE'LL BUILD:
# from .pe_time import simulate_pe_times


@dataclass(frozen=True)
class PEParams:
    """
    PE baseline and numeric covariate coefficients.

    breaks: (K+1,) increasing, in days
    lam:    (K,) baseline hazards for each interval
    beta_num: (4,) for [age_per10_centered, cci, log_tumor_size_mm, ses]
    """
    breaks: np.ndarray
    lam: np.ndarray
    beta_num: np.ndarray


def simulate_seer_crc_pe_dataset(
    n: int,
    pe: PEParams,
    cov_spec: Optional[SEERCRCSimSpec] = None,
    target_censor_frac: float = 0.30,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Simulate subject-level survival data resembling SEER–Medicare CRC, using a PE DGP.

    Output columns:
      time, event, plus covariates:
      age, cci, tumor_size_mm, ses, sex, race, stage, grade, surgery, chemo, radiation, dx_year
    """
    cov_spec = cov_spec or SEERCRCSimSpec(seed=seed)
    rng = np.random.default_rng(seed)

    breaks = np.asarray(pe.breaks, float)
    lam = np.asarray(pe.lam, float)
    beta_num = np.asarray(pe.beta_num, float)

    if len(breaks) != len(lam) + 1:
        raise ValueError("len(breaks) must equal len(lam)+1")
    if len(beta_num) != 4:
        raise ValueError("beta_num must have length 4")

    df_cov, X_num = draw_seer_crc_covariates(n=n, rng=rng, spec=cov_spec)
    offset = categorical_log_hr_offset(df_cov)

    # linear predictor
    eta = X_num @ beta_num + offset

    # ---- event time simulation (to be implemented next) ----
    from .pe_time import simulate_pe_times
    T_event = simulate_pe_times(breaks=breaks, lam=lam, eta=eta, rng=rng)

    # ---- censoring tuned to target rate ----
    censor_rate = tune_exponential_censor_rate(rng=rng, T_event=T_event, target_censor_frac=target_censor_frac)
    C = rng.exponential(scale=1.0 / censor_rate, size=n)

    T_obs = np.minimum(T_event, C)
    event = (T_event <= C).astype(int)

    df = df_cov.copy()
    df.insert(0, "event", event)
    df.insert(0, "time", T_obs)

    return df
