from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SEERCRCSimSpec:
    """
    SEER–Medicare-like colorectal cancer covariate generation spec.

    Numeric covariates returned in X_num correspond to beta_num:
      [age_per10_centered, cci, log_tumor_size_mm, ses]
    """
    # cohort size handled elsewhere
    seed: int = 123

    # age distribution (Medicare-ish)
    age_mean: float = 74.0
    age_sd: float = 6.5
    age_min: float = 65.0
    age_max: float = 95.0

    # comorbidity (CCI-like)
    cci_poisson_mean: float = 1.2
    cci_max: float = 10.0

    # tumor size distribution (mm)
    tumor_size_lognorm_mean_mm: float = 35.0
    tumor_size_lognorm_sigma: float = 0.45
    tumor_size_min_mm: float = 5.0
    tumor_size_max_mm: float = 150.0

    # SES (standardized)
    ses_mean: float = 0.0
    ses_sd: float = 1.0

    # categorical distributions
    p_female: float = 0.52
    race_levels: Tuple[str, ...] = ("White", "Black", "Hispanic", "Asian", "Other")
    race_probs: Tuple[float, ...] = (0.72, 0.12, 0.09, 0.04, 0.03)

    stage_levels: Tuple[str, ...] = ("I", "II", "III", "IV")
    stage_probs: Tuple[float, ...] = (0.22, 0.33, 0.28, 0.17)

    grade_levels: Tuple[str, ...] = ("1", "2", "3", "4")
    grade_probs: Tuple[float, ...] = (0.18, 0.42, 0.30, 0.10)

    # diagnosis year range
    dx_year_min: int = 2008
    dx_year_max: int = 2015


def draw_seer_crc_covariates(
    n: int,
    rng: np.random.Generator,
    spec: SEERCRCSimSpec,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Draw SEER–Medicare-like CRC covariates.

    Returns
    -------
    df_cov : DataFrame
        Raw covariates (numeric + categorical + treatments + dx_year).
    X_num : ndarray (n, 4)
        Numeric design matrix for beta_num:
          [age_per10_centered, cci, log_tumor_size_mm, ses]
    """
    # Numeric
    age = np.clip(
        rng.normal(loc=spec.age_mean, scale=spec.age_sd, size=n),
        spec.age_min,
        spec.age_max,
    )

    cci = rng.poisson(lam=spec.cci_poisson_mean, size=n).astype(float)
    cci = np.clip(cci, 0.0, spec.cci_max)

    tumor_size_mm = rng.lognormal(mean=np.log(spec.tumor_size_lognorm_mean_mm),
                                  sigma=spec.tumor_size_lognorm_sigma,
                                  size=n)
    tumor_size_mm = np.clip(tumor_size_mm, spec.tumor_size_min_mm, spec.tumor_size_max_mm)
    tumor_size_log = np.log(tumor_size_mm)

    ses = rng.normal(loc=spec.ses_mean, scale=spec.ses_sd, size=n)

    # Categorical
    sex = rng.choice(["F", "M"], size=n, p=[spec.p_female, 1.0 - spec.p_female])
    race = rng.choice(list(spec.race_levels), size=n, p=list(spec.race_probs))
    stage = rng.choice(list(spec.stage_levels), size=n, p=list(spec.stage_probs))
    grade = rng.choice(list(spec.grade_levels), size=n, p=list(spec.grade_probs))

    dx_year = rng.integers(spec.dx_year_min, spec.dx_year_max + 1, size=n)

    # Treatments (correlated with stage; simple association, not causal)
    p_chemo = np.where(stage == "I", 0.05,
               np.where(stage == "II", 0.15,
               np.where(stage == "III", 0.55, 0.65)))
    p_rt = np.where(stage == "I", 0.02,
            np.where(stage == "II", 0.04,
            np.where(stage == "III", 0.08, 0.12)))

    chemo = rng.binomial(1, p_chemo, size=n).astype(int)
    radiation = rng.binomial(1, p_rt, size=n).astype(int)
    surgery = rng.binomial(1, 0.85, size=n).astype(int)

    age_per10_centered = (age - spec.age_mean) / 10.0
    X_num = np.column_stack([age_per10_centered, cci, tumor_size_log, ses]).astype(float)
    
    df_cov = pd.DataFrame({
        "age": age,
        "age_per10_centered": age_per10_centered,
        "cci": cci,
        "tumor_size_mm": tumor_size_mm,
        "tumor_size_log": tumor_size_log,
        "ses": ses,
        "sex": sex,
        "race": race,
        "stage": stage,
        "grade": grade,
        "surgery": surgery,
        "chemo": chemo,
        "radiation": radiation,
        "dx_year": dx_year.astype(int),
    })
    return df_cov, X_num


def categorical_log_hr_offset(df_cov: pd.DataFrame) -> np.ndarray:
    """
    Fixed 'true' log-HR offsets for categorical/binary covariates.
    Keep this separate so it’s easy to tweak the DGP later.
    """
    sex_eff = {"F": 0.0, "M": 0.08}

    race_eff = {
        "White": 0.0,
        "Black": 0.12,
        "Hispanic": -0.02,
        "Asian": -0.05,
        "Other": 0.03,
    }

    stage_eff = {"I": 0.0, "II": 0.30, "III": 0.70, "IV": 1.25}
    grade_eff = {"1": 0.0, "2": 0.12, "3": 0.28, "4": 0.45}

    surgery_beta = -0.35
    chemo_beta = -0.08
    radiation_beta = -0.05

    year_beta = -0.06
    year_center = (df_cov["dx_year"].to_numpy(int) - 2011) / 5.0

    return (
        df_cov["sex"].map(sex_eff).to_numpy(float)
        + df_cov["race"].map(race_eff).to_numpy(float)
        + df_cov["stage"].map(stage_eff).to_numpy(float)
        + df_cov["grade"].map(grade_eff).to_numpy(float)
        + surgery_beta * df_cov["surgery"].to_numpy(int)
        + chemo_beta * df_cov["chemo"].to_numpy(int)
        + radiation_beta * df_cov["radiation"].to_numpy(int)
        + year_beta * year_center.astype(float)
    )
