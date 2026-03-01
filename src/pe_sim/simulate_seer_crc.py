from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PEParams:
    """
    Piecewise exponential parameters.

    breaks: length K+1, strictly increasing, in DAYS.
    lam:    length K, baseline hazard per DAY for each interval [breaks[k], breaks[k+1]).
    beta_num: length 4 numeric betas for:
              age_per10_centered, cci, tumor_size_log, ses
    """
    breaks: np.ndarray
    lam: np.ndarray
    beta_num: np.ndarray


def _check_pe_params(pe: PEParams) -> None:
    b = np.asarray(pe.breaks, float)
    lam = np.asarray(pe.lam, float)
    beta = np.asarray(pe.beta_num, float)

    if b.ndim != 1 or lam.ndim != 1 or beta.ndim != 1:
        raise ValueError("breaks/lam/beta_num must be 1D arrays")
    if len(b) < 2:
        raise ValueError("breaks must have length >= 2")
    if not np.all(np.diff(b) > 0):
        raise ValueError("breaks must be strictly increasing (days)")
    K = len(b) - 1
    if len(lam) != K:
        raise ValueError(f"lam must have length K={K}, got {len(lam)}")
    if np.any(lam <= 0):
        raise ValueError("lam must be positive (per-day hazard)")
    if len(beta) != 4:
        raise ValueError("beta_num must be length 4")


def _simulate_covariates(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    SEER-like colorectal covariates (synthetic; plausible structure).
    """
    # Numeric
    age = rng.normal(loc=70.0, scale=9.0, size=n)  # years
    age = np.clip(age, 40, 95)
    age_per10_centered = (age - 70.0) / 10.0

    cci = rng.poisson(lam=1.2, size=n).astype(float)
    cci = np.clip(cci, 0, 8)

    tumor_size_mm = rng.lognormal(mean=np.log(35.0), sigma=0.45, size=n)
    tumor_size_mm = np.clip(tumor_size_mm, 5, 150)
    tumor_size_log = np.log(tumor_size_mm)

    ses = rng.normal(loc=0.0, scale=1.0, size=n)

    # Categorical
    sex = rng.choice(["F", "M"], size=n, p=[0.52, 0.48])
    stage = rng.choice(["I", "II", "III", "IV"], size=n, p=[0.25, 0.35, 0.28, 0.12])

    return pd.DataFrame(
        {
            "age_per10_centered": age_per10_centered,
            "cci": cci,
            "tumor_size_log": tumor_size_log,
            "ses": ses,
            "sex": sex,
            "stage": stage,
        }
    )


def _linear_predictor(df: pd.DataFrame, beta_num: np.ndarray) -> np.ndarray:
    x_num = df[["age_per10_centered", "cci", "tumor_size_log", "ses"]].to_numpy(float)
    eta = x_num @ beta_num

    # Fixed PH effects for categorical covariates (baseline categories: sex=F, stage=I)
    stage_effect = {"I": 0.0, "II": 0.45, "III": 0.95, "IV": 1.55}
    sex_effect = {"F": 0.0, "M": 0.10}

    eta = eta + df["stage"].map(stage_effect).to_numpy(float)
    eta = eta + df["sex"].map(sex_effect).to_numpy(float)
    return eta


def _simulate_event_times_piecewise_exp(
    pe: PEParams,
    eta: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate event time T (in DAYS) under piecewise exponential hazard:
      h(t|x) = lam[k] * exp(eta) for t in interval k.
    """
    b = np.asarray(pe.breaks, float)
    lam0 = np.asarray(pe.lam, float)
    K = len(b) - 1
    n = len(eta)

    mult = np.exp(eta)
    T = np.empty(n, dtype=float)

    for i in range(n):
        t = 0.0
        for k in range(K):
            a = b[k]
            c = b[k + 1]
            width = c - a

            rate = lam0[k] * mult[i]  # per-day
            w = rng.exponential(scale=1.0 / rate)

            if w < width:
                t = a + w
                break
            else:
                t = c
        T[i] = t

    return T


def _solve_censor_rate_for_target(
    T: np.ndarray,
    followup_end: float,
    target_censor_frac: float,
    rng: np.random.Generator,
    p_admin: float,
    iters: int = 35,
) -> float:
    """
    Choose exponential censoring rate nu (per day) so that overall censor fraction
    is near target, using bisection on nu, under *mixture censoring*:

      With probability p_admin: C = followup_end (administrative censoring mass)
      With probability 1-p_admin: C ~ Exp(nu)

    Observed time = min(T, C, followup_end)
    Event = 1{T <= C and T < followup_end}   [strict at horizon]
    """
    if not (0.0 < target_censor_frac < 1.0):
        raise ValueError("target_censor_frac must be in (0,1)")
    if not (0.0 <= p_admin < 1.0):
        raise ValueError("p_admin must be in [0,1)")

    lo, hi = 1e-8, 0.2  # per-day bounds

    # Fixed uniforms for stable root finding
    u_admin = rng.uniform(size=T.shape[0])
    admin = (u_admin < p_admin)
    u_exp = rng.uniform(size=T.shape[0])

    def censor_frac_at(nu: float) -> float:
        C = np.empty_like(T, dtype=float)
        C[admin] = followup_end
        C[~admin] = -np.log(u_exp[~admin]) / nu

        event = ((T <= C) & (T < followup_end)).astype(int)
        return float(1.0 - event.mean())

    f_lo = censor_frac_at(lo)
    f_hi = censor_frac_at(hi)

    # If even extreme censoring doesn't achieve target, return hi.
    if f_hi < target_censor_frac:
        return hi
    # If even almost-zero censoring exceeds target, return lo.
    if f_lo > target_censor_frac:
        return lo

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid = censor_frac_at(mid)
        if f_mid > target_censor_frac:
            hi = mid  # too much censoring
        else:
            lo = mid  # too little censoring

    return 0.5 * (lo + hi)


def simulate_seer_crc_pe_dataset(
    n: int,
    pe: PEParams,
    target_censor_frac: float,
    seed: int = 123,
    p_admin: float = 0.22,
) -> pd.DataFrame:
    """
    Generate SEER-like CRC survival data under a PH piecewise exponential model.

    Output columns (base / non-long):
      id (int), time (days, float), event (0/1 int),
      age_per10_centered, cci, tumor_size_log, ses, sex, stage

    Debug columns included:
      _T_true, _C_true, _eta_true, _admin_cens, _nu_true
    """
    _check_pe_params(pe)
    rng = np.random.default_rng(seed)

    cov = _simulate_covariates(n, rng)
    eta = _linear_predictor(cov, np.asarray(pe.beta_num, float))
    followup_end = float(np.asarray(pe.breaks, float)[-1])

    # Event times (days)
    T = _simulate_event_times_piecewise_exp(pe, eta, rng)

    # Tune random censoring rate under mixture censoring
    rng_c = np.random.default_rng(seed + 999)
    nu = _solve_censor_rate_for_target(
        T=T,
        followup_end=followup_end,
        target_censor_frac=target_censor_frac,
        rng=rng_c,
        p_admin=p_admin,
    )

    # Simulate censor times with mixture: admin mass at followup_end + Exp(nu)
    rng_c2 = np.random.default_rng(seed + 1001)
    u_admin = rng_c2.uniform(size=n)
    admin = (u_admin < p_admin)

    C = np.empty(n, dtype=float)
    C[admin] = followup_end
    C[~admin] = rng_c2.exponential(scale=1.0 / nu, size=(~admin).sum())

    # Observed time (days)
    time = np.minimum(np.minimum(T, C), followup_end)

    # Strict horizon rule prevents events recorded at exactly followup_end
    event = ((T <= C) & (T < followup_end)).astype(int)

    df = cov.copy()
    df.insert(0, "id", np.arange(n, dtype=int))
    df["time"] = time.astype(float)
    df["event"] = event.astype(int)

    # Debug columns
    df["_T_true"] = T.astype(float)
    df["_C_true"] = C.astype(float)
    df["_eta_true"] = eta.astype(float)
    df["_admin_cens"] = admin.astype(int)
    df["_nu_true"] = float(nu)

    return df


# Optional quick runner
if __name__ == "__main__":
    breaks = np.array(
        [0, 30, 60, 90, 120, 150, 180, 240, 300, 365, 455, 545, 635, 725,
         815, 905, 995, 1085, 1175, 1265, 1355, 1445, 1535, 1625, 1715, 1805, 1825],
        dtype=float,
    )
    lam = np.array(
        [0.0012, 0.0011, 0.0010, 0.0010, 0.00095, 0.0009, 0.00085, 0.0008, 0.00075, 0.0007,
         0.00065, 0.00062, 0.00060, 0.00058, 0.00056, 0.00054, 0.00052, 0.00050, 0.00048, 0.00046,
         0.00044, 0.00042, 0.00040, 0.00038, 0.00036, 0.00034],
        dtype=float,
    )
    beta_num = np.array([0.10, 0.12, 0.18, -0.07], dtype=float)

    pe = PEParams(breaks=breaks, lam=lam, beta_num=beta_num)
    df = simulate_seer_crc_pe_dataset(n=5000, pe=pe, target_censor_frac=0.30, seed=123, p_admin=0.12)

    print(df.head())
    print("event rate:", df["event"].mean())
    print("censor frac:", 1.0 - df["event"].mean())
    print("n censored at 1825:", int(((df["time"] == breaks[-1]) & (df["event"] == 0)).sum()))