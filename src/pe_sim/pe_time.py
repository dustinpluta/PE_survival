# src/pe_sim/pe_time.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


DEFAULT_FOLLOWUP_DAYS = 1825.0  # 5 years


def default_calendar_breaks_5y() -> np.ndarray:
    """
    Fixed, interpretable calendar breaks in DAYS up to 5 years (1825 days).

    Hybrid resolution:
      - monthly (30d) in first 6 months
      - every 60d to 1 year
      - quarterly (90d) to 3 years
      - semiannual (180d) to 5 years
    """
    b1 = np.arange(0, 180 + 30, 30)           # 0..180 by 30
    b2 = np.arange(180, 365 + 60, 60)         # 180..360 by 60
    b3 = np.arange(365, 1095 + 90, 90)        # 1y..3y by 90
    b4 = np.arange(1095, 1825 + 180, 180)     # 3y..5y by 180

    breaks = np.unique(np.concatenate([b1, b2, b3, b4, np.array([DEFAULT_FOLLOWUP_DAYS])])).astype(float)
    breaks = breaks[(breaks >= 0.0) & (breaks <= DEFAULT_FOLLOWUP_DAYS)]
    breaks = np.unique(np.append(breaks, 0.0))
    breaks.sort()

    # ensure last exactly equals followup
    if breaks[-1] != DEFAULT_FOLLOWUP_DAYS:
        breaks = np.unique(np.append(breaks, DEFAULT_FOLLOWUP_DAYS))
        breaks.sort()

    # strict monotonicity
    breaks = np.unique(breaks)
    if len(breaks) < 3:
        breaks = np.linspace(0.0, DEFAULT_FOLLOWUP_DAYS, 10)

    return breaks


def _validate_breaks_lam(breaks: np.ndarray, lam: np.ndarray, followup_days: float) -> tuple[np.ndarray, np.ndarray]:
    breaks = np.asarray(breaks, float)
    lam = np.asarray(lam, float)

    if breaks.ndim != 1 or lam.ndim != 1:
        raise ValueError("breaks and lam must be 1D arrays")
    if len(breaks) != len(lam) + 1:
        raise ValueError("len(breaks) must equal len(lam) + 1")
    if not np.all(np.diff(breaks) > 0):
        raise ValueError("breaks must be strictly increasing")
    if breaks[0] < 0:
        raise ValueError("breaks must be nonnegative")
    if np.any(lam < 0):
        raise ValueError("lam must be nonnegative hazards")

    # enforce followup cap
    followup_days = float(followup_days)
    if followup_days <= 0:
        raise ValueError("followup_days must be positive")

    if breaks[-1] > followup_days + 1e-12:
        raise ValueError(f"breaks end at {breaks[-1]} which exceeds followup_days={followup_days}")

    if abs(breaks[-1] - followup_days) > 1e-9:
        # allow shorter breaks than followup, but then simulation caps at breaks[-1]
        pass

    return breaks, lam


def simulate_pe_times(
    breaks: np.ndarray,
    lam: np.ndarray,
    eta: np.ndarray,
    rng: np.random.Generator,
    followup_days: float = DEFAULT_FOLLOWUP_DAYS,
    return_censored_at_followup: bool = True,
) -> np.ndarray:
    """
    Simulate event times from a piecewise exponential proportional hazards model.

    Model:
      hazard for subject i in interval k:
        h_i(t) = lam[k] * exp(eta[i])   for t in [breaks[k], breaks[k+1])

    Simulation uses inverse transform:
      Draw E_i ~ Exp(1)  (via -log(U))
      Find T_i where cumulative hazard H_i(T_i) = E_i

    Parameters
    ----------
    breaks : (K+1,) array
        Interval endpoints (days). Must be increasing and <= followup_days.
    lam : (K,) array
        Baseline hazard per interval (>=0).
    eta : (n,) array
        Linear predictors for subjects.
    rng : np.random.Generator
        Random generator.
    followup_days : float
        Administrative follow-up cap in days (default 1825).
    return_censored_at_followup : bool
        If True, any subject whose simulated event time exceeds followup is set to followup_days.
        Note: This function returns *event time*; if you want a censor indicator, handle it outside.
        This option is mainly to keep times bounded in simulated cohorts.

    Returns
    -------
    T_event : (n,) array of floats
        Simulated event times (days). If return_censored_at_followup is True, values are <= followup_days.
    """
    breaks, lam = _validate_breaks_lam(breaks, lam, followup_days)
    eta = np.asarray(eta, float)
    if eta.ndim != 1:
        raise ValueError("eta must be a 1D array")
    n = eta.shape[0]

    followup_days = float(followup_days)
    t_end = min(followup_days, float(breaks[-1]))

    # Draw Exp(1) thresholds
    E = rng.exponential(scale=1.0, size=n)  # Exp(1)

    # Precompute interval widths and cumulative baseline hazard increments
    widths = np.diff(breaks)  # (K,)
    # baseline increments: lam[k] * width[k]
    base_incr = lam * widths
    base_cum = np.concatenate([[0.0], np.cumsum(base_incr)])  # (K+1,)

    T = np.empty(n, dtype=float)

    # Simulate per subject (vectorization possible but this is clear and robust)
    for i in range(n):
        s = float(np.exp(eta[i]))  # subject multiplier
        if s <= 0 or not np.isfinite(s):
            raise ValueError("exp(eta) must be positive and finite")

        # If all hazards are zero, event never occurs within modeled horizon
        if np.all(lam == 0):
            T[i] = t_end
            continue

        # scaled cumulative hazard threshold on baseline scale:
        # H_i(t) = s * H0(t). Find H0(t) = E_i / s.
        H0_target = E[i] / s

        # Find interval m where base_cum[m] <= H0_target < base_cum[m+1]
        # base_cum is increasing if any hazards >0; but can have flat segments.
        m = np.searchsorted(base_cum, H0_target, side="right") - 1
        m = int(np.clip(m, 0, len(lam) - 1))

        # If target exceeds max cumulative hazard within breaks, set to end
        if H0_target >= base_cum[-1] - 1e-15:
            T[i] = t_end
            continue

        # Now solve within interval m:
        # base_cum[m] + lam[m]*(t - breaks[m]) = H0_target
        # t = breaks[m] + (H0_target - base_cum[m]) / lam[m]
        if lam[m] <= 0:
            # If hazard is zero in this interval (flat), move forward until positive hazard interval
            mm = m
            while mm < len(lam) and lam[mm] <= 0:
                mm += 1
            if mm >= len(lam):
                T[i] = t_end
                continue
            m = mm
            if H0_target < base_cum[m] - 1e-15:
                # numeric weirdness: target before start of mm; clamp
                H0_target = base_cum[m]

        t = float(breaks[m] + (H0_target - base_cum[m]) / lam[m])

        # Clamp to [0, t_end]
        if t < 0:
            t = 0.0
        if t > t_end:
            t = t_end
        T[i] = t

    if return_censored_at_followup and t_end < followup_days:
        # If breaks end before followup_days, clamp to t_end anyway (already done).
        pass

    if return_censored_at_followup:
        T = np.minimum(T, followup_days)

    return T
