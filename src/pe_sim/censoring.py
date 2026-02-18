from __future__ import annotations

import numpy as np


def tune_exponential_censor_rate(
    rng: np.random.Generator,
    T_event: np.ndarray,
    target_censor_frac: float = 0.30,
    max_iter: int = 40,
) -> float:
    """
    Choose exponential censoring rate r so that P(C < T_event) ≈ target_censor_frac.
    C ~ Exp(rate=r).
    """
    if not (0.0 < target_censor_frac < 1.0):
        raise ValueError("target_censor_frac must be in (0,1)")

    lo, hi = 1e-6, 1.0

    # expand hi until enough censoring
    for _ in range(25):
        C = rng.exponential(scale=1.0 / hi, size=len(T_event))
        frac = float(np.mean(C < T_event))
        if frac >= target_censor_frac:
            break
        hi *= 2.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        C = rng.exponential(scale=1.0 / mid, size=len(T_event))
        frac = float(np.mean(C < T_event))
        if frac < target_censor_frac:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)
