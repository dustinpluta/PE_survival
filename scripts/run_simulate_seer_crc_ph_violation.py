from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from pe_sim.seer_crc_covariates import (
    SEERCRCSimSpec,
    draw_seer_crc_covariates,
    categorical_log_hr_offset,
)


# This must match the stage effect hard-coded inside categorical_log_hr_offset()
# so we can subtract it off cleanly before injecting a time-varying stage effect.
_STAGE_EFF_IN_BASE = {"I": 0.0, "II": 0.30, "III": 0.70, "IV": 1.25}


def _parse_floats(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def _stage_lp(stage: np.ndarray, stage_eff: Dict[str, float]) -> np.ndarray:
    out = np.zeros(len(stage), dtype=float)
    for lvl, b in stage_eff.items():
        out += (stage == lvl).astype(float) * float(b)
    return out


def _simulate_piecewise_event_times_nonph_stage(
    X_num: np.ndarray,
    base_cat_no_stage: np.ndarray,
    stage: np.ndarray,
    beta_num: np.ndarray,
    stage_eff_early: Dict[str, float],
    stage_eff_late: Dict[str, float],
    breaks: np.ndarray,
    lam: np.ndarray,
    t_change: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Piecewise exponential event-time simulation with a time-varying stage effect.

    Hazard in interval k:
      h_ik = lam[k] * exp( x_i' beta_num + base_cat_no_stage_i + stage_lp_i(t) )

    where stage_lp_i(t) is:
      - stage_eff_early for t <= t_change
      - stage_eff_late  for t >  t_change

    IMPORTANT: If t_change falls inside an interval, this simulator handles it
    by treating the interval as two segments (early + late) with potentially different rates.
    """
    n = X_num.shape[0]
    K = len(lam)
    if K != len(breaks) - 1:
        raise ValueError("len(lam) must equal len(breaks)-1")

    lp_num = X_num @ beta_num
    lp_stage_early = _stage_lp(stage, stage_eff_early)
    lp_stage_late = _stage_lp(stage, stage_eff_late)

    T_event = np.empty(n, dtype=float)
    followup = float(breaks[-1])

    for i in range(n):
        t = 0.0
        hit = False

        for k in range(K):
            a = float(breaks[k])
            b = float(breaks[k + 1])
            w = b - a
            if w <= 0:
                continue

            # baseline rate in this interval
            lam_k = float(lam[k])

            # segment 1: [a, b] ∩ (-inf, t_change]
            # segment 2: [a, b] ∩ (t_change, +inf)
            # If t_change is outside [a,b], one of these has length 0.
            seg1_start = a
            seg1_end = min(b, t_change)
            seg2_start = max(a, t_change)
            seg2_end = b

            # simulate sequentially within the interval: first early segment then late segment
            # Early segment (if positive length)
            if seg1_end > seg1_start:
                rate1 = lam_k * np.exp(lp_num[i] + base_cat_no_stage[i] + lp_stage_early[i])
                dt1 = rng.exponential(1.0 / rate1) if rate1 > 0 else np.inf
                len1 = seg1_end - seg1_start

                if dt1 < len1:
                    t = seg1_start + dt1
                    T_event[i] = t
                    hit = True
                    break
                # survived segment 1
                t = seg1_end

            # Late segment (if positive length)
            if seg2_end > seg2_start and not hit:
                rate2 = lam_k * np.exp(lp_num[i] + base_cat_no_stage[i] + lp_stage_late[i])
                dt2 = rng.exponential(1.0 / rate2) if rate2 > 0 else np.inf
                len2 = seg2_end - seg2_start

                if dt2 < len2:
                    t = seg2_start + dt2
                    T_event[i] = t
                    hit = True
                    break
                # survived segment 2
                t = seg2_end

        if not hit:
            T_event[i] = followup

    return T_event


def _apply_censoring_to_target(
    T_event: np.ndarray,
    followup: float,
    target_censor_frac: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Heuristic censoring to approximately hit a target censoring fraction.
    Starts with Uniform(0, followup), then scales censor times to adjust.
    """
    n = len(T_event)
    C = rng.uniform(0.0, followup, size=n)

    def compute(C_):
        time = np.minimum(T_event, C_)
        event = (T_event <= C_).astype(int)
        return time, event

    time, event = compute(C)

    for _ in range(25):
        cens_frac = 1.0 - float(event.mean())
        if abs(cens_frac - target_censor_frac) < 0.01:
            break

        if cens_frac > target_censor_frac:
            # too much censoring -> push C later
            C = np.minimum(followup, 0.10 * followup + 1.20 * C)
        else:
            # too little censoring -> pull C earlier
            C = np.maximum(0.0, 0.85 * C)

        time, event = compute(C)

    return time, event


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Simulate SEER-like CRC survival data with a PH violation (time-varying stage effect)."
    )
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", required=True)

    ap.add_argument("--breaks", required=True, help="Comma-separated breakpoints in days (start at 0).")
    ap.add_argument("--lam", required=True, help="Comma-separated baseline hazards, length=len(breaks)-1.")
    ap.add_argument("--beta_num", required=True, help="Comma-separated numeric betas (len=4).")

    ap.add_argument("--t_change", type=float, default=365.0, help="Change point (days) for stage effect.")
    ap.add_argument(
        "--stage_beta_early",
        required=True,
        help="Early stage log-HR for II,III,IV (I reference). Example: 0.30,0.70,1.25",
    )
    ap.add_argument(
        "--stage_beta_late",
        required=True,
        help="Late stage log-HR for II,III,IV (I reference). Example: 0.10,0.20,0.30",
    )
    ap.add_argument("--target_censor_frac", type=float, default=0.30)

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    breaks = np.asarray(_parse_floats(args.breaks), dtype=float)
    lam = np.asarray(_parse_floats(args.lam), dtype=float)
    if len(lam) != len(breaks) - 1:
        raise ValueError(f"len(lam)={len(lam)} must equal len(breaks)-1={len(breaks)-1}")

    beta_num = np.asarray(_parse_floats(args.beta_num), dtype=float)
    if len(beta_num) != 4:
        raise ValueError("beta_num must have length 4 (age_per10_centered, cci, tumor_size_log, ses).")

    bE = _parse_floats(args.stage_beta_early)
    bL = _parse_floats(args.stage_beta_late)
    if len(bE) != 3 or len(bL) != 3:
        raise ValueError("stage_beta_early and stage_beta_late must each have length 3 (II,III,IV).")

    stage_eff_early = {"I": 0.0, "II": bE[0], "III": bE[1], "IV": bE[2]}
    stage_eff_late = {"I": 0.0, "II": bL[0], "III": bL[1], "IV": bL[2]}

    # Draw covariates (your existing generator)
    spec = SEERCRCSimSpec(seed=args.seed)
    df_cov, X_num = draw_seer_crc_covariates(n=args.n, rng=rng, spec=spec)

    # Base categorical LP from your DGP, then remove the built-in stage component
    base_cat = categorical_log_hr_offset(df_cov)
    stage = df_cov["stage"].to_numpy()
    stage_base = df_cov["stage"].map(_STAGE_EFF_IN_BASE).to_numpy(float)
    base_cat_no_stage = base_cat - stage_base

    # Simulate event times under non-PH stage effect
    T_event = _simulate_piecewise_event_times_nonph_stage(
        X_num=X_num,
        base_cat_no_stage=base_cat_no_stage,
        stage=stage,
        beta_num=beta_num,
        stage_eff_early=stage_eff_early,
        stage_eff_late=stage_eff_late,
        breaks=breaks,
        lam=lam,
        t_change=float(args.t_change),
        rng=rng,
    )

    followup = float(breaks[-1])
    time, event = _apply_censoring_to_target(
        T_event=T_event,
        followup=followup,
        target_censor_frac=float(args.target_censor_frac),
        rng=rng,
    )

    df_out = df_cov.copy()
    df_out["time"] = time
    df_out["event"] = event

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)

    cens_frac = 1.0 - float(df_out["event"].mean())
    print(f"Wrote {out_path} (n={len(df_out)}, censor_frac={cens_frac:.3f})")
    print(f"t_change={float(args.t_change):.1f} days; stage early={stage_eff_early}; stage late={stage_eff_late}")


if __name__ == "__main__":
    main()
