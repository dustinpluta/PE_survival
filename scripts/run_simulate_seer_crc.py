from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pe_sim.simulate_seer_crc import PEParams, simulate_seer_crc_pe_dataset


def parse_floats(s: str) -> np.ndarray:
    return np.asarray([float(x.strip()) for x in s.split(",") if x.strip()], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--target_censor_frac", type=float, default=0.30)
    ap.add_argument("--out", type=str, default="data/simulated_seer_crc.csv")

    ap.add_argument("--breaks", type=str, required=True,
                    help="Comma-separated breaks in days, len=K+1")
    ap.add_argument("--lam", type=str, required=True,
                    help="Comma-separated baseline hazards, len=K")
    ap.add_argument("--beta_num", type=str, required=True,
                    help="Comma-separated numeric betas (len=4): age_per10_centered, cci, log_tumor_size, ses")

    args = ap.parse_args()

    breaks = parse_floats(args.breaks)
    lam = parse_floats(args.lam)
    beta_num = parse_floats(args.beta_num)

    pe = PEParams(breaks=breaks, lam=lam, beta_num=beta_num)
    df = simulate_seer_crc_pe_dataset(
        n=args.n,
        pe=pe,
        target_censor_frac=args.target_censor_frac,
        seed=args.seed,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("Saved:", out.resolve())
    print("Event rate:", float(df["event"].mean()))
    print("Censor frac:", float(1.0 - df["event"].mean()))
    print("Max time:", float(df["time"].max()))


if __name__ == "__main__":
    main()
