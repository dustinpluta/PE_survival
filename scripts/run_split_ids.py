from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Split subject IDs into train/test.")
    ap.add_argument("--csv", required=True, help="CSV containing id column (base or long)")
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_train_ids", required=True)
    ap.add_argument("--out_test_ids", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.id_col not in df.columns:
        raise ValueError(f"Missing '{args.id_col}' in {args.csv}")

    ids = np.asarray(sorted(df[args.id_col].dropna().unique().astype(int)))
    if len(ids) < 2:
        raise ValueError("Need at least 2 unique ids to split.")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(ids)

    n_train = int(round(args.train_frac * len(ids)))
    n_train = min(max(n_train, 1), len(ids) - 1)

    train_ids = perm[:n_train]
    test_ids = perm[n_train:]

    Path(args.out_train_ids).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_test_ids).parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({args.id_col: train_ids}).to_csv(args.out_train_ids, index=False)
    pd.DataFrame({args.id_col: test_ids}).to_csv(args.out_test_ids, index=False)

    print(f"Wrote: {args.out_train_ids} (n={len(train_ids)})")
    print(f"Wrote: {args.out_test_ids}  (n={len(test_ids)})")


if __name__ == "__main__":
    main()