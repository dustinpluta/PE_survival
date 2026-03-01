from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter a CSV by subject IDs.")
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--ids_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--id_col", default="id")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    ids = pd.read_csv(args.ids_csv)

    if args.id_col not in df.columns:
        raise ValueError(f"'{args.id_col}' not found in {args.in_csv}")
    if args.id_col not in ids.columns:
        raise ValueError(f"'{args.id_col}' not found in {args.ids_csv}")

    keep = set(ids[args.id_col].astype(int).tolist())
    out = df[df[args.id_col].astype(int).isin(keep)].copy()

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(f"Wrote: {args.out_csv} (rows={len(out):,}, subjects={out[args.id_col].nunique():,})")


if __name__ == "__main__":
    main()