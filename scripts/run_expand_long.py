# scripts/run_expand_long.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from utils.expand_long import ExpandLongConfig, expand_to_long


def parse_breaks(s: Optional[str], followup: float) -> np.ndarray:
    """
    Parse comma-separated breaks string into a sorted numpy array.
    If None, use a reasonable default grid for day units with 5-year follow-up:
      - 0..365 by 30 days
      - 365..1825 by 90 days
      - include 1825
    """
    if s is None or s.strip() == "":
        br = np.unique(
            np.concatenate(
                [
                    np.arange(0.0, 365.0 + 30.0, 30.0),
                    np.arange(365.0, followup + 90.0, 90.0),
                    np.array([followup]),
                ]
            )
        ).astype(float)
        br.sort()
        return br

    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    br = np.array([float(p) for p in parts], dtype=float)
    br = np.unique(br)
    br.sort()
    return br


def main() -> None:
    ap = argparse.ArgumentParser(description="Expand subject-level survival data to PE long format.")
    ap.add_argument("--in_csv", required=True, help="Input subject-level CSV path")
    ap.add_argument("--out_csv", required=True, help="Output long-format CSV path")
    ap.add_argument("--time_col", default="time", help="Time column name (days)")
    ap.add_argument("--event_col", default="event", help="Event indicator column name (0/1)")
    ap.add_argument("--followup", type=float, default=1825.0, help="Follow-up horizon in days (default 1825)")
    ap.add_argument(
        "--breaks",
        default=None,
        help="Comma-separated breakpoints in days (e.g. '0,30,60,...,1825'). If omitted, uses a default grid.",
    )
    ap.add_argument(
        "--keep_cols",
        default=None,
        help="Comma-separated covariate columns to carry through. If omitted, keeps all non-(time,event).",
    )
    ap.add_argument("--eps", type=float, default=1e-12, help="Tolerance for zero exposure filtering")

    args = ap.parse_args()

    in_path = Path(args.in_csv)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    breaks = parse_breaks(args.breaks, followup=float(args.followup))

    keep_cols: Optional[List[str]]
    if args.keep_cols is None or args.keep_cols.strip() == "":
        keep_cols = None
    else:
        keep_cols = [c.strip() for c in args.keep_cols.split(",") if c.strip() != ""]

    cfg = ExpandLongConfig(
        time_col=args.time_col,
        event_col=args.event_col,
        id_col="__row_id__",
        keep_cols=keep_cols,
        clip_to_followup=True,
        eps=float(args.eps),
    )

    long_df = expand_to_long(df, breaks, cfg=cfg)
    long_df.to_csv(out_path, index=False)

    # quick summary
    n = len(df)
    K = len(breaks) - 1
    print(f"Wrote long data to: {out_path}")
    print(f"Input rows: {n}")
    print(f"Intervals K: {K}")
    print(f"Long rows: {len(long_df)}")
    print(f"Event rows (d=1): {int(long_df['d'].sum())}")
    print(f"Mean exposure per long row: {float(long_df['y'].mean()):.6g}")


if __name__ == "__main__":
    main()
