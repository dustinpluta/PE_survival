from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


REQUIRED_COLS = [
    "time", "event",
    "age", "cci", "tumor_size_mm", "ses",
    "sex", "race", "stage", "grade",
    "surgery", "chemo", "radiation",
    "dx_year",
]

CATEGORICAL_COLS = ["sex", "race", "stage", "grade"]
BINARY_COLS = ["event", "surgery", "chemo", "radiation"]
NUMERIC_COLS = ["time", "age", "cci", "tumor_size_mm", "ses"]
FOLLOWUP_DAYS = 1825.0


def check(condition: bool, msg: str) -> None:
    if not condition:
        raise RuntimeError(f"[FAIL] {msg}")
    print(f"[PASS] {msg}")


def main(path: str) -> None:
    p = Path(path)
    df = pd.read_csv(p)
    print("Loaded:", p.resolve())
    print("Shape:", df.shape)

    # --- schema ---
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    check(len(missing) == 0, f"required columns present (missing={missing})")

    # --- types / coding ---
    for c in BINARY_COLS:
        vals = set(pd.unique(df[c].dropna()))
        check(vals.issubset({0, 1}), f"{c} is coded 0/1 (vals={sorted(vals)})")

    check((df["time"] >= 0).all(), "time is nonnegative")
    check((df["time"] <= FOLLOWUP_DAYS + 1e-9).all(), f"time <= {FOLLOWUP_DAYS} days")

    for c in NUMERIC_COLS:
        check(np.isfinite(df[c]).all(), f"{c} has finite values")
    check((df["tumor_size_mm"] > 0).all(), "tumor_size_mm > 0")

    # --- censoring ---
    censor_frac = float(1.0 - df["event"].mean())
    print("Censor fraction:", censor_frac)
    check(0.20 <= censor_frac <= 0.40, "censor fraction is near target (~0.30)")

    # --- quick summaries ---
    print("\nTime summary:")
    print(df["time"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]))

    print("\nCategory counts:")
    for c in CATEGORICAL_COLS:
        print(f"\n{c}:")
        print(df[c].value_counts(dropna=False))

    # --- face validity: stage should have monotone risk (usually) ---
    stage_event = df.groupby("stage")["event"].mean().sort_index()
    print("\nEvent rate by stage:")
    print(stage_event)

    # Optional: check monotonicity in a lenient way
    # (ordering I<II<III<IV expected with our simulation offsets)
    stage_order = ["I", "II", "III", "IV"]
    if all(s in stage_event.index for s in stage_order):
        rates = stage_event.loc[stage_order].to_numpy(float)
        check(np.all(np.diff(rates) >= -0.02), "event rate is roughly nondecreasing from I→IV")

    # --- duplicates / missingness ---
    miss = df.isna().mean().sort_values(ascending=False)
    print("\nMissingness (fraction):")
    print(miss.head(10))
    check(miss.max() == 0.0, "no missing values")

    print("\nAll checks passed.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    args = ap.parse_args()
    main(args.data)
