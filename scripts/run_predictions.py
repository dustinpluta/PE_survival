# scripts/run_predictions.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pe.predict import predict_long, predict_subject_cumhaz


def main():

    ap = argparse.ArgumentParser(
        description="Run PE model predictions from saved JSON model"
    )

    ap.add_argument("--model", required=True, help="Path to saved model JSON")
    ap.add_argument("--long_csv", required=True, help="Long-format CSV to predict on")
    ap.add_argument("--out_prefix", required=True, help="Output file prefix")

    ap.add_argument(
        "--id_col",
        default="__row_id__",
        help="Subject id column for cumulative hazard aggregation",
    )

    ap.add_argument(
        "--skip_cumhaz",
        action="store_true",
        help="If set, do not compute subject-level cumulative hazard",
    )

    args = ap.parse_args()

    model_path = Path(args.model)
    long_path = Path(args.long_csv)
    out_prefix = Path(args.out_prefix)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not long_path.exists():
        raise FileNotFoundError(f"Long CSV not found: {long_path}")

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    with open(model_path, "r", encoding="utf-8") as f:
        model = json.load(f)

    # Load data
    long_df = pd.read_csv(long_path)

    # Row-level predictions
    pred_df, _ = predict_long(long_df, model, return_X=False)

    pred_out = long_df.copy()
    pred_out = pd.concat([pred_out, pred_df], axis=1)

    pred_path = out_prefix.with_suffix(".predictions.csv")
    pred_out.to_csv(pred_path, index=False)

    print(f"Saved row-level predictions to {pred_path}")

    # Subject-level cumulative hazard
    if not args.skip_cumhaz:
        if args.id_col not in long_df.columns:
            raise ValueError(
                f"id_col '{args.id_col}' not found in long data. "
                f"Available columns: {list(long_df.columns)}"
            )

        cumhaz_df = predict_subject_cumhaz(
            long_df,
            model,
            id_col=args.id_col,
        )

        cumhaz_path = out_prefix.with_suffix(".cumhaz.csv")
        cumhaz_df.to_csv(cumhaz_path, index=False)

        print(f"Saved subject cumulative hazard to {cumhaz_path}")

    print("Prediction run complete.")


if __name__ == "__main__":
    main()
