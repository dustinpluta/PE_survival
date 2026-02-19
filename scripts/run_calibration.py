# scripts/run_calibration.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pe.calibration import (
    PECalibrationConfig,
    calibration_by_interval,
    calibration_by_interval_and_group,
    plot_group_interval_ratio,
    plot_interval_calibration,
    plot_interval_ratio,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run PE calibration from saved model JSON + long CSV")
    ap.add_argument("--model", required=True, help="Path to saved model JSON")
    ap.add_argument("--long_csv", required=True, help="Long-format CSV (expand_long output)")
    ap.add_argument("--out_dir", required=True, help="Directory to save calibration tables and plots")
    ap.add_argument("--group_col", default="", help="Optional group column (e.g., stage) for stratified calibration")

    args = ap.parse_args()

    model_path = Path(args.model)
    long_path = Path(args.long_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(model_path, "r", encoding="utf-8") as f:
        model = json.load(f)

    long_df = pd.read_csv(long_path)

    cfg = PECalibrationConfig(
        interval_col=model["config"]["interval_col"],
        event_col=model["config"]["event_col"],
        exposure_col=model["config"]["exposure_col"],
        eps=float(model["config"]["eps"]),
    )

    # ---- interval calibration ----
    cal_k = calibration_by_interval(long_df, model, cfg=cfg)
    cal_k_path = out_dir / "calibration_by_interval.csv"
    cal_k.to_csv(cal_k_path, index=False)

    plot_interval_calibration(cal_k, out_dir / "events_obs_vs_exp_by_interval.png")
    plot_interval_ratio(cal_k, out_dir / "ratio_obs_over_exp_by_interval.png")

    print(f"Wrote: {cal_k_path}")
    print(f"Wrote: {out_dir / 'events_obs_vs_exp_by_interval.png'}")
    print(f"Wrote: {out_dir / 'ratio_obs_over_exp_by_interval.png'}")

    # ---- group calibration (optional) ----
    if args.group_col.strip():
        group_col = args.group_col.strip()
        cal_kg = calibration_by_interval_and_group(long_df, model, group_col=group_col, cfg=cfg)
        cal_kg_path = out_dir / f"calibration_by_interval_and_{group_col}.csv"
        cal_kg.to_csv(cal_kg_path, index=False)

        plot_group_interval_ratio(
            cal_kg,
            group_col=group_col,
            out_path=out_dir / f"ratio_obs_over_exp_by_interval_and_{group_col}.png",
        )

        print(f"Wrote: {cal_kg_path}")
        print(f"Wrote: {out_dir / f'ratio_obs_over_exp_by_interval_and_{group_col}.png'}")

    print("Calibration run complete.")


if __name__ == "__main__":
    main()
