# scripts/run_pipeline.py
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _run(cmd: List[str]) -> None:
    print("\n>>> " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run full PE model pipeline from non-long CSV."
    )

    # Required inputs
    ap.add_argument("--input_csv", required=True, help="Non-long survival CSV")
    ap.add_argument("--long_csv", required=True, help="Output long-format CSV")
    ap.add_argument("--model_json", required=True, help="Output fitted model JSON")
    ap.add_argument("--pred_prefix", required=True, help="Prediction output prefix")
    ap.add_argument("--diag_dir", required=True, help="Diagnostics output directory")
    ap.add_argument("--cal_dir", required=True, help="Calibration output directory")

    # Fit configuration
    ap.add_argument("--covariates", required=True, help="Comma-separated covariates")
    ap.add_argument("--categorical", default="", help="Comma-separated categorical covariates")

    # Calibration configuration
    ap.add_argument("--breaks", default=None, help="Breaks used")
    ap.add_argument("--cal_group_col", default="stage", help="Group column for calibration ('' for none)")
    ap.add_argument("--strata_col", default="stage", help="Stratification column for calibration risk")
    ap.add_argument("--horizons", default=None)
    ap.add_argument("--n_bins", default=10)

    # Optional step skips
    ap.add_argument("--skip_expand", action="store_true")
    ap.add_argument("--skip_fit", action="store_true")
    ap.add_argument("--skip_predictions", action="store_true")
    ap.add_argument("--skip_diagnostics", action="store_true")
    ap.add_argument("--skip_calibration", action="store_true")
    ap.add_argument("--skip_calibration_risk", action="store_true")

    args = ap.parse_args()
    py = sys.executable

    # Create directories
    Path(args.long_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.model_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.pred_prefix).parent.mkdir(parents=True, exist_ok=True)
    Path(args.diag_dir).mkdir(parents=True, exist_ok=True)
    Path(args.cal_dir).mkdir(parents=True, exist_ok=True)

    # 1) Expand to long
    if not args.skip_expand:
        _run([
            py,
            "scripts/run_expand_long.py",
            "--in_csv",
            args.input_csv,
            "--out_csv",
            args.long_csv,
        ])

    # 2) Fit model
    if not args.skip_fit:
        _run([
            py,
            "scripts/run_pe_fit.py",
            "--long_csv",
            args.long_csv,
            "--out_json",
            args.model_json,
            "--covariates",
            args.covariates,
            "--categorical",
            args.categorical,
        ])

    # 3) Predictions
    if not args.skip_predictions:
        _run([
            py,
            "scripts/run_predictions.py",
            "--model",
            args.model_json,
            "--long_csv",
            args.long_csv,
            "--out_prefix",
            args.pred_prefix,
        ])

    # 4) Diagnostics
    if not args.skip_diagnostics:
        _run([
            py,
            "scripts/run_diagnostics.py",
            "--model",
            args.model_json,
            "--long_csv",
            args.long_csv,
            "--out_dir",
            args.diag_dir,
        ])

    # 5) Calibration
    if not args.skip_calibration:
        cmd = [
            py,
            "scripts/run_calibration.py",
            "--model",
            args.model_json,
            "--long_csv",
            args.long_csv,
            "--out_dir",
            args.cal_dir,
            "--breaks",
            args.breaks
        ]
        if args.cal_group_col.strip():
            cmd += ["--group_col", args.cal_group_col.strip()]
        _run(cmd)
    
    if not args.skip_calibration_risk:
        cmd = [
            py,
            "scripts/run_calibration_risk.py",
            "--model",
            args.model_json,
            "--long_csv",
            args.long_csv,
            "--input_csv",
            args.input_csv,
            "--out_dir",
            args.cal_dir,
            "--breaks",
            args.breaks,
            "--horizons",
            args.horizons,
            "--n_bins",
            args.n_bins,
            "--strata_col",
            args.strata_col
        ]

    print("\n=== PIPELINE COMPLETE ===")
    print(f"Input CSV:  {args.input_csv}")
    print(f"Long CSV:   {args.long_csv}")
    print(f"Model JSON: {args.model_json}")
    print(f"Diag Dir:   {args.diag_dir}")
    print(f"Cal Dir:    {args.cal_dir}")


if __name__ == "__main__":
    main()