# scripts/run_diagnostics.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pe.diagnostics import (
    PEDiagnosticsConfig,
    compute_diagnostics,
    plot_cox_snell,
    plot_residual_histograms,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run PE diagnostics from saved model JSON + long CSV")
    ap.add_argument("--model", required=True, help="Path to saved model JSON")
    ap.add_argument("--long_csv", required=True, help="Long-format CSV (expand_long output)")
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Directory to write diagnostics.json and plots",
    )
    ap.add_argument("--id_col", default="__row_id__", help="Subject id column in long data")
    ap.add_argument("--max_points", type=int, default=20000, help="Downsample size for hist plots")

    args = ap.parse_args()

    model_path = Path(args.model)
    long_path = Path(args.long_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not long_path.exists():
        raise FileNotFoundError(f"Long CSV not found: {long_path}")

    with open(model_path, "r", encoding="utf-8") as f:
        model = json.load(f)

    long_df = pd.read_csv(long_path)

    cfg = PEDiagnosticsConfig(id_col=args.id_col, max_points=int(args.max_points))

    # compute metrics
    diag = compute_diagnostics(long_df, model, cfg=cfg)

    diag_path = out_dir / "diagnostics.json"
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    # plots
    plot_residual_histograms(long_df, model, out_dir / "pearson_residuals_hist.png", cfg=cfg)
    plot_cox_snell(long_df, model, out_dir / "cox_snell.png", cfg=cfg)

    print(f"Wrote: {diag_path}")
    print(f"Wrote: {out_dir / 'pearson_residuals_hist.png'}")
    print(f"Wrote: {out_dir / 'cox_snell.png'}")
    print(f"Events: {diag['cox_snell']['n_events']}, mean r (events): {diag['cox_snell']['mean_r_events']:.4f}")


if __name__ == "__main__":
    main()
