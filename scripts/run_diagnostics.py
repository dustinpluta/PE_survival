from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pe.diagnostics import (
    DiagnosticsConfig,
    compute_row_predictions,
    pearson_residuals,
    deviance_residuals,
    observed_expected_by_interval,
    cox_snell_residuals,
    plot_pearson_hist,
    plot_cox_snell,
    plot_obs_exp,
    plot_ratio_by_interval,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to fitted model JSON.")
    ap.add_argument("--long_csv", required=True, help="Observed long CSV (train/test).")
    ap.add_argument("--out_dir", required=True, help="Output directory for diagnostics tables/plots.")

    ap.add_argument("--id_col", default="id")
    ap.add_argument("--k_col", default="k")
    ap.add_argument("--y_col", default="y")
    ap.add_argument("--d_col", default="d")
    ap.add_argument("--group_col", default=None, help="Optional grouping column (e.g. stage) for stratified obs/exp.")

    ap.add_argument("--max_rows", type=int, default=None, help="Optional row cap for faster runs/debugging.")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    long_df = pd.read_csv(args.long_csv)

    if args.max_rows is not None and args.max_rows > 0:
        long_df = long_df.iloc[: args.max_rows].copy()

    cfg = DiagnosticsConfig(
        id_col=args.id_col,
        k_col=args.k_col,
        y_col=args.y_col,
        d_col=args.d_col,
        group_col=args.group_col,
    )

    # ---- per-row predictions ----
    pred = compute_row_predictions(long_df, model)

    # ---- residuals ----
    pear = pearson_residuals(long_df, pred, cfg=cfg)
    dev = deviance_residuals(long_df, pred, cfg=cfg)

    resid_df = pd.DataFrame(
        {
            "pearson": pear.astype(float),
            "deviance": dev.astype(float),
        }
    )
    resid_csv = out_dir / "residuals_summary.csv"
    resid_df.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_csv(resid_csv)
    print(f"Wrote: {resid_csv}")

    plot_pearson_hist(pear, str(out_dir / "pearson_residuals_hist.png"), title="Pearson residuals (Poisson PE long data)")
    print(f"Wrote: {out_dir / 'pearson_residuals_hist.png'}")

    # ---- observed vs expected by interval ----
    tab = observed_expected_by_interval(long_df, pred, cfg=cfg)
    tab_csv = out_dir / ("obs_exp_by_interval.csv" if cfg.group_col is None else f"obs_exp_by_interval_and_{cfg.group_col}.csv")
    tab.to_csv(tab_csv, index=False)
    print(f"Wrote: {tab_csv}")

    if cfg.group_col is None:
        plot_obs_exp(tab, str(out_dir / "events_obs_vs_exp_by_interval.png"), cfg=cfg, title="Observed vs Expected events by interval")
        print(f"Wrote: {out_dir / 'events_obs_vs_exp_by_interval.png'}")
    plot_ratio_by_interval(tab, str(out_dir / ("ratio_obs_over_exp_by_interval.png" if cfg.group_col is None else f"ratio_obs_over_exp_by_interval_and_{cfg.group_col}.png")),
                           cfg=cfg,
                           title=("Observed / Expected by interval" if cfg.group_col is None else f"Observed / Expected by interval and {cfg.group_col}"))
    print("Wrote:", out_dir / ("ratio_obs_over_exp_by_interval.png" if cfg.group_col is None else f"ratio_obs_over_exp_by_interval_and_{cfg.group_col}.png"))

    # ---- Cox–Snell residual check ----
    cs = cox_snell_residuals(long_df, pred, cfg=cfg)
    cs_csv = out_dir / "cox_snell_residuals.csv"
    cs.to_csv(cs_csv, index=False)
    print(f"Wrote: {cs_csv}")

    plot_cox_snell(cs, str(out_dir / "cox_snell.png"), title="Cox–Snell residual check: -log(KM) vs residual")
    print(f"Wrote: {out_dir / 'cox_snell.png'}")

    # brief numeric summaries to stdout
    print("\nDiagnostics summary")
    print("  Pearson resid: mean =", float(np.mean(pear)), " sd =", float(np.std(pear)))
    print("  Deviance resid: mean =", float(np.mean(dev)), " sd =", float(np.std(dev)))
    print("  Cox–Snell r: mean =", float(cs['r'].mean()), " sd =", float(cs['r'].std()))

    print("\nDiagnostics complete.")


if __name__ == "__main__":
    main()