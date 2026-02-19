# scripts/run_pe_fit_minimal.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from pe.fit import PEFitMinimalConfig, fit_pe_minimal


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal PE fit from long-format CSV.")
    ap.add_argument("--long_csv", required=True, help="Long-format CSV (output of expand_long)")
    ap.add_argument("--out_json", required=True, help="Output JSON file for fitted model")
    ap.add_argument("--covariates", required=True, help="Comma-separated covariate columns")
    ap.add_argument("--categorical", default="", help="Comma-separated subset of covariates treated as categorical")
    ap.add_argument("--eps", type=float, default=1e-12, help="Floor for y in log-offset")
    ap.add_argument("--verbose", type=bool, default=True, help="Print model diagnostics")
    args = ap.parse_args()

    df = pd.read_csv(args.long_csv)
    covariates = _parse_csv_list(args.covariates)
    categorical = _parse_csv_list(args.categorical) if args.categorical.strip() else []

    cfg = PEFitMinimalConfig(eps=float(args.eps))
    fit = fit_pe_minimal(df, covariates=covariates, categorical=categorical, cfg=cfg)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fit, indent=2), encoding="utf-8")

    # also save a readable summary
    summary_path = out_path.with_suffix(".summary.txt")
    summary_path.write_text(fit["summary"], encoding="utf-8")

    print(f"Wrote: {out_path}")
    print(f"Wrote: {summary_path}")
    print(f"Rows={fit['fit_stats']['n_rows']}, Events={fit['fit_stats']['n_events']}, AIC={fit['fit_stats']['aic']:.3f}")
    print(fit['schema_version'])
    print(fit['fit_stats'])
    print(len(fit["design_info"]["exog_names"]))
    print(fit["design_info"]["interval_levels"][:5])
    print(fit["design_info"]["categorical_levels"].keys())
    print(fit["baseline"]["K"])
    print(fit["inference"]["covariate_table"])
if __name__ == "__main__":
    main()
