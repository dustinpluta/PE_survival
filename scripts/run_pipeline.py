# scripts/run_pipeline.py
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# -------------------------
# Utilities
# -------------------------
def _as_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        return [t.strip() for t in s.split(",") if t.strip()]
    return [str(x)]


def _require(d: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d or d[key] in (None, ""):
        raise ValueError(f"Missing required key '{key}' in section '{ctx}'.")
    return d[key]


def _maybe(d: Dict[str, Any], key: str, default=None) -> Any:
    v = d.get(key, default)
    return default if v in (None, "") else v


def run_cmd(cmd: List[str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_dir(p: str | Path) -> Path:
    out = Path(p)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_config(path: str) -> Dict[str, Any]:
    # JSON only (by design: no extra deps)
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Pipeline config must be a JSON object at the top level.")
    return cfg


def py() -> str:
    # Always call the same Python that is running the pipeline
    return sys.executable


# -------------------------
# Pipeline
# -------------------------
@dataclass(frozen=True)
class ScriptPaths:
    expand_long: str = "scripts/run_expand_long.py"
    fit: str = "scripts/run_pe_fit.py"
    predictions: str = "scripts/run_predictions.py"
    diagnostics: str = "scripts/run_diagnostics.py"
    calibration: str = "scripts/run_calibration.py"
    calibration_risk: str = "scripts/run_calibration_risk.py"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run PE pipeline (expand -> fit -> predict -> diag -> calibration).")
    ap.add_argument("--config", required=True, help="Path to pipeline config JSON")
    ap.add_argument("--root", default=".", help="Project root (used to resolve relative paths)")
    ap.add_argument("--skip_expand", action="store_true", help="Skip expand_long step")
    ap.add_argument("--skip_fit", action="store_true", help="Skip fit step")
    ap.add_argument("--skip_predictions", action="store_true", help="Skip predictions step")
    ap.add_argument("--skip_diagnostics", action="store_true", help="Skip diagnostics step")
    ap.add_argument("--skip_calibration", action="store_true", help="Skip interval calibration step")
    ap.add_argument("--skip_calibration_risk", action="store_true", help="Skip risk calibration step")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    cfg = load_config(args.config)

    scripts_cfg = cfg.get("scripts", {})
    scripts = ScriptPaths(
        expand_long=str(_maybe(scripts_cfg, "expand_long", ScriptPaths.expand_long)),
        fit=str(_maybe(scripts_cfg, "fit", ScriptPaths.fit)),
        predictions=str(_maybe(scripts_cfg, "predictions", ScriptPaths.predictions)),
        diagnostics=str(_maybe(scripts_cfg, "diagnostics", ScriptPaths.diagnostics)),
        calibration=str(_maybe(scripts_cfg, "calibration", ScriptPaths.calibration)),
        calibration_risk=str(_maybe(scripts_cfg, "calibration_risk", ScriptPaths.calibration_risk)),
    )

    # ---- I/O
    io = cfg.get("io", {})
    input_csv = (root / _require(io, "input_csv", "io")).resolve()
    out_dir = ensure_dir(root / _require(io, "out_dir", "io"))

    # Where we write pipeline artifacts by convention
    long_csv = (out_dir / _maybe(io, "long_csv", "long.csv")).resolve()
    model_json = (out_dir / _maybe(io, "model_json", "model.json")).resolve()
    pred_out_prefix = (out_dir / _maybe(io, "pred_out_prefix", "pred")).resolve()
    diag_out_dir = ensure_dir(out_dir / _maybe(io, "diagnostics_dir", "diagnostics"))
    calib_out_dir = ensure_dir(out_dir / _maybe(io, "calibration_dir", "calibration"))
    calib_risk_out_dir = ensure_dir(out_dir / _maybe(io, "calibration_risk_dir", "calibration_risk"))

    # ---- Shared columns
    cols = cfg.get("columns", {})
    time_col = str(_maybe(cols, "time_col", "time"))
    event_col = str(_maybe(cols, "event_col", "event"))
    id_col = str(_maybe(cols, "id_col", "id"))
    k_col = str(_maybe(cols, "k_col", "k"))

    # ---- Breaks/horizons (used by expand/calibration)
    timegrid = cfg.get("timegrid", {})
    calib_risk = cfg.get("calibration_risk", {})
    breaks = str(_require(timegrid, "breaks", "timegrid"))  # keep as comma-string for CLIs
    horizons = str(_maybe(timegrid, "horizons", ""))        # optional; some CLIs have defaults
    n_bins = str(_maybe(calib_risk, "nbins", ""))

    # ---- Expand step
    expand = cfg.get("expand_long", {})
    if not args.skip_expand:
        cmd = [
            py(), str((root / scripts.expand_long).resolve()),
            "--in_csv", str(input_csv),
            "--out_csv", str(long_csv),
        ]
        # optional pass-through if your expand CLI supports them
        # (safe: only append if present in config)
        if "keep_cols" in expand and expand["keep_cols"] not in (None, "", []):
            cmd += ["--keep_cols", ",".join(_as_list(expand["keep_cols"]))]
        if "time_col" in expand and expand["time_col"]:
            cmd += ["--time_col", str(expand["time_col"])]
        if "event_col" in expand and expand["event_col"]:
            cmd += ["--event_col", str(expand["event_col"])]
        if "id_col" in expand and expand["id_col"]:
            cmd += ["--id_col", str(expand["id_col"])]
        if "breaks" in expand and expand["breaks"]:
            # allow override; otherwise use timegrid.breaks
            cmd += ["--breaks", str(expand["breaks"])]
        else:
            # only if your run_expand_long supports breaks; if it does not, remove these two lines
            cmd += ["--breaks", breaks]

        run_cmd(cmd)

    # ---- Fit step
    fit = cfg.get("fit", {})
    covariates = _as_list(_require(fit, "covariates", "fit"))
    categorical = _as_list(_maybe(fit, "categorical", []))

    if not args.skip_fit:
        cmd = [
            py(), str((root / scripts.fit).resolve()),
            "--long_csv", str(long_csv),
            "--out_json", str(model_json),
            "--covariates", ",".join(covariates),
        ]
        if categorical:
            cmd += ["--categorical", ",".join(categorical)]
        run_cmd(cmd)

    # ---- Predictions step (pred_grid-based)
    pred = cfg.get("predictions", {})
    if not args.skip_predictions:
        cmd = [
            py(), str((root / scripts.predictions).resolve()),
            "--model", str(model_json),
            "--long_csv", str(long_csv),
            "--out_prefix", str(pred_out_prefix),
            "--base_csv", str(input_csv)
        ]
        # horizons: optional; only pass if specified
        pred_h = _maybe(pred, "horizons", horizons)
        if pred_h:
            cmd += ["--horizons", str(pred_h)]
        # optional: group column for plots (e.g., stage)
        pred_group = _maybe(pred, "group_col", "")
        if pred_group:
            cmd += ["--group_col", str(pred_group)]
        run_cmd(cmd)

    # ---- Diagnostics step (UPDATED CALL: includes --group_col)
    diag = cfg.get("diagnostics", {})
    if not args.skip_diagnostics:
        cmd = [
            py(), str((root / scripts.diagnostics).resolve()),
            "--model", str(model_json),
            "--long_csv", str(long_csv),
            "--out_dir", str(diag_out_dir),
        ]
        diag_group = _maybe(diag, "group_col", "")
        if diag_group:
            cmd += ["--group_col", str(diag_group)]
        run_cmd(cmd)

    # ---- Interval calibration step
    calib = cfg.get("calibration", {})
    if not args.skip_calibration:
        cmd = [
            py(), str((root / scripts.calibration).resolve()),
            "--model", str(model_json),
            "--long_csv", str(long_csv),
            "--out_dir", str(calib_out_dir),
        ]
        # if your run_calibration requires breaks (or uses them for midpoints), pass them
        calib_breaks = _maybe(calib, "breaks", breaks)
        if calib_breaks:
            cmd += ["--breaks", str(calib_breaks)]
        calib_group = _maybe(calib, "group_col", "")
        if calib_group:
            cmd += ["--group_col", str(calib_group)]
        run_cmd(cmd)

    # ---- Risk calibration step (KM vs predicted at horizons)
    print(n_bins)
    calib_risk = cfg.get("calibration_risk", {})
    if not args.skip_calibration_risk:
        cmd = [
            py(), str((root / scripts.calibration_risk).resolve()),
            "--model", str(model_json),
            "--long_csv", str(long_csv),
            "--base_csv", str(input_csv),
            "--out_dir", str(calib_risk_out_dir),
            "--id_col", id_col,
            "--time_col", time_col,
            "--event_col", event_col,
            "--horizons", str(horizons),
        ]
        cr_h = _maybe(calib_risk, "horizons", horizons)
        if cr_h:
            cmd += ["--horizons", str(cr_h)]
        n_bins = _maybe(calib_risk, "n_bins", None)
        if n_bins is not None:
            cmd += ["--n_bins", str(int(n_bins))]
        group_col = _maybe(calib_risk, "group_col")
        if group_col is not None:
            cmd += ["--group_col", str(group_col)]
        run_cmd(cmd)

    print("\n=== Pipeline complete ===")
    print(f"input_csv          : {input_csv}")
    print(f"long_csv           : {long_csv}")
    print(f"model_json         : {model_json}")
    print(f"pred_out_prefix    : {pred_out_prefix}")
    print(f"diagnostics_dir    : {diag_out_dir}")
    print(f"calibration_dir    : {calib_out_dir}")
    print(f"calibration_risk   : {calib_risk_out_dir}")


if __name__ == "__main__":
    main()