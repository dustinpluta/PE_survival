import json
import numpy as np
import pandas as pd

from pe.predict import predict_long

MODEL_JSON = "models/pe_fit/pe_seer_crc_train.json"              # <-- update
LONG_CSV   = "data/dev/pipeline_seer_crc/long_train.csv"        # <-- update

with open(MODEL_JSON, "r", encoding="utf-8") as f:
    model = json.load(f)

long_df = pd.read_csv(LONG_CSV)

print("=== Step 3: design matrix / coefficient alignment (dict params) ===")

# ----- Pull authoritative exog_names (design columns) -----
exog_names = model.get("design_info", {}).get("exog_names", None)
if not exog_names:
    raise KeyError("Model JSON missing design_info.exog_names (needed to align params).")

print(f"n_exog_names = {len(exog_names)}")
print("first 15 exog_names:", exog_names[:15])

# ----- Pull params dict -----
params = model.get("params", None)
if not isinstance(params, dict):
    raise KeyError("Model JSON missing params dict.")

# Verify all required terms exist in params
missing_terms = [nm for nm in exog_names if nm not in params]
if missing_terms:
    raise RuntimeError(
        "Model params missing terms present in design_info.exog_names. "
        f"First few missing: {missing_terms[:10]}"
    )

beta_vec = np.asarray([float(params[nm]) for nm in exog_names], dtype=float)

print(f"beta_vec length = {len(beta_vec)}")
print(f"Intercept (params['Intercept']) = {params['Intercept']}")

# ----- Build predictions + capture X from predict_long -----
# We want X to be returned so we can inspect/align it.
pred, X = predict_long(long_df, model, return_X=True)

# X may be DataFrame (preferred) or ndarray.
if hasattr(X, "columns"):
    X_df = X.copy()
    X_cols = list(X_df.columns)
    print(f"predict_long returned X as DataFrame: shape={X_df.shape}")
    print("first 15 X columns:", X_cols[:15])

    # Compare column sets
    exog_set = set(exog_names)
    X_set = set(X_cols)

    only_in_model = sorted(list(exog_set - X_set))
    only_in_pred  = sorted(list(X_set - exog_set))

    if only_in_model:
        print("\n[FAIL] Columns required by model but missing in predict_long X.")
        print("First few missing:", only_in_model[:20])
        raise RuntimeError("predict_long X is missing required columns (see above).")

    if only_in_pred:
        # Not necessarily fatal, but indicates predict_long built extras that won't be used.
        print("\n[WARN] predict_long X has extra columns not in model design (will be ignored).")
        print("First few extras:", only_in_pred[:20])

    # Reorder and subset to match model exog_names exactly
    X_use = X_df[exog_names].to_numpy(dtype=float)

else:
    # If X is ndarray, we can only check shape (no names).
    X_use = np.asarray(X, dtype=float)
    print(f"predict_long returned X as ndarray: shape={X_use.shape}")
    if X_use.shape[1] != len(beta_vec):
        raise RuntimeError(
            f"[FAIL] X has {X_use.shape[1]} columns but model expects {len(beta_vec)}."
        )

# ----- Compute eta and hazard -----
# IMPORTANT: since your design includes the baseline via Intercept + C(k)[T.*],
# eta = X @ beta_vec already includes baseline. Do NOT add alpha[k] again.
eta = X_use @ beta_vec
hazard = np.exp(eta)

print("\neta summary:")
print(pd.Series(eta).describe(percentiles=[0.01, 0.1, 0.5, 0.9, 0.99]).to_string())

print("\nhazard summary:")
print(pd.Series(hazard).describe(percentiles=[0.01, 0.1, 0.5, 0.9, 0.99]).to_string())

# ----- Sanity: hazard should vary by k and covariates -----
k = long_df["k"].to_numpy(dtype=int)
df_tmp = pd.DataFrame({"k": k, "haz": hazard})
by_k = df_tmp.groupby("k")["haz"].mean()

print("\nmean hazard by interval k (first 10):")
print(by_k.head(10).to_string())

print("\nmean hazard by interval k (last 5):")
print(by_k.tail(5).to_string())

# ----- Optional: verify predict_long hazard equals exp(X @ beta_vec) -----
if isinstance(pred, dict) and "hazard" in pred:
    haz_pred = np.asarray(pred["hazard"], dtype=float)
    max_abs = float(np.max(np.abs(haz_pred - hazard)))
    rel = float(max_abs / (np.mean(hazard) + 1e-12))
    print("\nConsistency check with predict_long output:")
    print(f"max |haz_pred - exp(X@beta)| = {max_abs:.6g}")
    print(f"relative (divide by mean hazard) = {rel:.6g}")
else:
    print("\n[WARN] predict_long did not return pred['hazard'] as expected; skipping hazard consistency check.")