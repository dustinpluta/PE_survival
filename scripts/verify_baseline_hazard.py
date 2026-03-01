import json
import numpy as np
import pandas as pd

MODEL_JSON = "models/pe_fit/pe_seer_crc_train.json"  # <-- update

with open(MODEL_JSON, "r", encoding="utf-8") as f:
    model = json.load(f)

# ---- locate alpha vector (baseline log-hazards by interval) ----
# Adjust these keys if your model schema differs.
alpha = None

# common possibilities
for path in [
    ("params", "alpha"),
    ("baseline", "theta"),         # sometimes called theta
    ("theta",),                  # top-level
    ("alpha",),                  # top-level
]:
    cur = model
    ok = True
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            ok = False
            break
    if ok:
        alpha = np.asarray(cur, dtype=float)
        alpha_path = ".".join(path)
        break

if alpha is None:
    raise KeyError(
        "Could not find alpha/theta vector in model JSON. "
        "Tried params.alpha, params.theta, theta, alpha."
    )

fitted_lam = np.exp(alpha)

# ---- breaks / K consistency check (optional) ----
breaks = None
for path in [
    ("config", "breaks"),
    ("cfg", "breaks"),
    ("breaks",),
]:
    cur = model
    ok = True
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            ok = False
            break
    if ok:
        breaks = np.asarray(cur, dtype=float)
        breaks_path = ".".join(path)
        break

K_alpha = len(alpha)
if breaks is not None:
    K_breaks = len(breaks) - 1
    if K_breaks != K_alpha:
        print(f"[WARN] K mismatch: len({alpha_path})={K_alpha}, len({breaks_path})-1={K_breaks}")

# ---- true lam comparison if available ----
true_lam = None
for path in [
    ("config", "lam"),
    ("cfg", "lam"),
    ("lam",),
]:
    cur = model
    ok = True
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            ok = False
            break
    if ok:
        true_lam = np.asarray(cur, dtype=float)
        true_lam_path = ".".join(path)
        break

# ---- build and print comparison table ----
df = pd.DataFrame({
    "k": np.arange(K_alpha, dtype=int),
    "alpha_hat": alpha,
    "lam_hat": fitted_lam,
})

if true_lam is not None and len(true_lam) == K_alpha:
    df["lam_true"] = true_lam
    df["ratio_hat_over_true"] = df["lam_hat"] / df["lam_true"]
else:
    if true_lam is not None:
        print(f"[WARN] Found {true_lam_path} but length {len(true_lam)} != K={K_alpha}; skipping lam_true comparison.")

# show first few rows
print(f"Using baseline vector from: {alpha_path} (K={K_alpha})")
if breaks is not None:
    print(f"Using breaks from: {breaks_path} (followup_end={breaks[-1]})")
if true_lam is not None:
    print(f"Using true lam from: {true_lam_path}")

print("\nFirst 10 intervals:")
print(df.head(10).to_string(index=False))

# summary diagnostics
print("\nBaseline hazard summary (lam_hat):")
print(pd.Series(fitted_lam).describe(percentiles=[0.01, 0.1, 0.5, 0.9, 0.99]).to_string())

if "ratio_hat_over_true" in df.columns:
    r = df["ratio_hat_over_true"].to_numpy(float)
    print("\nRatio lam_hat / lam_true summary:")
    print(pd.Series(r).describe(percentiles=[0.01, 0.1, 0.5, 0.9, 0.99]).to_string())