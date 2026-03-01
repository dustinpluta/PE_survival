import json
import numpy as np
import pandas as pd
from pe.predict import predict_long


BASE_CSV  = "data/simulated_seer_crc.csv"              # <-- update
LONG_CSV  = "data/dev/pipeline_seer_crc/long_train.csv"  # <-- update (must match cohort used)
MODEL_JSON = "models/pe_fit/pe_seer_crc_train.json"    # <-- update

ID_COL = "id"
TIME_COL = "time"
EVENT_COL = "event"

base = pd.read_csv(BASE_CSV)
long_df_obs = pd.read_csv(LONG_CSV)
model = json.load(open(MODEL_JSON, "r"))

# Cohort: restrict to IDs present in long_df_obs (train/test)
ids = np.sort(long_df_obs[ID_COL].unique())
base_sub = base[base[ID_COL].isin(ids)].copy()
print("Unique ids: base_sub =", base_sub[ID_COL].nunique(), " long =", len(ids))

# ---- derive breaks from observed long_df (robust) ----
# We assume t_left/t_right are exactly the calendar breaks used in expansion.
t_pairs = long_df_obs[["t_left", "t_right"]].drop_duplicates().sort_values(["t_left","t_right"])
# Build breaks as sorted unique endpoints
breaks = np.unique(np.r_[t_pairs["t_left"].to_numpy(float), t_pairs["t_right"].to_numpy(float)])
breaks = np.sort(breaks)
K = len(breaks) - 1
print(f"Derived K={K} intervals, followup_end={breaks[-1]}")

# Map interval index k -> [breaks[k], breaks[k+1])
# Note: your model uses C(k)[T.*], where k is the interval index in the long data.
# We'll construct k = 0..K-1 consistent with that.
interval_df = pd.DataFrame({
    "k": np.arange(K, dtype=int),
    "t_left": breaks[:-1],
    "t_right": breaks[1:],
})
interval_df["width"] = interval_df["t_right"] - interval_df["t_left"]

# ---- Build prediction grid up to horizon t0 ----
def build_pred_long(base_sub: pd.DataFrame, t0: float) -> pd.DataFrame:
    # intervals that intersect [0, t0]
    keep = interval_df[interval_df["t_left"] < t0].copy()
    keep["t_right_h"] = np.minimum(keep["t_right"].to_numpy(float), t0)
    keep["y"] = keep["t_right_h"] - keep["t_left"]
    keep = keep[keep["y"] > 0].copy()

    # Cartesian product: subjects x kept intervals
    subj = base_sub.drop(columns=[TIME_COL, EVENT_COL], errors="ignore").copy()
    subj = subj[[ID_COL] + [c for c in subj.columns if c != ID_COL]]

    out = subj.merge(keep[["k","t_left","t_right_h","y"]], how="cross")
    out = out.rename(columns={"t_right_h": "t_right"})
    # d is not used for prediction; include for schema completeness
    out["d"] = 0
    return out

def km_survival_at(t0, times, events):
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    event_times = np.unique(times[(events == 1) & (times <= t0)])
    if event_times.size == 0:
        return 1.0
    S = 1.0
    for tj in event_times:
        n_risk = np.sum(times >= tj)
        d = np.sum((times == tj) & (events == 1))
        if n_risk > 0:
            S *= (1.0 - d / n_risk)
    return float(S)

def mean_predicted_risk_at(t0: float) -> float:
    pred_long = build_pred_long(base_sub, t0)
    pred, _ = predict_long(pred_long, model, return_X=False)
    haz = pred["hazard"].to_numpy(dtype=float)
    inc = pred_long["y"].to_numpy(dtype=float) * haz
    tmp = pd.DataFrame({ID_COL: pred_long[ID_COL].to_numpy(int), "_inc": inc})
    cumhaz = tmp.groupby(ID_COL)["_inc"].sum()
    risk = 1.0 - np.exp(-cumhaz)
    return float(risk.mean())

for t0 in [365.0, 725.0, 1825.0]:
    S_km = km_survival_at(t0, base_sub[TIME_COL].to_numpy(), base_sub[EVENT_COL].to_numpy())
    km_risk = 1.0 - S_km
    mean_pred = mean_predicted_risk_at(t0)

    print(f"\nHorizon t={int(t0)}")
    print(f"KM risk        : {km_risk:.6f}")
    print(f"Mean predicted : {mean_pred:.6f}")
    print(f"Difference     : {mean_pred - km_risk:.6f}")