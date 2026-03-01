import pandas as pd
import numpy as np

BASE_CSV = "data/simulated_seer_crc.csv"              # <-- update
LONG_CSV = "data/dev/pipeline_seer_crc/long_train.csv"     # <-- update

ID_COL = "id"
TIME_COL = "time"
EVENT_COL = "event"
K_COL = "k"
Y_COL = "y"
D_COL = "d"

base = pd.read_csv(BASE_CSV)
long_df = pd.read_csv(LONG_CSV)

# ---- basic sanity ----
for c in [ID_COL, TIME_COL, EVENT_COL]:
    if c not in base.columns:
        raise ValueError(f"base_csv missing '{c}'")
for c in [ID_COL, K_COL, Y_COL, D_COL]:
    if c not in long_df.columns:
        raise ValueError(f"long_csv missing '{c}'")

base[ID_COL] = pd.to_numeric(base[ID_COL], errors="raise").astype(int)
long_df[ID_COL] = pd.to_numeric(long_df[ID_COL], errors="raise").astype(int)
long_df[K_COL] = pd.to_numeric(long_df[K_COL], errors="raise").astype(int)

base[TIME_COL] = pd.to_numeric(base[TIME_COL], errors="raise").astype(float)
base[EVENT_COL] = pd.to_numeric(base[EVENT_COL], errors="raise").astype(int)

long_df[Y_COL] = pd.to_numeric(long_df[Y_COL], errors="raise").astype(float)
long_df[D_COL] = pd.to_numeric(long_df[D_COL], errors="raise").astype(int)

print("=== Step 2: expand_long invariants + exposure diagnostics ===")
print(f"Unique ids: base={base[ID_COL].nunique():,}, long={long_df[ID_COL].nunique():,}")

# ---- event count invariant: sum_k d_ik == event_i (on overlapping IDs) ----
overlap_ids = np.intersect1d(base[ID_COL].unique(), long_df[ID_COL].unique())
base_sub = base[base[ID_COL].isin(overlap_ids)][[ID_COL, TIME_COL, EVENT_COL]].copy()

d_by_id = long_df.groupby(ID_COL, as_index=False)[D_COL].sum().rename(columns={D_COL: "d_sum"})
y_by_id = long_df.groupby(ID_COL, as_index=False)[Y_COL].sum().rename(columns={Y_COL: "y_sum"})

chk = base_sub.merge(d_by_id, on=ID_COL, how="left").merge(y_by_id, on=ID_COL, how="left")
chk["d_sum"] = chk["d_sum"].fillna(0).astype(int)
chk["y_sum"] = chk["y_sum"].fillna(0.0).astype(float)

frac_event_match = float((chk["d_sum"] == chk[EVENT_COL]).mean())
max_abs_event_diff = int((chk["d_sum"] - chk[EVENT_COL]).abs().max())

print("\n[Invariant] sum_k d_ik == event_i (on overlap ids)")
print(f"  overlap ids           : {len(overlap_ids):,}")
print(f"  fraction matching     : {frac_event_match:.6f}")
print(f"  max |d_sum - event|   : {max_abs_event_diff}")

if frac_event_match < 0.999:
    bad = chk.loc[chk["d_sum"] != chk[EVENT_COL], [ID_COL, EVENT_COL, "d_sum"]].head(10)
    print("  examples of mismatches:")
    print(bad.to_string(index=False))

# ---- exposure invariant: sum_k y_ik should equal observed time (or followup-truncated time) ----
# For the long df built from base times, y_sum should equal base time for each overlapping id.
abs_err = np.abs(chk["y_sum"] - chk[TIME_COL].to_numpy(float))
print("\n[Invariant] sum_k y_ik ~= time_i (on overlap ids)")
print(f"  max abs error         : {abs_err.max():.12g}")
print(f"  mean abs error        : {abs_err.mean():.12g}")
print(f"  fraction allclose     : {float(np.mean(abs_err < 1e-10)):.6f}")

# ---- totals ----
print("\n[Totals]")
print(f"  Base total events (overlap): {int(base_sub[EVENT_COL].sum()):,}")
print(f"  Long total events (overlap): {int(long_df[long_df[ID_COL].isin(overlap_ids)][D_COL].sum()):,}")
print(f"  Total person-time (overlap): {float(long_df[long_df[ID_COL].isin(overlap_ids)][Y_COL].sum()):.6g}")

# ---- exposure by interval ----
print("\n[Exposure by interval] sum y by k")
y_by_k = long_df.groupby(K_COL)[Y_COL].sum().sort_index()
print(y_by_k.to_string())

print("\n[Events by interval] sum d by k")
d_by_k = long_df.groupby(K_COL)[D_COL].sum().sort_index()
print(d_by_k.to_string())

# ---- who reaches late follow-up? ----
last_k = int(long_df[K_COL].max())
n_in_last = int(long_df.loc[long_df[K_COL] == last_k, ID_COL].nunique())
print("\n[Late follow-up coverage]")
print(f"  last k                : {last_k}")
print(f"  unique ids in last k  : {n_in_last:,}")