import pandas as pd
from utils.pred_grid import build_pred_long_from_long

BASE_CSV = "data/simulated_seer_crc.csv"
LONG_TRAIN_CSV = "data/dev/pipeline_seer_crc/long_train.csv"

base = pd.read_csv(BASE_CSV)
long_train = pd.read_csv(LONG_TRAIN_CSV)

# restrict base to training ids
train_ids = long_train["id"].unique()
base_train = base[base["id"].isin(train_ids)].copy()

pred_long_365, breaks = build_pred_long_from_long(
    base_train,
    long_train,
    t0=365.0
)

print("Breaks length:", len(breaks))
print("K intervals:", len(breaks) - 1)
print("Prediction long shape:", pred_long_365.shape)
print("Unique ids in pred grid:", pred_long_365["id"].nunique())
print("Min/Max y:", pred_long_365["y"].min(), pred_long_365["y"].max())