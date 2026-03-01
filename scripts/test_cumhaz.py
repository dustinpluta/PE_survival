import json, numpy as np, pandas as pd
from pe.predict import predict_long

long_df = pd.read_csv("C:/Users/dpluta/Documents/Github/PE_survival_v2/data/dev/pipeline_seer_crc/long_all.csv")
model = json.load(open("C:/Users/dpluta/Documents/Github/PE_survival_v2/models/pe_fit/pe_seer_crc_train.json","r"))

pred, _ = predict_long(long_df, model, return_X=False)
haz = pred["hazard"].to_numpy(float)
mu = long_df["y"].to_numpy(float) * haz

print("haz quantiles:", np.quantile(haz, [0,.5,.9,.99,.999]))
print("mu  quantiles:", np.quantile(mu,  [0,.5,.9,.99,.999]))
print("frac mu>0.1:", (mu>0.1).mean(), "frac mu>1:", (mu>1).mean())