import numpy as np
import pandas as pd

df = pd.read_csv("data/simulated_seer_crc.csv")
t = df["time"].to_numpy(float)
e = df["event"].to_numpy(int)

# event times
event_times = np.unique(t[(e == 1)])
S = 1.0
first_zero = None

for tj in event_times:
    n_risk = np.sum(t >= tj)
    d = np.sum((t == tj) & (e == 1))
    S *= (1.0 - d / n_risk)
    if S == 0.0:
        first_zero = tj
        break

print("First time KM hits 0:", first_zero)
print("Max time:", t.max())

end = 1825.0
mask_end = np.isclose(df["time"].to_numpy(float), end)
print("n time==1825:", mask_end.sum())
print("among time==1825, censored:", int(((df.loc[mask_end, "event"] == 0)).sum()))
print("among time==1825, events:", int(((df.loc[mask_end, "event"] == 1)).sum()))