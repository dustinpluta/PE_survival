import numpy as np
import pandas as pd
import json
from pe.predict import predict_long


def km_survival_at(t0, times, events):
    """
    Compute Kaplan–Meier survival at time t0.
    times: observed follow-up times
    events: event indicator (1=event, 0=censored)
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)

    # event times up to t0
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

def predicted_risk_at_horizon(long_df, model, t0, id_col="id"):
    """
    Computes predicted risk_i(t0) for each subject.
    """

    pred, _ = predict_long(long_df, model, return_X=False)
    hazard = pred["hazard"].to_numpy(float)

    t_left = long_df["t_left"].to_numpy(float)
    t_right = long_df["t_right"].to_numpy(float)

    # horizon exposure
    y_h = np.maximum(0.0, np.minimum(t0, t_right) - t_left)

    inc = y_h * hazard

    tmp = pd.DataFrame({
        id_col: long_df[id_col].to_numpy(int),
        "_inc": inc
    })

    cumhaz = tmp.groupby(id_col)["_inc"].sum()

    risk = 1.0 - np.exp(-cumhaz)

    return risk

# load data
base_df = pd.read_csv("C:/Users/dpluta/Documents/Github/PE_survival_v2/data/simulated_seer_crc.csv")
long_df = pd.read_csv("C:/Users/dpluta/Documents/Github/PE_survival_v2/data/dev/pipeline_seer_crc/long_all.csv")
model = json.load(open("C:/Users/dpluta/Documents/Github/PE_survival_v2/models/pe_fit/pe_seer_crc_train.json","r"))



horizons = [365, 725, 1825]

for t0 in horizons:

    # KM risk
    S_km = km_survival_at(
        t0,
        base_df["time"].to_numpy(),
        base_df["event"].to_numpy()
    )
    km_risk = 1.0 - S_km

    # predicted risk
    pred_risk = predicted_risk_at_horizon(long_df, model, t0)
    mean_pred = pred_risk.mean()

    print(f"\nHorizon t={t0}")
    print(f"KM risk        : {km_risk:.6f}")
    print(f"Mean predicted : {mean_pred:.6f}")
    print(f"Difference     : {mean_pred - km_risk:.6f}")