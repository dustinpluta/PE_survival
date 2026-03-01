import json
import pandas as pd
from pe.predict import predict_long

MODEL_JSON = "models/pe_fit/pe_seer_crc_train.json"              # <-- update
LONG_CSV   = "data/dev/pipeline_seer_crc/long_train.csv"        # <-- update

model = json.load(open(MODEL_JSON, "r"))
long_df = pd.read_csv(LONG_CSV)

pred, X = predict_long(long_df, model, return_X=True)

print("type(pred):", type(pred))
if isinstance(pred, dict):
    print("pred keys:", sorted(pred.keys()))
    for k in sorted(pred.keys()):
        v = pred[k]
        # summarize shape/preview
        if hasattr(v, "shape"):
            print(f"  {k}: type={type(v)} shape={v.shape}")
        else:
            try:
                n = len(v)
                print(f"  {k}: type={type(v)} len={n}")
            except Exception:
                print(f"  {k}: type={type(v)} (no len/shape)")
    # print a small sample for likely candidates
    for cand in ["hazard", "rate", "mu", "linpred", "eta", "cumhaz"]:
        if cand in pred:
            vv = pred[cand]
            print(f"\nSample pred['{cand}']:", list(vv[:5]) if hasattr(vv, "__getitem__") else vv)
else:
    print("pred repr:", repr(pred)[:500])