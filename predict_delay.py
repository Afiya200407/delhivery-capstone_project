import os
import pandas as pd
import joblib
from src.shipment_prep import clean_shipments, build_features

os.makedirs("outputs", exist_ok=True)

new_raw = pd.read_csv("data/new_shipments.csv", encoding="utf-8")
hubs = pd.read_csv("data/hubs.csv", encoding="utf-8")
model = joblib.load("models/delay_model.pkl")
threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()
imp = pd.read_csv("models/impute_values.csv")
medians = {row["column"]: float(row["median"]) for _, row in imp.iterrows()}
segments = pd.read_csv("outputs/hub_segments.csv")

clean, _ = clean_shipments(new_raw, medians)       
feats = build_features(clean, hubs)
X_new = feats[feature_order]                      

result = feats[["shipment_id", "hub_id"]].copy()
result["delay_probability"] = model.predict_proba(X_new)[:, 1].round(4)
result["flagged"] = (result["delay_probability"] >= threshold).astype(int)
result = result.merge(segments[["hub_id", "segment"]], on="hub_id", how="left")

def decide(row):
    if row["flagged"] == 1 and row["segment"] == "Escalate":
        return "Escalate to hub manager"
    if row["flagged"] == 1:
        return "Proactive customer call"
    return "No action"

result["action"] = result.apply(decide, axis=1)
out = result[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
out.to_csv("outputs/new_shipment_actions.csv", index=False)
print(out.head())
print(len(out), out["flagged"].sum(), (out["action"] == "Escalate to hub manager").sum())