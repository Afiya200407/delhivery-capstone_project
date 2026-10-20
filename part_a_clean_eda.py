import numpy as np
import pandas as pd
import os
from src.shipment_prep import clean_shipments, build_features, FEATURES

os.makedirs("outputs", exist_ok=True)

raw = pd.read_csv("data/raw_shipments.csv", encoding="utf-8")
hubs = pd.read_csv("data/hubs.csv", encoding="utf-8")

dedup = raw.drop_duplicates()

raw_rows = len(raw)
rows_after_dedup = len(dedup)
duplicate_rows_removed = len(raw) - len(dedup)
missing_distance_km_before = dedup["distance_km"].isna().sum()
missing_parcel_weight_before = dedup["parcel_weight_kg"].isna().sum() 
#This is the count before the sentinel step, so the 999.0 rows aren’t missing yet and aren’t counted.
sentinel_weight_rows = (dedup["parcel_weight_kg"] > 100).sum()
missing_declared_value_before = dedup["declared_value"].isna().sum()
missing_payment_mode_before = dedup["payment_mode"].isna().sum()
service_type_spellings_before = dedup["service_type"].nunique()

clean, medians = clean_shipments(raw)
df = build_features(clean, hubs)
df["is_delayed"] = (df["delivery_hours"] > df["promised_hours"]).astype(int)
df.to_csv("data/modelling_table.csv", index=False)

print(raw_rows, rows_after_dedup, duplicate_rows_removed)
print(missing_distance_km_before, missing_parcel_weight_before, sentinel_weight_rows,
    missing_declared_value_before, missing_payment_mode_before, service_type_spellings_before)
print(len(df), df["is_delayed"].sum(), df[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())


q1 = df["delivery_hours"].quantile(0.25)
q3 = df["delivery_hours"].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

checks = {
    "raw_rows": raw_rows,
    "duplicate_rows_removed": duplicate_rows_removed,
    "rows_after_dedup": rows_after_dedup,
    "missing_distance_km_before": missing_distance_km_before,
    "missing_parcel_weight_before": missing_parcel_weight_before,
    "sentinel_weight_rows": sentinel_weight_rows,
    "missing_declared_value_before": missing_declared_value_before,
    "missing_payment_mode_before": missing_payment_mode_before,
    "service_type_spellings_before": service_type_spellings_before,
    "service_type_spellings_after": df["service_type"].nunique(),       
    "declared_value_mean_inr": df["declared_value_inr"].mean(),            
    "declared_value_median_inr": df["declared_value_inr"].median(),
    "log_declared_value_mean": df["log_declared_value"].mean(),            
    "log_declared_value_median": df["log_declared_value"].median(),
    "median_distance_km": medians["distance_km"],
    "median_parcel_weight_kg": medians["parcel_weight_kg"],
    "median_declared_value_inr": medians["declared_value_inr"], 
    "rows_after_hub_merge": len(df),               
    "nulls_in_features_and_target": df[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum(),       
    "delayed_count": df["is_delayed"].sum(),
    "delay_rate": df["is_delayed"].mean(),
    "hours_q1": q1,
    "hours_q3": q3,
    "hours_lower_bound": lower,
    "hours_upper_bound": upper,
    "hours_outlier_count": ((df["delivery_hours"] < lower) | (df["delivery_hours"] > upper)).sum(),                
}

checks_df = pd.DataFrame({"metric": list(checks.keys()), "value": list(checks.values())})
checks_df["value"] = checks_df["value"].astype(float).round(4)
checks_df.to_csv("outputs/part_a_checks.csv", index=False)