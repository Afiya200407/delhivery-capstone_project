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

#5
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

#6
df_g = df.copy()
df_g["distance_band"] = pd.cut(df_g["distance_km"], bins=[0, 200, 400, 600, 2000],
                               labels=["0-200", "201-400", "401-600", "601+"])

def delay_summary(data, col):
    s = data.groupby(col, observed=True).agg(
        shipments=("is_delayed", "size"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean"),
    ).reset_index()
    s.insert(0, "dimension", col)
    return s.rename(columns={col: "group"})

tables = [delay_summary(df_g, c) for c in ["service_type", "hub_tier", "distance_band"]]
result = pd.concat(tables, ignore_index=True).round(4)
result.to_csv("outputs/part_a_delay_by_group.csv", index=False)

no_cols = df.select_dtypes("number")
correlations = no_cols.corr()
correlations.round(4).to_csv(
    "outputs/part_a_correlations.csv", 
    index=True, 
    index_label="feature"
)

#Findings 1,2,3
print(pd.read_csv('outputs/part_a_delay_by_group.csv').to_string())

#Findings 4
corr_with_hours = correlations["delivery_hours"].drop("delivery_hours")
top3 = corr_with_hours.abs().sort_values(ascending=False).head(3).index
print(corr_with_hours[top3].round(4))