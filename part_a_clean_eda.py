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
