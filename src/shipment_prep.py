import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no",
                 "sticker_serial_mod", "shelf_slot_no"]

def clean_shipments(raw, medians=None):
    df = raw.copy()
    df = df.drop_duplicates()                                            # i
    df['service_type'] = df['service_type'].str.strip().str.title()      # ii
    df['payment_mode'] = df['payment_mode'].fillna('Unknown')            # iii
    df['declared_value_inr'] = pd.to_numeric(                            # iv
        df['declared_value'].str.replace("₹", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce")
    df = df.drop(columns="declared_value")
    df['dispatch_time'] = pd.to_datetime(df['dispatch_time'], format="%Y-%m-%d %H:%M")   # v
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan    # vi

    cols = ["distance_km", "parcel_weight_kg", "declared_value_inr"]     # vii
    if medians is None:
        medians = {c: float(df[c].median()) for c in cols}
    for c in cols:
        df[c] = df[c].fillna(medians[c])
    return df, medians

def build_features(clean, hubs):
    df = pd.merge(clean, hubs, on='hub_id', how='left')
    df['promised_hours'] = df['service_type'].map(PROMISED_HOURS).astype(int)
    df['is_cod'] = (df['payment_mode'] == "COD").astype(int)
    df['is_weekend'] = (df['dispatch_time'].dt.dayofweek >= 5).astype(int)
    df['dispatch_hour'] = df['dispatch_time'].dt.hour
    df['log_declared_value'] = np.log(df['declared_value_inr'])
    return df