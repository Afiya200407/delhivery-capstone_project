import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from src.shipment_prep import FEATURES, clean_shipments

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

cfg = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
max_depth = None if str(cfg["max_depth"]) == "unlimited" else int(float(cfg["max_depth"]))
threshold = float(cfg["threshold"])

df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=int(cfg["selector_k"]))),
    ("model", RandomForestClassifier(random_state=42,
                                     n_estimators=int(cfg["n_estimators"]),
                                     max_depth=max_depth,
                                     min_samples_leaf=int(cfg["min_samples_leaf"]))),
])
final_model.fit(X_train, y_train)

joblib.dump(final_model, "models/delay_model.pkl")
pd.DataFrame({"threshold": [threshold]}).to_csv("models/delay_threshold.csv", index=False)
pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)

raw = pd.read_csv("data/raw_shipments.csv", encoding="utf-8")
_, medians = clean_shipments(raw)                       # three medians from the full history
pd.DataFrame({"column": list(medians.keys()),
              "median": [round(v, 4) for v in medians.values()]}
             ).to_csv("models/impute_values.csv", index=False)

reloaded = joblib.load("models/delay_model.pkl")
p_reloaded = reloaded.predict_proba(X_test)[:, 1]
p_memory = final_model.predict_proba(X_test)[:, 1]
diff = np.abs(p_reloaded - p_memory).max()       # largest absolute difference between the two probability arrays

check = pd.DataFrame({
    "metric": ["test_rows_scored", "reload_max_abs_probability_diff", "threshold"],
    "value": [len(X_test), diff, threshold],
})
check.round(4).to_csv("outputs/part_d_reload_check.csv", index=False)
print(check)