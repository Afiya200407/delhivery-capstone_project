import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]          # 17 columns
y = df["delivery_hours"]

#1
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#2
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
train_scaled_max_abs_mean = np.abs(X_train_s.mean(axis=0)).max()
test_scaled_max_abs_mean  = np.abs(X_test_s.mean(axis=0)).max()

checks = pd.DataFrame({
    "metric": ["train_rows", "val_rows", "test_rows",
               "train_scaled_max_abs_mean", "test_scaled_max_abs_mean"],
    "value": [len(X_train), len(X_val), len(X_test),
              train_scaled_max_abs_mean, test_scaled_max_abs_mean],
})
checks["value"] = checks["value"].round(4)
checks.to_csv("outputs/part_b_checks.csv", index=False)

#3
def scores(y_true, pred):
    mse = mean_squared_error(y_true, pred)
    return {"MAE": mean_absolute_error(y_true, pred), "MSE": mse,
            "RMSE": np.sqrt(mse), "R2": r2_score(y_true, pred)}

baseline_pred = np.full(len(y_val), y_train.mean())
baseline = scores(y_val, baseline_pred)
print("baseline", baseline)

#4
depths = [1, 2, 3, 4, 6, 8, 12, None]
rows = []
best_depth, best_val_rmse = None, float("inf")

for d in depths:
    tree = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
    train_rmse = scores(y_train, tree.predict(X_train))["RMSE"]   # RMSE of tree on X_train vs y_train
    val_rmse = scores(y_val, tree.predict(X_val))["RMSE"]    # RMSE of tree on X_val vs y_val
    rows.append({"max_depth": "unlimited" if d is None else d,
                 "train_rmse": train_rmse, "val_rmse": val_rmse})
    if val_rmse < best_val_rmse:      
        best_val_rmse, best_depth = val_rmse, d

depth_df = pd.DataFrame(rows).round(4)
depth_df.to_csv("outputs/part_b_tree_depth.csv", index=False)
print(depth_df)
print("best depth:", best_depth) 

#5,7
models = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()), 
        ("model", LinearRegression())]),
    "decision_tree": DecisionTreeRegressor(max_depth=best_depth, random_state=42),
    "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, 
                                                   learning_rate=0.1, 
                                                   random_state=42),
}

metric_rows = [{"model": "mean_baseline", "split": "val", **baseline}]
for name, model in models.items():
    model.fit(X_train, y_train)                       
    metric_rows.append({"model": name, "split": "val", 
                        **scores(y_val, model.predict(X_val))})

metrics_df = pd.DataFrame(metric_rows)
print(metrics_df.round(4))

#6
lr_rows = []
for lr in [0.01, 0.1, 1.0]:
    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, random_state=42).fit(X_train, y_train)
    s = scores(y_val, gb.predict(X_val))
    lr_rows.append({"learning_rate": lr, "val_rmse": s["RMSE"], "val_r2": s["R2"]})
lr_df = pd.DataFrame(lr_rows).round(4)
lr_df.to_csv("outputs/part_b_gb_learning_rate.csv", index=False)
print(lr_df)

#7
ranked = metrics_df[metrics_df["model"] != "mean_baseline"].sort_values("RMSE")
top2 = ranked["model"].head(2).tolist()
final_name = ranked["model"].iloc[0]

X_tv = pd.concat([X_train, X_val])        
y_tv = pd.concat([y_train, y_val])
kf = KFold(n_splits=5, shuffle=True, random_state=42)    # ONE object, shared by both models

cv_rows = []
for name in top2:
    s = cross_val_score(models[name], X_tv, y_tv, cv=kf)  # default scoring = R²
    cv_rows.append({"model": name, "mean_r2": s.mean(), "std_r2": s.std()})
cv_df = pd.DataFrame(cv_rows).round(4)
cv_df.to_csv("outputs/part_b_cv.csv", index=False)
print(cv_df)

test_scores = scores(y_test, models[final_name].predict(X_test))
metric_rows.append({"model": final_name, "split": "test", **test_scores})
pd.DataFrame(metric_rows).round(4).to_csv("outputs/part_b_metrics.csv", index=False)
print(test_scores)