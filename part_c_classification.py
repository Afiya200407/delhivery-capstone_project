import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from src.shipment_prep import FEATURES, NOISE_COLUMNS

os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/modelling_table.csv")
X = df[FEATURES]
y = df["is_delayed"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(len(X_train), len(X_val), len(X_test))
print(y_train.sum(), y_val.sum(), y_test.sum())

#2
def eval_row(name, y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

rows = []

# blind baseline
rows.append(eval_row("blind_all_on_time", y_val, np.zeros(len(y_val), dtype=int)))

# logistic regression
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipe.fit(X_train, y_train)
rows.append(eval_row("logistic_regression", y_val, pipe.predict(X_val)))

# decision tree: pick the depth with the highest validation F1 
dt_depths = [2, 3, 4, 5, 6, 8, 10]
best_row, best_d = None, None
for d in dt_depths:
    model = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42)
    model.fit(X_train, y_train)
    row = eval_row(f"decision_tree_depth_{d}", y_val, model.predict(X_val))
    if best_row is None or row["f1"] > best_row["f1"]:     # > keeps the smaller depth on ties
        best_row, best_d = row, d
rows.append(best_row)

# random forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rows.append(eval_row("random_forest_default", y_val, rf_model.predict(X_val)))

models_df = pd.DataFrame(rows)
models_df.round(4).to_csv("outputs/part_c_models.csv", index=False)
print(models_df.round(4))
print("best tree depth:", best_d)

#3
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score

# default forest on all 17 columns of the TRAINING rows
rf_importance = rf_model.feature_importances_

selector_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)
f_scores = selector_all.scores_

scores_df = pd.DataFrame({
    "feature": FEATURES,
    "f_score": f_scores,
    "rf_importance": rf_importance,
    "is_noise": [f in NOISE_COLUMNS for f in FEATURES],
})
scores_df["f_rank"] = scores_df["f_score"].rank(ascending=False, method="first").astype(int)   # 1 = highest F-score
scores_df = scores_df[["feature", "f_score", "f_rank", "rf_importance", "is_noise"]].sort_values("f_rank")
scores_df.round(4).to_csv("outputs/part_c_feature_scores.csv", index=False)
print(scores_df.round(4))
print("noise share of importance:", scores_df.loc[scores_df["is_noise"], "rf_importance"].sum())

k_rows = []
for k in [5, 7, 10, 12, 17]:
    pipe_k = Pipeline([
        ("selector", SelectKBest(f_classif, k=k)),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    pipe_k.fit(X_train, y_train)
    k_rows.append({"k": k, "val_f1": f1_score(y_val, pipe_k.predict(X_val))})

k_df = pd.DataFrame(k_rows)
k_df.round(4).to_csv("outputs/part_c_selectk.csv", index=False)
print(k_df.round(4))

#4
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

sel7 = SelectKBest(f_classif, k=7).fit(X_train, y_train)
X_train_7 = sel7.transform(X_train)
X_val_7 = sel7.transform(X_val)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10],
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_train_7, y_train)

rand = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid,
                          n_iter=9, cv=5, random_state=42)
rand.fit(X_train_7, y_train)

default_rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_7, y_train)

tuning_rows = [
    {"search": "grid_search", "candidates": 27, "cv_folds": 5, "total_fits": 27 * 5,
     "best_params": str(grid.best_params_), "best_cv_accuracy": grid.best_score_,
     "val_f1": f1_score(y_val, grid.best_estimator_.predict(X_val_7))},

     {"search": "randomized_search", "candidates": 9, "cv_folds": 5, "total_fits": 9 * 5,
     "best_params": str(rand.best_params_), "best_cv_accuracy": rand.best_score_,
     "val_f1": f1_score(y_val, rand.best_estimator_.predict(X_val_7))},

    {"search": "default_forest_same_7_features", "candidates": 1, "cv_folds": 0, "total_fits": 1,
     "best_params": "n_estimators=100, max_depth=None, min_samples_leaf=1",
     "best_cv_accuracy": None,
     "val_f1": f1_score(y_val, default_rf.predict(X_val_7))},
]
tuning_df = pd.DataFrame(tuning_rows)
tuning_df.round(4).to_csv("outputs/part_c_tuning.csv", index=False)
print(tuning_df.round(4))

#5
candidates = {
    "default_forest_same_7_features": dict(n_estimators=100, max_depth=None, min_samples_leaf=1),
    "grid_search": grid.best_params_,
    "randomized_search": rand.best_params_,
}
priority = ["default_forest_same_7_features", "grid_search", "randomized_search"]   # tie-break order

best_f1 = tuning_df["val_f1"].max()
chosen_from = next(name for name in priority
                   if np.isclose(tuning_df.loc[tuning_df["search"] == name, "val_f1"].iloc[0], best_f1))
chosen_params = candidates[chosen_from]
print("chosen:", chosen_from, chosen_params)

final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(random_state=42, **chosen_params)),
])
final_model.fit(X_train, y_train)

#6
proba_val = final_model.predict_proba(X_val)[:, 1]      # probability of 'delayed'
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

thr_rows = []
for t in thresholds:
    pred = (proba_val >= t).astype(int)                  # delayed when probability >= t
    row = eval_row("x", y_val, pred)                      # reuse helper
    row["threshold"] = t
    del row["model"]
    thr_rows.append(row)

thr_df = pd.DataFrame(thr_rows)[["threshold", "TN", "FP", "FN", "TP",
                                 "accuracy", "precision", "recall", "f1"]]
thr_df.round(4).to_csv("outputs/part_c_thresholds.csv", index=False)
print(thr_df.round(4))

ok = thr_df[thr_df["recall"] >= 0.80]
chosen_threshold = ok["threshold"].max() if len(ok) > 0 else 0.1
print("chosen threshold:", chosen_threshold)

#7
proba_test = final_model.predict_proba(X_test)[:, 1]
pred_test = (proba_test >= chosen_threshold).astype(int)
test_row = eval_row("x", y_test, pred_test)
test_row["threshold"] = chosen_threshold
del test_row["model"]
final_test = pd.DataFrame([test_row])[["threshold", "TN", "FP", "FN", "TP",
                                       "accuracy", "precision", "recall", "f1"]]
final_test.round(4).to_csv("outputs/part_c_final_test.csv", index=False)
print(final_test.round(4))

config = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": chosen_params["n_estimators"],
    "max_depth": "unlimited" if chosen_params["max_depth"] is None else chosen_params["max_depth"],
    "min_samples_leaf": chosen_params["min_samples_leaf"],
    "threshold": chosen_threshold,
    "chosen_from": chosen_from,
}])
config.to_csv("outputs/part_c_final_config.csv", index=False)