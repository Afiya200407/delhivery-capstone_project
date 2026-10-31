import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/modelling_table.csv")

profile = df.groupby("hub_id").agg(
    shipments=("hub_id", "size"),
    mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean"),
).reset_index()

#2
profile.round(4).to_csv("outputs/hub_profile.csv", index=False)
print(profile.round(4).head())

profile_cols = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
X_scaled = StandardScaler().fit_transform(profile[profile_cols])

km3 = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
profile["cluster"] = km3.labels_

#3
dist = euclidean_distances(X_scaled)
i, j = np.triu_indices(len(X_scaled), k=1)         # 276 unordered pairs
pair_d = dist[i, j]

sep_rows = []
for K in [2, 3, 4, 5]:
    labels = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X_scaled).labels_
    same = labels[i] == labels[j]
    ratio = pair_d[~same].mean() / pair_d[same].mean()
    sizes = sorted(np.bincount(labels), reverse=True)
    sep_rows.append({"K": K, "across_over_within": ratio,
                     "cluster_sizes": "-".join(str(s) for s in sizes)})

sep_df = pd.DataFrame(sep_rows)
sep_df.round(4).to_csv("outputs/part_d_separation.csv", index=False)
print(sep_df.round(4))

#4
cluster_rate = profile.groupby("cluster")["delay_rate"].mean().sort_values()
name_map = dict(zip(cluster_rate.index, ["Standard", "Watch", "Escalate"]))
profile["segment"] = profile["cluster"].map(name_map)

hubs = pd.read_csv("data/hubs.csv")
hub_segments = profile[["hub_id", "segment"]].merge(
    hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left")
hub_segments.to_csv("outputs/hub_segments.csv", index=False)

seg_profile = profile.groupby("segment").agg(
    hubs=("hub_id", "size"),
    mean_delivery_hours=("mean_delivery_hours", "mean"),
    delay_rate=("delay_rate", "mean"),
    mean_distance_km=("mean_distance_km", "mean"),
    mean_n_stops=("mean_n_stops", "mean"),
).reset_index()
seg_profile["segment"] = pd.Categorical(seg_profile["segment"],
                                        ["Standard", "Watch", "Escalate"], ordered=True)
seg_profile = seg_profile.sort_values("segment")
seg_profile.round(4).to_csv("outputs/segment_profile.csv", index=False)
print(seg_profile.round(4))

tier_implies = {1: "Standard", 2: "Watch", 3: "Escalate"}
implied = hub_segments["hub_tier"].map(tier_implies)
mismatch = hub_segments[hub_segments["segment"] != implied]
mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)
print(mismatch)