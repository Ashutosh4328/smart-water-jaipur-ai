# src/scarcity_clustering.py

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("data/raw/tanker_bookings.csv")

# -------------------------------
# FEATURE AGGREGATION (AREA LEVEL)
# -------------------------------
area_features = df.groupby("area").agg({
    "price": "mean",
    "quantity_liters": "mean",
    "delivery_time_hours": "mean"
}).reset_index()

print("\nArea Level Features:\n", area_features)

# -------------------------------
# CLUSTERING
# -------------------------------
X = area_features[["price", "quantity_liters", "delivery_time_hours"]]

kmeans = KMeans(n_clusters=3, random_state=42)
area_features["scarcity_cluster"] = kmeans.fit_predict(X)

# -------------------------------
# CLUSTER INTERPRETATION (FIXED)
# -------------------------------
cluster_summary = area_features.groupby("scarcity_cluster")[[
    "price",
    "quantity_liters",
    "delivery_time_hours"
]].mean()

print("\nCluster Summary:\n", cluster_summary)

# -------------------------------
# VISUALIZATION
# -------------------------------
plt.figure()
plt.scatter(
    area_features["price"],
    area_features["delivery_time_hours"],
    c=area_features["scarcity_cluster"]
)
plt.xlabel("Average Price")
plt.ylabel("Average Delivery Time")
plt.title("Water Scarcity Zones in Jaipur")
plt.show()

print("✅ Water scarcity clustering completed successfully")
