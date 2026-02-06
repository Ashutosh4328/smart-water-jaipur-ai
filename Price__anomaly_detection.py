# src/price_anomaly_detection.py

import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("data/raw/tanker_bookings.csv")

# -------------------------------
# FEATURE SELECTION
# -------------------------------
X = df[["price", "delivery_time_hours"]]

# -------------------------------
# ISOLATION FOREST MODEL
# -------------------------------
model = IsolationForest(
    contamination=0.05,   # 5% anomalies
    random_state=42
)

df["price_anomaly"] = model.fit_predict(X)

# -1 = anomaly, 1 = normal
df["price_anomaly"] = df["price_anomaly"].map({1: 0, -1: 1})

# -------------------------------
# ANOMALY ANALYSIS
# -------------------------------
anomalies = df[df["price_anomaly"] == 1]

print("🚨 Total Price Exploitation Cases Detected:", len(anomalies))
print(anomalies.head())

# -------------------------------
# SAVE RESULTS
# -------------------------------
anomalies.to_csv("data/processed/price_exploitation_cases.csv", index=False)

# -------------------------------
# VISUALIZATION
# -------------------------------
plt.figure()
plt.scatter(
    df["delivery_time_hours"],
    df["price"],
    c=df["price_anomaly"]
)
plt.xlabel("Delivery Time (Hours)")
plt.ylabel("Price (INR)")
plt.title("Price Exploitation Detection")
plt.show()

print("✅ Price anomaly detection completed successfully")
