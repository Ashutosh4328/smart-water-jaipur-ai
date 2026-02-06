# src/price_anomaly_detection.py

import pandas as pd
from sklearn.ensemble import IsolationForest

# Load data
df = pd.read_csv("data/raw/tanker_bookings.csv")

# Anomaly detection
model = IsolationForest(contamination=0.05, random_state=42)
df["price_anomaly"] = model.fit_predict(df[["price"]])

# -1 means anomaly
anomalies = df[df["price_anomaly"] == -1]

print("🚨 Price anomalies detected:", anomalies.shape[0])
print("✅ Price anomaly detection completed")
