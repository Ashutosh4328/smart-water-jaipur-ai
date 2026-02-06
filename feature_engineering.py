# src/feature_engineering.py

import pandas as pd
import os

os.makedirs("data/processed", exist_ok=True)

# Load raw data
df = pd.read_csv("data/raw/tanker_bookings.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"])

# Feature engineering
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["weekday"] = df["date"].dt.weekday

# Demand = number of bookings per day per area
daily_demand = (
    df.groupby(["date", "area"])
    .size()
    .reset_index(name="demand")
)

daily_demand.to_csv("data/processed/daily_demand.csv", index=False)

print("✅ Feature engineering completed")
