# src/data_cleaning_eda.py

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# LOAD DATA
# ---------------------------
df = pd.read_csv("data/raw/tanker_bookings.csv")

print("Initial Data Shape:", df.shape)
print(df.head())

# ---------------------------
# DATA TYPE CHECK
# ---------------------------
df["date"] = pd.to_datetime(df["date"])

# ---------------------------
# MISSING VALUES
# ---------------------------
print("\nMissing Values:\n", df.isnull().sum())

# ---------------------------
# BASIC STATISTICS
# ---------------------------
print("\nStatistical Summary:\n", df.describe())

# ---------------------------
# OUTLIER CHECK (PRICE)
# ---------------------------
plt.figure()
plt.boxplot(df["price"])
plt.title("Tanker Price Distribution")
plt.ylabel("Price (INR)")
plt.show()

# ---------------------------
# DAILY DEMAND ANALYSIS
# ---------------------------
daily_demand = df.groupby("date").size()

plt.figure()
daily_demand.plot()
plt.title("Daily Water Tanker Demand")
plt.xlabel("Date")
plt.ylabel("Number of Bookings")
plt.show()

print("EDA completed successfully")
