# src/demand_prediction.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load processed data
df = pd.read_csv("data/processed/daily_demand.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"])

# Create time features
df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["weekday"] = df["date"].dt.weekday

# Encode area
df = pd.get_dummies(df, columns=["area"], drop_first=True)

# Features & target
X = df.drop(columns=["date", "demand"])
y = df["demand"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
print("📊 Mean Absolute Error:", round(mae, 2))
print("✅ Demand prediction model trained successfully")
