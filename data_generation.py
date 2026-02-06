# src/data_generation.py

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Ensure folders exist
os.makedirs("data/raw", exist_ok=True)

# Areas in Jaipur
areas = [
    "Mansarovar",
    "Vaishali Nagar",
    "Malviya Nagar",
    "Jagatpura",
    "Jhotwara",
    "Sodala"
]

start_date = datetime(2024, 1, 1)
days = 180

tanker_data = []

for day in range(days):
    date = start_date + timedelta(days=day)

    for area in areas:
        bookings = random.randint(5, 20)

        for _ in range(bookings):
            tanker_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "area": area,
                "price": random.randint(600, 1600),
                "quantity_liters": random.choice([3000, 5000, 8000]),
                "delivery_time_hours": round(random.uniform(2, 8), 1)
            })

df_tanker = pd.DataFrame(tanker_data)
df_tanker.to_csv("data/raw/tanker_bookings.csv", index=False)

print("✅ Tanker booking data generated successfully")
