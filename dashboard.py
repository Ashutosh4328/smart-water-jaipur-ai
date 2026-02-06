# src/dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Smart Water Jaipur Dashboard", layout="wide")

st.title("🚰 Smart Water Tanker AI Dashboard - Jaipur")
st.markdown("AI/ML system for predicting demand, detecting scarcity, and price exploitation.")

# -------------------------------
# LOAD DATA
# -------------------------------
demand = pd.read_csv("data/processed/daily_demand.csv")
scarcity = pd.read_csv("data/raw/tanker_bookings.csv")  # using raw for clustering example
price_anomalies = pd.read_csv("data/processed/price_exploitation_cases.csv")

# -------------------------------
# DEMAND PREDICTION
# -------------------------------
st.header("📊 Daily Tanker Demand")
area_filter = st.selectbox("Select Area:", demand["area"].unique())
filtered_demand = demand[demand["area"] == area_filter]

fig1 = px.line(filtered_demand, x="date", y="demand", title=f"Daily Tanker Demand - {area_filter}")
st.plotly_chart(fig1, use_container_width=True)

# -------------------------
# SCARCITY ZONES
# -------------------------
st.header("🗺️ Water Scarcity Zones")

scarcity_grouped = scarcity.groupby("area").agg({
    "quantity_liters": "sum",
    "delivery_time_hours": "mean",
    "price": "mean"
}).reset_index()

fig2 = px.scatter(
    scarcity_grouped,
    x="price",
    y="delivery_time_hours",
    size="quantity_liters",
    color="area",
    hover_name="area",
    title="Water Scarcity Zone Visualization"
)
st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# PRICE EXPLOITATION
# -------------------------------
st.header("💰 Price Exploitation Detection")
st.markdown(f"Total abnormal price cases detected: {len(price_anomalies)}")

fig3 = px.scatter(
    price_anomalies,
    x="delivery_time_hours",
    y="price",
    color="area",
    hover_data=["date", "quantity_liters"],
    title="Price Exploitation Cases"
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.markdown("Project by **Ashutosh Kumar Singh | AI/ML Intern - Kvon Tech Jaipur**")
st.markdown("Data Source: **Jaipur Municipal Corporation**")
