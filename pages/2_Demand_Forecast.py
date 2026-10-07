import os
import subprocess
import sys

import numpy as np
import pandas as pd
import streamlit as st

from utils.ai_explanation import render_ai_insight

st.set_page_config(page_title="Demand & Forecast", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPARED_FILE = os.path.join(BASE_DIR, "prepared_supply_chain.csv")
PREP_SCRIPT = os.path.join(BASE_DIR, "prepare_data.py")

CHART_HEIGHT = 320


@st.cache_data(show_spinner="Loading data...")
def load_data() -> pd.DataFrame:
    # Generate the prepared file with the existing script if it is missing.
    if not os.path.exists(PREPARED_FILE):
        subprocess.run([sys.executable, PREP_SCRIPT], cwd=BASE_DIR, check=True)
    return pd.read_csv(PREPARED_FILE, parse_dates=["Date"])


def kpi(col, label, value, delta=None):
    with col.container(border=True):
        st.metric(label, value, delta)


st.title("Demand & Forecast")

df = load_data()

# ---------------- Filters ----------------
with st.container(border=True):
    f1, f2, f3 = st.columns([2, 2, 3])
    skus = f1.multiselect("SKU", sorted(df["SKU_ID"].unique()))
    warehouses = f2.multiselect("Warehouse", sorted(df["Warehouse_ID"].unique()))
    min_d, max_d = df["Date"].min().date(), df["Date"].max().date()
    date_range = f3.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)

if skus:
    df = df[df["SKU_ID"].isin(skus)]
if warehouses:
    df = df[df["Warehouse_ID"].isin(warehouses)]
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    df = df[(df["Date"].dt.date >= date_range[0]) & (df["Date"].dt.date <= date_range[1])]

if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ---------------- Calculations ----------------
# All demand figures are per SKU-Warehouse-day (average across the selection).
avg_demand = df["Units_Sold"].mean()
peak_demand = df["Units_Sold"].max()
demand_cv = df["Units_Sold"].std() / avg_demand if avg_demand > 0 else np.nan

err = df["Units_Sold"] - df["Demand_Forecast"]
avg_forecast = df["Demand_Forecast"].mean()
mae = err.abs().mean()
rmse = np.sqrt((err ** 2).mean())
nonzero = df["Units_Sold"] > 0  # MAPE is undefined where actual demand is 0
mape = (err[nonzero].abs() / df.loc[nonzero, "Units_Sold"]).mean() * 100 if nonzero.any() else np.nan

daily = df.groupby("Date")[["Units_Sold", "Demand_Forecast"]].mean()

# Trend: last 30 days vs the 30 days before, within the selected range
trend_pct = np.nan
if len(daily) >= 60:
    recent = daily["Units_Sold"].iloc[-30:].mean()
    prior = daily["Units_Sold"].iloc[-60:-30].mean()
    if prior > 0:
        trend_pct = (recent - prior) / prior * 100


# ---------------- AI Insight ----------------
_promo_avg = df.groupby("Promotion_Flag")["Units_Sold"].mean()
render_ai_insight("demand_forecast", {
    "period": f"{df['Date'].min().date()} to {df['Date'].max().date()}",
    "average_demand": avg_demand,
    "peak_demand": peak_demand,
    "demand_variability_cv": demand_cv,
    "average_forecast_demand": avg_forecast,
    "trend_pct": trend_pct,
    "avg_demand_promotion": _promo_avg.get(1),
    "avg_demand_no_promotion": _promo_avg.get(0),
    "note": "Demand figures are units per SKU-Warehouse per day. Forecast is the dataset's existing baseline forecast.",
})

# ---------------- Demand KPIs ----------------
st.subheader("Demand KPIs")
k1, k2, k3, k4 = st.columns(4)
kpi(k1, "Average Demand (units/day)", f"{avg_demand:,.2f}")
kpi(k2, "Peak Demand (units/day)", f"{peak_demand:,.0f}")
kpi(k3, "Demand Variability (CV)", f"{demand_cv:.2f}" if pd.notna(demand_cv) else "N/A")
if pd.notna(trend_pct):
    direction = "Rising" if trend_pct > 0 else "Falling" if trend_pct < 0 else "Flat"
    kpi(k4, "Demand Trend (30D vs prior 30D)", direction, f"{trend_pct:+.1f}%")
else:
    kpi(k4, "Demand Trend (30D vs prior 30D)", "N/A")

# ---------------- Historical vs Forecast ----------------
st.subheader("Historical vs Forecast")
with st.container(border=True):
    st.markdown("**Actual Units Sold vs Baseline Demand Forecast (avg units/day)**")
    st.line_chart(
        daily.rename(columns={"Units_Sold": "Actual (Units_Sold)", "Demand_Forecast": "Baseline Forecast"}),
        height=CHART_HEIGHT,
    )

# ---------------- Demand Trend ----------------
st.subheader("Demand Trend")
trend = daily[["Units_Sold"]].copy()
trend["7-Day Avg"] = trend["Units_Sold"].rolling(7, min_periods=1).mean()
trend["30-Day Avg"] = trend["Units_Sold"].rolling(30, min_periods=1).mean()
with st.container(border=True):
    st.markdown("**Demand Over Time (avg units/day)**")
    st.line_chart(trend.rename(columns={"Units_Sold": "Daily"}), height=CHART_HEIGHT)

# ---------------- Promotion vs Demand ----------------
st.subheader("Promotion vs Demand")
promo = (
    df.groupby("Promotion_Flag")["Units_Sold"]
    .agg(avg_demand="mean", rows="count")
    .rename(index={0: "No Promotion", 1: "Promotion"})
)

p1, p2 = st.columns([3, 2])
with p1.container(border=True):
    st.markdown("**Average Demand: Promotion vs No Promotion (units/day)**")
    st.bar_chart(promo["avg_demand"], height=CHART_HEIGHT)
with p2.container(border=True):
    st.markdown("**Summary**")
    table = promo.rename(columns={"avg_demand": "Avg Demand", "rows": "Observations"}).round({"Avg Demand": 2})
    st.dataframe(table, width="stretch")
    if {"Promotion", "No Promotion"}.issubset(promo.index) and promo.loc["No Promotion", "avg_demand"] > 0:
        lift = promo.loc["Promotion", "avg_demand"] / promo.loc["No Promotion", "avg_demand"] - 1
        st.metric("Demand Lift on Promotion", f"{lift * 100:+.1f}%")