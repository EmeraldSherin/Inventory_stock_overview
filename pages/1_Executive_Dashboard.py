import os
import subprocess
import sys

import pandas as pd
import streamlit as st

from utils.ai_explanation import render_ai_insight

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Executive Dashboard",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PREPARED_FILE = os.path.join(
    BASE_DIR,
    "prepared_supply_chain.csv"
)

PREP_SCRIPT = os.path.join(
    BASE_DIR,
    "prepare_data.py"
)

CHART_HEIGHT = 320


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(show_spinner="Loading data...")
def load_data() -> pd.DataFrame:

    # Generate prepared file only if it does not exist
    if not os.path.exists(PREPARED_FILE):
        subprocess.run(
            [sys.executable, PREP_SCRIPT],
            cwd=BASE_DIR,
            check=True
        )

    return pd.read_csv(
        PREPARED_FILE,
        parse_dates=["Date"]
    )


# ============================================================
# KPI CARD
# ============================================================

def kpi(col, label, value):

    with col.container(border=True):
        st.metric(
            label=label,
            value=value
        )


# ============================================================
# PAGE TITLE
# ============================================================

st.title("Executive Dashboard")


# ============================================================
# LOAD DATA
# ============================================================

df = load_data()


# ============================================================
# FILTERS
# ============================================================

with st.container(border=True):

    f1, f2, f3 = st.columns([2, 2, 3])

    regions = f1.multiselect(
        "Region",
        sorted(df["Region"].unique())
    )

    warehouses = f2.multiselect(
        "Warehouse",
        sorted(df["Warehouse_ID"].unique())
    )

    min_d = df["Date"].min().date()
    max_d = df["Date"].max().date()

    date_range = f3.date_input(
        "Date range",
        (min_d, max_d),
        min_value=min_d,
        max_value=max_d
    )


# ============================================================
# APPLY FILTERS
# ============================================================

if regions:
    df = df[
        df["Region"].isin(regions)
    ]

if warehouses:
    df = df[
        df["Warehouse_ID"].isin(warehouses)
    ]

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:

    df = df[
        (df["Date"].dt.date >= date_range[0])
        &
        (df["Date"].dt.date <= date_range[1])
    ]


# ============================================================
# EMPTY DATA CHECK
# ============================================================

if df.empty:

    st.warning(
        "No data for the selected filters."
    )

    st.stop()


# ============================================================
# CURRENT SNAPSHOT
# ============================================================

latest_date = df["Date"].max()

latest = df[
    df["Date"] == latest_date
].copy()


# ============================================================
# EXECUTIVE METRICS
# ============================================================

revenue = df["Revenue"].sum()

profit = df["Gross_Profit"].sum()

units = df["Units_Sold"].sum()

margin = (
    profit / revenue * 100
    if revenue != 0
    else 0
)

current_inventory_value = (
    latest["Inventory_Value"].sum()
)

below_rop_now = int(
    latest["Below_Reorder_Point"].sum()
)

total_combinations = len(latest)

below_rop_rate = (
    latest["Below_Reorder_Point"].mean() * 100
)

# ============================================================
# AI INSIGHT
# ============================================================

_region_rev = df.groupby("Region")["Revenue"].sum().sort_values(ascending=False)
_wh_rate = (
    latest.groupby("Warehouse_ID")["Below_Reorder_Point"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

render_ai_insight("executive_dashboard", {
    "period": f"{df['Date'].min().date()} to {df['Date'].max().date()}",
    "total_revenue": revenue,
    "gross_profit": profit,
    "gross_margin_pct": margin,
    "units_sold": units,
    "current_inventory_value": current_inventory_value,
    "inventory_snapshot_date": str(latest_date.date()),
    "inventory_items": total_combinations,
    "below_reorder_count": below_rop_now,
    "below_reorder_rate_pct": below_rop_rate,
    "top_region": _region_rev.index[0],
    "top_region_revenue_share_pct": _region_rev.iloc[0] / revenue * 100 if revenue else None,
    "highest_risk_warehouse": _wh_rate.index[0] if below_rop_now > 0 else None,
    "highest_risk_warehouse_rate_pct": _wh_rate.iloc[0] if below_rop_now > 0 else None,
})

# ============================================================
# KEY METRICS
# ============================================================

st.subheader("Key Metrics")


k1, k2, k3, k4 = st.columns(4)

kpi(
    k1,
    "Total Revenue",
    f"{revenue:,.0f}"
)

kpi(
    k2,
    "Gross Profit",
    f"{profit:,.0f}"
)

kpi(
    k3,
    "Gross Margin",
    f"{margin:.1f}%"
)

kpi(
    k4,
    "Units Sold",
    f"{units:,.0f}"
)


k5, k6, k7 = st.columns(3)

kpi(
    k5,
    "Current Inventory Value",
    f"{current_inventory_value:,.0f}"
)

kpi(
    k6,
    "Below Reorder Point",
    f"{below_rop_now} / {total_combinations}"
)

kpi(
    k7,
    "Below Reorder Point Rate",
    f"{below_rop_rate:.1f}%"
)


st.caption(
    f"Inventory snapshot date: {latest_date.date()}"
)


# ============================================================
# TRENDS
# ============================================================

st.subheader("Trends")


# Monthly revenue and profit

monthly = (
    df.assign(
        Month=df["Date"]
        .dt
        .to_period("M")
        .dt
        .to_timestamp()
    )
    .groupby("Month")
    .agg(
        Revenue=("Revenue", "sum"),
        Gross_Profit=("Gross_Profit", "sum")
    )
)


# Monthly inventory value

monthly_inventory = (
    df.assign(
        Month=df["Date"]
        .dt
        .to_period("M")
        .dt
        .to_timestamp()
    )
    .groupby("Month")["Inventory_Value"]
    .sum()
)


c1, c2 = st.columns(2)


with c1.container(border=True):

    st.markdown(
        "**Monthly Revenue & Gross Profit**"
    )

    st.line_chart(
        monthly,
        height=CHART_HEIGHT
    )


with c2.container(border=True):

    st.markdown(
        "**Monthly Inventory Value**"
    )

    st.line_chart(
        monthly_inventory,
        height=CHART_HEIGHT
    )


# ============================================================
# SALES BREAKDOWN
# ============================================================

st.subheader("Sales Breakdown")


# Revenue by region

by_region = (
    df.groupby("Region")["Revenue"]
    .sum()
    .sort_values(
        ascending=False
    )
)


# Top 10 SKUs

top_skus = (
    df.groupby("SKU_ID")["Revenue"]
    .sum()
    .sort_values(
        ascending=False
    )
    .head(10)
)


c3, c4 = st.columns(2)


with c3.container(border=True):

    st.markdown(
        "**Revenue by Region**"
    )

    st.bar_chart(
        by_region,
        height=CHART_HEIGHT
    )


with c4.container(border=True):

    st.markdown(
        "**Top 10 SKUs by Revenue**"
    )

    st.bar_chart(
        top_skus,
        height=CHART_HEIGHT
    )


# ============================================================
# INVENTORY STATUS
# ============================================================

st.subheader("Inventory Status")


# Warehouse risk

warehouse_risk = (
    latest
    .groupby("Warehouse_ID")["Below_Reorder_Point"]
    .mean()
    .mul(100)
    .sort_values(
        ascending=False
    )
)


# Current inventory status

inventory_status = (
    latest["Inventory_Status"]
    .value_counts()
    .rename("SKU-Warehouse Count")
)


c5, c6 = st.columns(2)


with c5.container(border=True):

    st.markdown(
        "**Below Reorder Point Rate by Warehouse**"
    )

    st.bar_chart(
        warehouse_risk.rename(
            "Below Reorder Point %"
        ),
        height=CHART_HEIGHT
    )


with c6.container(border=True):

    st.markdown(
        "**Inventory Status - Latest Day**"
    )

    st.bar_chart(
        inventory_status,
        height=CHART_HEIGHT
    )


# ============================================================
# AT-RISK INVENTORY
# ============================================================

st.subheader(
    "SKUs Below Reorder Point"
)


at_risk = latest[
    latest["Below_Reorder_Point"]
][
    [
        "SKU_ID",
        "Warehouse_ID",
        "Inventory_Level",
        "Reorder_Point",
        "Days_of_Inventory",
        "Inventory_Status"
    ]
].sort_values(
    "Inventory_Level"
)


with st.container(border=True):

    if at_risk.empty:

        st.info(
            "No SKU-warehouse combinations are below reorder point on the latest day."
        )

    else:

        st.dataframe(
            at_risk,
            width="stretch",
            hide_index=True
        )