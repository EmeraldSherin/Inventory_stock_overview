import os
import subprocess
import sys

import numpy as np
import pandas as pd
import streamlit as st

from utils.ai_explanation import render_ai_insight

st.set_page_config(page_title="Replenishment & Optimization", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPARED_FILE = os.path.join(BASE_DIR, "prepared_supply_chain.csv")
PREP_SCRIPT = os.path.join(BASE_DIR, "prepare_data.py")

CHART_HEIGHT = 320
N_TOP = 10
PRIORITY_ORDER = ["High", "Medium", "Low", "No Replenishment"]
REQUIRED_COLS = [
    "Inventory_Level", "Reorder_Point", "Demand_Forecast",
    "Supplier_Lead_Time_Days", "Unit_Cost", "Inventory_Status",
]


@st.cache_data(show_spinner="Loading data...")
def load_data() -> pd.DataFrame:
    # Generate the prepared file with the existing script if it is missing.
    if not os.path.exists(PREPARED_FILE):
        subprocess.run([sys.executable, PREP_SCRIPT], cwd=BASE_DIR, check=True)
    return pd.read_csv(PREPARED_FILE, parse_dates=["Date"])


def kpi(col, label, value, note=None, help=None):
    with col.container(border=True):
        st.metric(label, value, help=help)
        if note:
            st.caption(note)


st.title("Replenishment & Optimization")

df = load_data()

# ---------------- 1. Filters ----------------
with st.container(border=True):
    f1, f2, f3, f4, f5 = st.columns([2, 2, 2, 2, 3])
    skus = f1.multiselect("SKU", sorted(df["SKU_ID"].unique()))
    warehouses = f2.multiselect("Warehouse", sorted(df["Warehouse_ID"].unique()))
    regions = f3.multiselect("Region", sorted(df["Region"].unique()))
    suppliers = f4.multiselect("Supplier", sorted(df["Supplier_ID"].unique()))
    min_d, max_d = df["Date"].min().date(), df["Date"].max().date()
    date_range = f5.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)

if skus:
    df = df[df["SKU_ID"].isin(skus)]
if warehouses:
    df = df[df["Warehouse_ID"].isin(warehouses)]
if regions:
    df = df[df["Region"].isin(regions)]
if suppliers:
    df = df[df["Supplier_ID"].isin(suppliers)]
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    df = df[(df["Date"].dt.date >= date_range[0]) & (df["Date"].dt.date <= date_range[1])]

if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ---------------- Current inventory snapshot ----------------
# Latest available record for each SKU + Warehouse combination in the filtered
# data. Inventory is a stock level, so it is never summed across dates.
snap = df.loc[df.groupby(["SKU_ID", "Warehouse_ID"], observed=True)["Date"].idxmax()].copy()
snap["Inventory_Status"] = snap["Inventory_Status"].astype(str)

# Rows missing any value needed for the calculation are excluded, not filled in.
incomplete = snap[REQUIRED_COLS].isna().any(axis=1)
if incomplete.any():
    st.warning(f"{int(incomplete.sum())} SKU-Warehouse combination(s) excluded: missing values needed for the calculation.")
    snap = snap[~incomplete].copy()
if snap.empty:
    st.warning("No complete records available for the selected filters.")
    st.stop()

# ---------------- Replenishment calculation ----------------
snap["Item"] = snap["SKU_ID"].astype(str) + " — " + snap["Warehouse_ID"].astype(str)

# Expected demand during the supplier lead time
snap["Lead_Time_Demand"] = snap["Demand_Forecast"] * snap["Supplier_Lead_Time_Days"]
# Inventory expected to remain when a new order would arrive, if ordered today
snap["Projected_Inventory_At_Arrival"] = snap["Inventory_Level"] - snap["Lead_Time_Demand"]

below_rop = snap["Inventory_Level"] < snap["Reorder_Point"]
cannot_cover = snap["Inventory_Level"] < snap["Lead_Time_Demand"]
severe = snap["Inventory_Status"].isin(["Critical", "Stockout"])
projected_below_rop = snap["Projected_Inventory_At_Arrival"] < snap["Reorder_Point"]

snap["Priority"] = np.select(
    [
        severe | cannot_cover,          # High
        below_rop,                      # Medium
        projected_below_rop,            # Low
    ],
    ["High", "Medium", "Low"],
    default="No Replenishment",
)
snap["Recommendation"] = snap["Priority"].map(
    {"High": "Replenish", "Medium": "Replenish", "Low": "Monitor", "No Replenishment": "No Replenishment"}
)

# Target inventory = Lead Time Demand + Reorder Point (existing buffer level)
snap["Target_Inventory"] = snap["Lead_Time_Demand"] + snap["Reorder_Point"]
shortfall = (snap["Target_Inventory"] - snap["Inventory_Level"]).clip(lower=0)
snap["Recommended_Replenishment_Quantity"] = np.where(
    snap["Recommendation"] == "Replenish", np.ceil(shortfall), 0
).astype(int)
snap["Replenishment_Cost"] = snap["Recommended_Replenishment_Quantity"] * snap["Unit_Cost"]
snap["Historical_Order_Quantity"] = snap["Order_Quantity"]

total_items = len(snap)
replenish = snap[snap["Recommendation"] == "Replenish"]
n_replenish = len(replenish)
total_units = int(replenish["Recommended_Replenishment_Quantity"].sum())
total_cost = replenish["Replenishment_Cost"].sum()
n_high = int((snap["Priority"] == "High").sum())

st.caption(f"Inventory snapshot date: {snap['Date'].max().date()}  |  SKU-Warehouse combinations: {total_items}")


# ---------------- AI Insight ----------------
_top = replenish.sort_values("Recommended_Replenishment_Quantity", ascending=False).head(3)
render_ai_insight("replenishment", {
    "snapshot_date": str(snap["Date"].max().date()),
    "inventory_items": total_items,
    "items_to_replenish": n_replenish,
    "recommended_units": total_units,
    "replenishment_cost": total_cost,
    "high_priority_count": n_high,
    "priority_counts": snap["Priority"].value_counts().to_dict(),
    "monitor_count": int((snap["Recommendation"] == "Monitor").sum()),
    "top_items": [
        {"item": r.Item, "recommended_quantity": r.Recommended_Replenishment_Quantity,
         "priority": r.Priority, "lead_time_demand": r.Lead_Time_Demand}
        for r in _top.itertuples()
    ],
    "note": "Lead time demand = Demand_Forecast x Supplier_Lead_Time_Days (rule-based calculation, not machine learning).",
})


# ---------------- 2. KPIs ----------------
st.subheader("Replenishment KPIs")
k1, k2, k3, k4 = st.columns(4)
kpi(k1, "Items Requiring Replenishment", f"{n_replenish} / {total_items}",
    help="High and Medium priority items.")
kpi(k2, "Recommended Units to Replenish", f"{total_units:,}")
kpi(k3, "Estimated Replenishment Cost", f"{total_cost:,.0f}", help="Recommended quantity × Unit_Cost.")
kpi(k4, "High Priority Items", f"{n_high}")

# ---------------- 3. Priority ----------------
st.subheader("Replenishment Priority")
priority_counts = (
    snap["Priority"].value_counts().reindex(PRIORITY_ORDER, fill_value=0)
    .rename("SKU-Warehouse combinations")
    .rename_axis("Priority")
    .reset_index()
)
with st.container(border=True):
    st.markdown("**SKU-Warehouse Combinations by Replenishment Priority**")
    st.bar_chart(priority_counts, x="Priority", y="SKU-Warehouse combinations", sort=False, height=CHART_HEIGHT)

# ---------------- 4. Recommended replenishment ----------------
st.subheader("Recommended Replenishment")
table_cols = [
    "SKU_ID", "Warehouse_ID", "Supplier_ID", "Region", "Inventory_Level", "Reorder_Point",
    "Demand_Forecast", "Supplier_Lead_Time_Days", "Lead_Time_Demand", "Days_of_Inventory",
    "Historical_Order_Quantity", "Recommended_Replenishment_Quantity", "Replenishment_Cost",
    "Priority", "Recommendation",
]
with st.container(border=True):
    show_all = st.checkbox("Include items with no replenishment")
    view = snap if show_all else snap[snap["Recommendation"] != "No Replenishment"]
    if view.empty:
        st.info("No replenishment is required for the current filtered inventory snapshot.")
    else:
        view = view.assign(_p=view["Priority"].map({p: i for i, p in enumerate(PRIORITY_ORDER)}))
        view = view.sort_values(["_p", "Recommended_Replenishment_Quantity"], ascending=[True, False])
        st.dataframe(
            view[table_cols].round({
                "Demand_Forecast": 1, "Lead_Time_Demand": 1, "Days_of_Inventory": 1, "Replenishment_Cost": 2,
            }),
            width="stretch", hide_index=True,
        )

# ---------------- 5. Recommended quantity ----------------
st.subheader("Recommended Quantity by Item")
with st.container(border=True):
    top_qty = replenish.sort_values("Recommended_Replenishment_Quantity", ascending=False).head(N_TOP)
    if top_qty.empty:
        st.info("No replenishment is required for the current filtered inventory snapshot.")
    else:
        st.markdown(f"**Recommended Replenishment Quantity — Top {len(top_qty)} Items**")
        st.bar_chart(
            top_qty[["Item", "Recommended_Replenishment_Quantity"]].rename(
                columns={"Recommended_Replenishment_Quantity": "Recommended Quantity"}
            ),
            x="Item", y="Recommended Quantity", horizontal=True, sort=False, height=CHART_HEIGHT,
        )

# ---------------- 6. Replenishment cost ----------------
st.subheader("Replenishment Cost")
if replenish.empty:
    with st.container(border=True):
        st.info("No replenishment is required for the current filtered inventory snapshot.")
else:
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("**Replenishment Cost by Warehouse**")
        cost_wh = replenish.groupby("Warehouse_ID", observed=True)["Replenishment_Cost"].sum()
        st.bar_chart(cost_wh.rename("Replenishment Cost"), height=CHART_HEIGHT)
    with c2.container(border=True):
        st.markdown(f"**Replenishment Cost — Top {N_TOP} Items**")
        cost_item = replenish.sort_values("Replenishment_Cost", ascending=False).head(N_TOP)
        st.bar_chart(
            cost_item[["Item", "Replenishment_Cost"]].rename(columns={"Replenishment_Cost": "Replenishment Cost"}),
            x="Item", y="Replenishment Cost", horizontal=True, sort=False, height=CHART_HEIGHT,
        )

# ---------------- 7. Supplier / lead time ----------------
st.subheader("Supplier & Lead Time")
sup = snap.groupby("Supplier_ID", observed=True).agg(
    units=("Recommended_Replenishment_Quantity", "sum"),
    lead_time=("Supplier_Lead_Time_Days", "mean"),
)
s1, s2 = st.columns(2)
with s1.container(border=True):
    st.markdown("**Recommended Replenishment Units by Supplier**")
    st.bar_chart(sup["units"].rename("Recommended Units"), height=CHART_HEIGHT)
with s2.container(border=True):
    st.markdown("**Average Supplier Lead Time (days)**")
    st.bar_chart(sup["lead_time"].rename("Average Lead Time (days)"), height=CHART_HEIGHT)