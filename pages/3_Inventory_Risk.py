import os
import subprocess
import sys

import pandas as pd
import streamlit as st

from utils.ai_explanation import render_ai_insight

st.set_page_config(page_title="Inventory Risk", layout="wide")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREPARED_FILE = os.path.join(BASE_DIR, "prepared_supply_chain.csv")
PREP_SCRIPT = os.path.join(BASE_DIR, "prepare_data.py")

CHART_HEIGHT = 320
N_AT_RISK = 10
STATUS_ORDER = ["Healthy", "Below Reorder Point", "Critical", "Stockout"]
SEVERITY = {"Stockout": 0, "Critical": 1, "Below Reorder Point": 2, "Healthy": 3}


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


st.title("Inventory Risk")

df = load_data()

# ---------------- 1. Filters ----------------
with st.container(border=True):
    f1, f2, f3, f4 = st.columns([2, 2, 2, 3])
    skus = f1.multiselect("SKU", sorted(df["SKU_ID"].unique()))
    warehouses = f2.multiselect("Warehouse", sorted(df["Warehouse_ID"].unique()))
    regions = f3.multiselect("Region", sorted(df["Region"].unique()))
    min_d, max_d = df["Date"].min().date(), df["Date"].max().date()
    date_range = f4.date_input("Date range", (min_d, max_d), min_value=min_d, max_value=max_d)

if skus:
    df = df[df["SKU_ID"].isin(skus)]
if warehouses:
    df = df[df["Warehouse_ID"].isin(warehouses)]
if regions:
    df = df[df["Region"].isin(regions)]
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    df = df[(df["Date"].dt.date >= date_range[0]) & (df["Date"].dt.date <= date_range[1])]

if df.empty:
    st.warning("No data for the selected filters.")
    st.stop()

# ---------------- Current inventory snapshot ----------------
# Latest available record for each SKU + Warehouse combination in the filtered
# data. Inventory is a stock level, so it is never summed across dates.
latest = df.loc[df.groupby(["SKU_ID", "Warehouse_ID"], observed=True)["Date"].idxmax()].copy()
latest["Inventory_Status"] = latest["Inventory_Status"].astype(str)
latest["Item"] = latest["SKU_ID"].astype(str) + " — " + latest["Warehouse_ID"].astype(str)

total_items = len(latest)
inv_value = latest["Inventory_Value"].sum()
below_rop = int(latest["Below_Reorder_Point"].sum())
below_rop_pct = below_rop / total_items * 100
critical = int((latest["Inventory_Status"] == "Critical").sum())
avg_doi = latest["Days_of_Inventory"].mean()

st.caption(f"Inventory snapshot date: {latest['Date'].max().date()}  |  SKU-Warehouse combinations: {total_items}")


# ---------------- AI Insight ----------------
_wh_below = latest.groupby("Warehouse_ID", observed=True)["Below_Reorder_Point"].sum().sort_values(ascending=False)
_risk = latest.sort_values("Distance_to_Reorder_Point").head(3)
render_ai_insight("inventory_risk", {
    "snapshot_date": str(latest["Date"].max().date()),
    "inventory_items": total_items,
    "total_inventory_value": inv_value,
    "below_reorder_count": below_rop,
    "below_reorder_pct": below_rop_pct,
    "critical_count": critical,
    "stockout_count": int((latest["Inventory_Status"] == "Stockout").sum()),
    "average_days_of_inventory": avg_doi,
    "status_counts": latest["Inventory_Status"].value_counts().to_dict(),
    "most_at_risk": [
        {"item": r.Item, "inventory_level": r.Inventory_Level, "reorder_point": r.Reorder_Point,
         "distance_to_reorder_point": r.Distance_to_Reorder_Point}
        for r in _risk.itertuples()
    ],
    "highest_risk_warehouse": _wh_below.index[0] if below_rop > 0 else None,
    "highest_risk_warehouse_count": _wh_below.iloc[0] if below_rop > 0 else None,
})


# ---------------- 2. KPIs ----------------
st.subheader("Current Inventory Risk KPIs")
k1, k2, k3, k4, k5 = st.columns(5)
kpi(k1, "Total Inventory Value", f"{inv_value:,.0f}")
kpi(
    k2, "Below Reorder Point", f"{below_rop}",
    help="Inventory Level is below the Reorder Point. Includes Critical items.",
)
kpi(
    k3,"Rate of Below Reorder Point", f"{below_rop_pct:.1f}%", )
kpi(
    k4, "Critical Items", f"{critical}",
    note="No critical inventory observed in the current snapshot." if critical == 0 else None,
    help="Inventory Level is at or below 50% of the Reorder Point.",
)

kpi(k5, "Average Days of Inventory", f"{avg_doi:.1f}" if pd.notna(avg_doi) else "N/A")

# ---------------- 3. Status distribution ----------------
st.subheader("Inventory Status Distribution")
status_counts = (
    latest["Inventory_Status"].value_counts().reindex(STATUS_ORDER, fill_value=0)
    .rename("SKU-Warehouse combinations")
    .rename_axis("Inventory Status")
    .reset_index()
)
with st.container(border=True):
    st.markdown("**SKU-Warehouse Combinations by Inventory Status**")
    st.bar_chart(
        status_counts, x="Inventory Status", y="SKU-Warehouse combinations",
        sort=False, height=CHART_HEIGHT,
    )

# ---------------- 4. Inventory Level vs Reorder Point ----------------
st.subheader("Inventory Level vs Reorder Point")
most_at_risk = latest.sort_values("Distance_to_Reorder_Point").head(N_AT_RISK)
with st.container(border=True):
    st.markdown(f"**Inventory Level vs Reorder Point — {len(most_at_risk)} Most At-Risk Items**")
    st.caption("Items are ordered by Distance to Reorder Point (Inventory Level minus Reorder Point), lowest first.")
    chart_data = most_at_risk[["Item", "Inventory_Level", "Reorder_Point"]].rename(
        columns={"Inventory_Level": "Inventory Level", "Reorder_Point": "Reorder Point"}
    )
    st.bar_chart(
        chart_data, x="Item", y=["Inventory Level", "Reorder Point"],
        horizontal=True, stack=False, sort=False, height=CHART_HEIGHT + 80,
    )
    st.dataframe(
        most_at_risk[[
            "SKU_ID", "Warehouse_ID", "Inventory_Level", "Reorder_Point",
            "Distance_to_Reorder_Point", "Days_of_Inventory", "Inventory_Status",
        ]].round({"Days_of_Inventory": 1}),
        width="stretch", hide_index=True,
    )

# ---------------- 5. Days of inventory ----------------
st.subheader("Days of Inventory")
doi = latest["Days_of_Inventory"].dropna()
d1, d2 = st.columns(2)
with d1.container(border=True):
    st.markdown("**Distribution of Days of Inventory (5-day ranges)**")
    if doi.empty:
        st.info("Days of Inventory is not available for the selected data.")
    else:
        edges = list(range(0, int(doi.max() // 5 + 2) * 5, 5))
        bins = pd.cut(doi, bins=edges, right=False)
        hist = bins.value_counts(sort=False).reset_index()
        hist.columns = ["Days of Inventory", "SKU-Warehouse combinations"]
        hist["Days of Inventory"] = hist["Days of Inventory"].apply(lambda i: f"{int(i.left)}-{int(i.right)}")
        st.bar_chart(hist, x="Days of Inventory", y="SKU-Warehouse combinations", sort=False, height=CHART_HEIGHT)
with d2.container(border=True):
    st.markdown("**Lowest Days of Inventory**")
    lowest = (
        latest.dropna(subset=["Days_of_Inventory"])
        .sort_values("Days_of_Inventory")
        .head(10)[["SKU_ID", "Warehouse_ID", "Region", "Days_of_Inventory", "Inventory_Level", "Inventory_Status"]]
    )
    st.dataframe(lowest.round({"Days_of_Inventory": 1}), width="stretch", hide_index=True, height=CHART_HEIGHT - 40)

# ---------------- 6. Warehouse risk ----------------
st.subheader("Warehouse Risk")
wh = latest.groupby("Warehouse_ID", observed=True).agg(
    below=("Below_Reorder_Point", "sum"),
    critical=("Inventory_Status", lambda s: (s == "Critical").sum()),
    avg_doi=("Days_of_Inventory", "mean"),
    value=("Inventory_Value", "sum"),
)
w1, w2 = st.columns(2)
with w1.container(border=True):
    st.markdown("**Items Below Reorder Point by Warehouse**")
    st.bar_chart(wh["below"].rename("Below Reorder Point"), height=CHART_HEIGHT)
with w2.container(border=True):
    st.markdown("**Critical Items by Warehouse**")
    st.bar_chart(wh["critical"].rename("Critical Items"), height=CHART_HEIGHT)
w3, w4 = st.columns(2)
with w3.container(border=True):
    st.markdown("**Average Days of Inventory by Warehouse**")
    st.bar_chart(wh["avg_doi"].rename("Average Days of Inventory"), height=CHART_HEIGHT)
with w4.container(border=True):
    st.markdown("**Inventory Value by Warehouse**")
    st.bar_chart(wh["value"].rename("Inventory Value"), height=CHART_HEIGHT)

# ---------------- 7. At-risk inventory ----------------
st.subheader("At-Risk Inventory")
at_risk = latest[latest["Below_Reorder_Point"]].copy()
at_risk["_severity"] = at_risk["Inventory_Status"].map(SEVERITY)
at_risk = at_risk.sort_values(["_severity", "Distance_to_Reorder_Point"])[
    [
        "SKU_ID", "Warehouse_ID", "Region", "Inventory_Level", "Reorder_Point",
        "Distance_to_Reorder_Point", "Days_of_Inventory", "Inventory_Status", "Inventory_Value",
    ]
]
with st.container(border=True):
    if at_risk.empty:
        st.info("No SKU-Warehouse combinations are below reorder point in the current snapshot.")
    else:
        st.dataframe(at_risk.round({"Days_of_Inventory": 1, "Inventory_Value": 2}), width="stretch", hide_index=True)