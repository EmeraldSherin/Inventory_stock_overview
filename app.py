import streamlit as st

st.set_page_config(page_title="Inventory Stock Overview", layout="wide")

pages = [
    st.Page("pages/1_Executive_Dashboard.py", default=True),
    st.Page("pages/2_Demand_Forecast.py"),
    st.Page("pages/3_Inventory_Risk.py"),
    st.Page("pages/4_Replenishment_Optimization.py"),
]

# Navigation is built manually so the title appears above the page links
# and the entry file ("app") does not show up in the sidebar.
nav = st.navigation(pages, position="hidden")

with st.sidebar:
    st.title("INVENTORY STOCK OVERVIEW")
    for p in pages:
        st.page_link(p)

nav.run()