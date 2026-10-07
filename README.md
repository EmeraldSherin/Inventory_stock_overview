# Inventory Stock Overview

An AI-assisted supply chain inventory analytics and optimization system built with Python and Streamlit. The system helps understand sales performance, demand trends, inventory risk, and replenishment requirements using data-driven analytics.

## Project Overview

Inventory Stock Overview provides an interactive dashboard for analyzing supply chain and inventory data.

The system answers four key business questions:

1. **What's happening?** — Executive sales and inventory overview
2. **What's coming?** — Demand and forecast analysis
3. **What's wrong?** — Inventory risk identification
4. **What should I do?** — Replenishment recommendations

The application also provides a small AI explanation layer using Google Gemini to convert calculated dashboard metrics into concise business insights.

## Features

### 1. Executive Dashboard
- Revenue and gross profit
- Gross margin
- Units sold
- Inventory value
- Items below reorder point
- Monthly revenue and profit trends
- Regional revenue analysis
- Top-performing SKUs
- Warehouse inventory risk

### 2. Demand & Forecast
- Average demand
- Peak demand
- Demand variability
- Demand trends
- Actual demand vs baseline forecast
- Promotion vs non-promotion demand analysis

### 3. Inventory Risk
- Current inventory position
- Reorder-point monitoring
- Critical inventory identification
- Days of inventory
- Inventory status distribution
- Warehouse-level risk analysis
- At-risk SKU and warehouse combinations

### 4. Replenishment & Optimization
- Items requiring replenishment
- Recommended replenishment quantities
- Lead-time demand
- Replenishment cost estimation
- Priority classification
- SKU-Warehouse level recommendations

### 5. AI Insights
Google Gemini generates short explanations from the calculated dashboard metrics.

The AI layer:
- Uses only supplied dashboard metrics
- Does not generate inventory calculations
- Does not invent business values
- Provides contextual explanations
- Uses a rule-based fallback when the LLM is unavailable

## Technology Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Streamlit**
- **Google Gemini API**
- **Matplotlib / Plotly**
- **Joblib**
- **Jupyter Notebook**

## Dataset

The project uses a supply chain inventory dataset containing information related to:

- Date
- SKU
- Warehouse
- Supplier
- Region
- Units Sold
- Inventory Level
- Supplier Lead Time
- Reorder Point
- Order Quantity
- Unit Cost
- Unit Price
- Promotion Flag
- Stockout Flag
- Demand Forecast

The dataset contains daily inventory records across multiple SKUs, warehouses, suppliers, and regions.

## Data Processing

The preprocessing pipeline:

1. Loads the raw supply chain dataset
2. Validates required columns
3. Converts data types
4. Handles date parsing
5. Checks missing values and duplicates
6. Validates SKU-Warehouse data grain
7. Calculates business metrics
8. Creates the prepared dataset used by the dashboard

Derived metrics include:

- Revenue
- Inventory Value
- Gross Profit
- Margin Percentage
- Average Daily Demand
- Demand Variability
- Days of Inventory
- Distance to Reorder Point
- Reorder Status

## Forecasting

Multiple forecasting approaches were evaluated using chronological validation.

The tested models were compared against the existing dataset forecast. The existing baseline forecast was retained because it provided better predictive performance than the tested machine-learning forecasting approaches.

This prevented the system from using a weaker model simply for the sake of including machine learning.

## Project Structure

```text
Inventory_Stock_Overview/
│
├── app.py
├── prepare_data.py
├── requirements.txt
├── .gitignore
│
├── pages/
│   ├── 1_Executive_Dashboard.py
│   ├── 2_Demand_Forecast.py
│   ├── 3_Inventory_Risk.py
│   └── 4_Replenishment_Optimization.py
│
├── utils/
│   ├── __init__.py
│   └── ai_explanation.py
│
├── models/
│
├── notebook/
│
└── data/
