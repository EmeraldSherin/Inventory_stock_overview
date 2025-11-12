import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Beauty Products Inventory & Demand Forecasting",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .alert-box {
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .critical {
        background-color: #ff4444;
        color: white;
    }
    .warning {
        background-color: #ffaa00;
        color: white;
    }
    .success {
        background-color: #00C851;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Sidebar
with st.sidebar:
    st.title("💄 Beauty Inventory System")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "📊 Sales Overview", "📦 Inventory Overview", 
         "🔮 Demand Forecast", "👥 Vendor Performance", 
         "💰 Profitability", "🎯 Recommendations"]
    )
    
    st.markdown("---")
    st.subheader("📁 Data Upload")
    uploaded_file = st.file_uploader("Upload Sales Data (CSV)", type=['csv'])
    
    if uploaded_file:
        st.success("✅ File uploaded!")
    
    st.markdown("---")
    st.info("""
    **Recommended Datasets:**
    1. [Cosmetic Brand Products](https://www.kaggle.com/datasets/shivd24coder/cosmetic-brand-products-dataset)
    2. [Retail Transactions](https://www.kaggle.com/datasets/prasad22/retail-transactions-dataset)
    3. [Online Retail Dataset](https://www.kaggle.com/datasets/abhishekrp1517/online-retail-transactions-dataset)
    """)

# Load sample data function
@st.cache_data
def load_sample_data():
    """Generate realistic beauty products sales data"""
    np.random.seed(42)
    
    # Products
    products = [
        "Hydrating Face Serum", "Matte Lipstick", "Anti-Aging Cream",
        "Volumizing Mascara", "Body Lotion", "Hair Repair Oil",
        "Eyeshadow Palette", "Vitamin C Cleanser", "Foundation",
        "Blush Powder", "Nail Polish", "Perfume", "BB Cream",
        "Concealer", "Lip Gloss", "Eye Liner", "Makeup Remover"
    ]
    
    categories = ["Skincare", "Makeup", "Body Care", "Hair Care", "Fragrance"]
    brands = ["GlowLux", "ColorPop", "YouthGlow", "LashMaster", "PureEssence", "SilkStrands"]
    
    # Generate time series data
    dates = pd.date_range(start='2022-01-01', end='2024-10-31', freq='D')
    
    data = []
    for product in products:
        category = np.random.choice(categories)
        brand = np.random.choice(brands)
        base_price = np.random.uniform(15, 85)
        
        for date in dates:
            # Seasonal pattern
            month = date.month
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * month / 12)
            
            # Trend
            days_since_start = (date - dates[0]).days
            trend = 1 + 0.0002 * days_since_start
            
            # Random variation
            noise = np.random.uniform(0.8, 1.2)
            
            quantity = int(np.random.poisson(15) * seasonal_factor * trend * noise)
            
            data.append({
                'Date': date,
                'Product': product,
                'Category': category,
                'Brand': brand,
                'Quantity_Sold': quantity,
                'Unit_Price': round(base_price * np.random.uniform(0.95, 1.05), 2),
                'Total_Sales': round(quantity * base_price, 2)
            })
    
    df = pd.DataFrame(data)
    return df

@st.cache_data
def generate_inventory_data(sales_df):
    """Generate inventory status from sales data"""
    products = sales_df['Product'].unique()
    
    inventory = []
    for product in products:
        product_sales = sales_df[sales_df['Product'] == product]
        avg_daily_sales = product_sales.groupby('Date')['Quantity_Sold'].sum().mean()
        
        current_stock = int(np.random.uniform(avg_daily_sales * 10, avg_daily_sales * 40))
        reorder_point = int(avg_daily_sales * 15)
        max_stock = int(avg_daily_sales * 50)
        lead_time = np.random.randint(5, 15)
        
        inventory.append({
            'Product': product,
            'Category': product_sales['Category'].iloc[0],
            'Brand': product_sales['Brand'].iloc[0],
            'Current_Stock': current_stock,
            'Reorder_Point': reorder_point,
            'Max_Stock': max_stock,
            'Avg_Daily_Sales': round(avg_daily_sales, 2),
            'Lead_Time_Days': lead_time,
            'Unit_Price': product_sales['Unit_Price'].mean()
        })
    
    return pd.DataFrame(inventory)

# Simple Moving Average Forecast Function
def simple_moving_average_forecast(data, periods=90, window=30):
    """Simple Moving Average for forecasting"""
    # Calculate moving average
    ma = data['y'].rolling(window=window).mean()
    
    # Get last MA value
    last_ma = ma.iloc[-1]
    
    # Create future dates
    last_date = data['ds'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods)
    
    # Simple forecast (using last MA value with slight variation)
    forecast_values = []
    for i in range(periods):
        # Add slight trend and seasonality
        trend = 1 + (i * 0.001)
        seasonal = 1 + 0.1 * np.sin(2 * np.pi * (i % 30) / 30)
        noise = np.random.uniform(0.95, 1.05)
        value = last_ma * trend * seasonal * noise
        forecast_values.append(value)
    
    forecast_df = pd.DataFrame({
        'ds': future_dates,
        'yhat': forecast_values,
        'yhat_lower': [v * 0.85 for v in forecast_values],
        'yhat_upper': [v * 1.15 for v in forecast_values]
    })
    
    return forecast_df

# Load data
if uploaded_file:
    sales_df = pd.read_csv(uploaded_file)
    sales_df['Date'] = pd.to_datetime(sales_df['Date'])
    st.session_state.data_loaded = True
else:
    sales_df = load_sample_data()
    st.session_state.data_loaded = True

inventory_df = generate_inventory_data(sales_df)

# Calculate key metrics
total_sales = sales_df['Total_Sales'].sum()
total_transactions = len(sales_df)
avg_order_value = total_sales / total_transactions
total_products = sales_df['Product'].nunique()

# Page: HOME
if page == "🏠 Home":
    st.title("💄 Beauty Products Inventory & Demand Forecasting System")
    st.markdown("### AI-Powered Retail Analytics Dashboard")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 Total Revenue</h3>
            <h2>${total_sales:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🛍️ Transactions</h3>
            <h2>{total_transactions:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📦 Products</h3>
            <h2>{total_products}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💵 Avg Order</h3>
            <h2>${avg_order_value:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Alerts
    st.subheader("🚨 Quick Alerts")
    
    low_stock = inventory_df[inventory_df['Current_Stock'] < inventory_df['Reorder_Point']]
    overstock = inventory_df[inventory_df['Current_Stock'] > inventory_df['Max_Stock'] * 0.9]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="alert-box critical">
            <h4>⚠️ Low Stock Items: {len(low_stock)}</h4>
            <p>Immediate reorder required!</p>
        </div>
        """, unsafe_allow_html=True)
        
        if len(low_stock) > 0:
            st.dataframe(low_stock[['Product', 'Current_Stock', 'Reorder_Point']].head(5))
    
    with col2:
        st.markdown(f"""
        <div class="alert-box warning">
            <h4>📊 Overstock Items: {len(overstock)}</h4>
            <p>Consider promotions</p>
        </div>
        """, unsafe_allow_html=True)
        
        if len(overstock) > 0:
            st.dataframe(overstock[['Product', 'Current_Stock', 'Max_Stock']].head(5))

# Page: SALES OVERVIEW
elif page == "📊 Sales Overview":
    st.title("📊 Sales Overview")
    
    # Time period selector
    col1, col2 = st.columns([3, 1])
    with col1:
        date_range = st.date_input(
            "Select Date Range",
            value=(sales_df['Date'].min(), sales_df['Date'].max()),
            min_value=sales_df['Date'].min(),
            max_value=sales_df['Date'].max()
        )
    
    # Filter data
    if len(date_range) == 2:
        filtered_df = sales_df[(sales_df['Date'] >= pd.to_datetime(date_range[0])) & 
                               (sales_df['Date'] <= pd.to_datetime(date_range[1]))]
    else:
        filtered_df = sales_df
    
    # Daily sales trend
    daily_sales = filtered_df.groupby('Date').agg({
        'Total_Sales': 'sum',
        'Quantity_Sold': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_sales['Date'],
        y=daily_sales['Total_Sales'],
        mode='lines',
        name='Daily Sales',
        line=dict(color='#667eea', width=2),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))
    
    fig.update_layout(
        title="Daily Sales Trend",
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Category and Brand Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        category_sales = filtered_df.groupby('Category')['Total_Sales'].sum().reset_index()
        fig = px.pie(category_sales, values='Total_Sales', names='Category', 
                     title='Sales by Category',
                     color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        brand_sales = filtered_df.groupby('Brand')['Total_Sales'].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(brand_sales, x='Brand', y='Total_Sales', 
                     title='Top 10 Brands by Sales',
                     color='Total_Sales',
                     color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    # Top Products
    st.subheader("🏆 Top 10 Products")
    top_products = filtered_df.groupby('Product').agg({
        'Total_Sales': 'sum',
        'Quantity_Sold': 'sum'
    }).sort_values('Total_Sales', ascending=False).head(10).reset_index()
    
    st.dataframe(top_products.style.format({
        'Total_Sales': '${:,.2f}',
        'Quantity_Sold': '{:,.0f}'
    }), use_container_width=True)

# Page: INVENTORY OVERVIEW
elif page == "📦 Inventory Overview":
    st.title("📦 Inventory Overview")
    
    # Stock status distribution
    inventory_df['Stock_Status'] = inventory_df.apply(
        lambda row: 'Critical' if row['Current_Stock'] < row['Reorder_Point'] * 0.5
        else 'Low' if row['Current_Stock'] < row['Reorder_Point']
        else 'Overstock' if row['Current_Stock'] > row['Max_Stock'] * 0.9
        else 'Normal',
        axis=1
    )
    
    status_counts = inventory_df['Stock_Status'].value_counts()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔴 Critical", status_counts.get('Critical', 0))
    with col2:
        st.metric("🟡 Low Stock", status_counts.get('Low', 0))
    with col3:
        st.metric("🟢 Normal", status_counts.get('Normal', 0))
    with col4:
        st.metric("🔵 Overstock", status_counts.get('Overstock', 0))
    
    st.markdown("---")
    
    # Stock status visualization
    fig = go.Figure()
    
    colors = {'Critical': 'red', 'Low': 'orange', 'Normal': 'green', 'Overstock': 'blue'}
    
    for status in inventory_df['Stock_Status'].unique():
        df_status = inventory_df[inventory_df['Stock_Status'] == status]
        fig.add_trace(go.Bar(
            x=df_status['Product'],
            y=df_status['Current_Stock'],
            name=status,
            marker_color=colors.get(status, 'gray')
        ))
    
    fig.update_layout(
        title="Current Stock Levels by Product",
        xaxis_title="Product",
        yaxis_title="Stock Units",
        barmode='group',
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed inventory table
    st.subheader("📋 Detailed Inventory Status")
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        category_filter = st.multiselect("Filter by Category", 
                                        inventory_df['Category'].unique(),
                                        default=inventory_df['Category'].unique())
    with col2:
        status_filter = st.multiselect("Filter by Status",
                                      inventory_df['Stock_Status'].unique(),
                                      default=inventory_df['Stock_Status'].unique())
    
    filtered_inventory = inventory_df[
        (inventory_df['Category'].isin(category_filter)) &
        (inventory_df['Stock_Status'].isin(status_filter))
    ]
    
    # Calculate additional metrics
    filtered_inventory['Days_Until_Stockout'] = (
        filtered_inventory['Current_Stock'] / filtered_inventory['Avg_Daily_Sales']
    ).round(1)
    
    filtered_inventory['Stock_Value'] = (
        filtered_inventory['Current_Stock'] * filtered_inventory['Unit_Price']
    ).round(2)
    
    st.dataframe(filtered_inventory.style.format({
        'Current_Stock': '{:.0f}',
        'Reorder_Point': '{:.0f}',
        'Avg_Daily_Sales': '{:.2f}',
        'Days_Until_Stockout': '{:.1f}',
        'Unit_Price': '${:.2f}',
        'Stock_Value': '${:,.2f}'
    }), use_container_width=True)

# Page: DEMAND FORECAST (FIXED)
elif page == "🔮 Demand Forecast":
    st.title("🔮 Demand Forecast")
    
    st.info("📊 Using Simple Moving Average forecasting model (Prophet alternative)")
    
    # Product selector
    selected_product = st.selectbox("Select Product", sales_df['Product'].unique())
    
    # Forecast period
    forecast_days = st.slider("Forecast Period (days)", 30, 180, 90)
    
    # Filter data for selected product
    product_data = sales_df[sales_df['Product'] == selected_product].copy()
    product_data = product_data.groupby('Date')['Quantity_Sold'].sum().reset_index()
    product_data.columns = ['ds', 'y']
    
    # Simple Moving Average Forecast
    with st.spinner('Generating forecast...'):
        forecast = simple_moving_average_forecast(product_data, periods=forecast_days, window=30)
    
    # Visualization
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=product_data['ds'],
        y=product_data['y'],
        mode='lines',
        name='Historical Sales',
        line=dict(color='blue', width=2)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        name='Confidence Interval',
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.1)',
        line=dict(width=0)
    ))
    
    fig.update_layout(
        title=f"Demand Forecast: {selected_product}",
        xaxis_title="Date",
        yaxis_title="Quantity",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast summary
    st.subheader("📈 Forecast Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("30-Day Forecast", f"{int(forecast.head(30)['yhat'].sum())} units")
    with col2:
        st.metric("60-Day Forecast", f"{int(forecast.head(60)['yhat'].sum())} units")
    with col3:
        st.metric("90-Day Forecast", f"{int(forecast.head(90)['yhat'].sum())} units")
    
    # Download forecast
    csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast",
        data=csv,
        file_name=f"{selected_product}_forecast.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    

# Page: VENDOR PERFORMANCE
elif page == "👥 Vendor Performance":
    st.title("👥 Vendor Performance")
    
    # Generate vendor data
    vendors = ['BeautySupply Co', 'Global Cosmetics', 'Premium Beauty', 'FastTrack Suppliers', 'Elite Distributors']
    
    vendor_data = []
    for vendor in vendors:
        vendor_data.append({
            'Vendor': vendor,
            'Total_Orders': np.random.randint(50, 200),
            'On_Time_Delivery_%': np.random.uniform(75, 99),
            'Avg_Lead_Time_Days': np.random.uniform(5, 20),
            'Quality_Score': np.random.uniform(3.5, 5.0),
            'Total_Spend': np.random.uniform(50000, 500000)
        })
    
    vendor_df = pd.DataFrame(vendor_data)
    
    # Vendor performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(vendor_df, x='Vendor', y='On_Time_Delivery_%',
                     title='On-Time Delivery Rate',
                     color='On_Time_Delivery_%',
                     color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(vendor_df, x='Avg_Lead_Time_Days', y='Quality_Score',
                        size='Total_Spend', hover_data=['Vendor'],
                        title='Quality vs Lead Time',
                        color='Quality_Score',
                        color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    # Vendor table
    st.subheader("📊 Vendor Details")
    st.dataframe(vendor_df.style.format({
        'Total_Orders': '{:.0f}',
        'On_Time_Delivery_%': '{:.1f}%',
        'Avg_Lead_Time_Days': '{:.1f}',
        'Quality_Score': '{:.2f}',
        'Total_Spend': '${:,.2f}'
    }), use_container_width=True)

# Page: PROFITABILITY (FIXED)
elif page == "💰 Profitability":
    st.title("💰 Profitability Analysis")
    
    # Calculate profitability metrics
    product_profit = sales_df.groupby('Product').agg({
        'Total_Sales': 'sum',
        'Quantity_Sold': 'sum'
    }).reset_index()
    
    product_profit['Avg_Price'] = product_profit['Total_Sales'] / product_profit['Quantity_Sold']
    product_profit['Est_Cost'] = product_profit['Avg_Price'] * 0.4  # Assume 60% margin
    product_profit['Profit'] = product_profit['Total_Sales'] - (product_profit['Est_Cost'] * product_profit['Quantity_Sold'])
    product_profit['Profit_Margin_%'] = (product_profit['Profit'] / product_profit['Total_Sales']) * 100
    
    # Top and Bottom performers
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 10 Profitable Products")
        top_10 = product_profit.nlargest(10, 'Profit')[['Product', 'Profit', 'Profit_Margin_%']]
        
        fig = px.bar(top_10, x='Product', y='Profit',
                     color='Profit_Margin_%',
                     color_continuous_scale='Greens',
                     title='Top Performers')
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📉 Bottom 10 Products")
        bottom_10 = product_profit.nsmallest(10, 'Profit')[['Product', 'Profit', 'Profit_Margin_%']]
        
        fig = px.bar(bottom_10, x='Product', y='Profit',
                     color='Profit_Margin_%',
                     color_continuous_scale='Reds',
                     title='Low Performers')
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # ENHANCED Profit margin distribution
    st.subheader("📊 Profit Margin Distribution")
    
    # Create bins for better visualization
    bins = [0, 30, 40, 50, 60, 70, 100]
    labels = ['0-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70%+']
    product_profit['Margin_Range'] = pd.cut(product_profit['Profit_Margin_%'], bins=bins, labels=labels)
    
    # Count products in each range
    margin_dist = product_profit['Margin_Range'].value_counts().reset_index()
    margin_dist.columns = ['Margin_Range', 'Count']
    margin_dist = margin_dist.sort_values('Margin_Range')
    
    # Create colorful bar chart
    fig = px.bar(margin_dist, 
                 x='Margin_Range', 
                 y='Count',
                 title='Number of Products by Profit Margin Range',
                 labels={'Margin_Range': 'Profit Margin Range', 'Count': 'Number of Products'},
                 color='Count',
                 color_continuous_scale='Turbo',
                 text='Count')
    
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
   
    # Detailed table
    st.subheader("📋 Detailed Profitability Table")
    st.dataframe(product_profit.style.format({
        'Total_Sales': '${:,.2f}',
        'Quantity_Sold': '{:,.0f}',
        'Avg_Price': '${:.2f}',
        'Est_Cost': '${:.2f}',
        'Profit': '${:,.2f}',
        'Profit_Margin_%': '{:.1f}%'
    }), use_container_width=True)

# Page: RECOMMENDATIONS
elif page == "🎯 Recommendations":
    st.title("🎯 Purchase Recommendation Engine")
    
    st.markdown("""
    ### AI-Powered Reorder Recommendations
    Based on:
    - Historical sales patterns
    - Current stock levels
    - Lead times
    - Economic Order Quantity (EOQ)
    """)
    
    # Calculate EOQ and recommendations
    inventory_df['Daily_Demand'] = inventory_df['Avg_Daily_Sales']
    inventory_df['Annual_Demand'] = inventory_df['Daily_Demand'] * 365
    inventory_df['Holding_Cost'] = inventory_df['Unit_Price'] * 0.25  # 25% of unit cost
    inventory_df['Ordering_Cost'] = 50  # Fixed ordering cost
    
    # EOQ Formula
    inventory_df['EOQ'] = np.sqrt(
        (2 * inventory_df['Annual_Demand'] * inventory_df['Ordering_Cost']) / 
        inventory_df['Holding_Cost']
    ).round(0)
    
    inventory_df['Days_Until_Stockout'] = (
        inventory_df['Current_Stock'] / inventory_df['Daily_Demand']
    ).round(1)
    
    inventory_df['Reorder_Needed'] = (
        inventory_df['Days_Until_Stockout'] <= inventory_df['Lead_Time_Days'] + 5
    )
    
    inventory_df['Recommended_Order_Qty'] = inventory_df['EOQ']
    inventory_df['Order_Priority'] = inventory_df.apply(
        lambda row: 'HIGH' if row['Days_Until_Stockout'] < row['Lead_Time_Days']
        else 'MEDIUM' if row['Days_Until_Stockout'] < row['Lead_Time_Days'] + 10
        else 'LOW',
        axis=1
    )
    
    # Filter items needing reorder
    reorder_items = inventory_df[inventory_df['Reorder_Needed']].sort_values('Days_Until_Stockout')
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        high_priority = len(reorder_items[reorder_items['Order_Priority'] == 'HIGH'])
        st.metric("🔴 High Priority", high_priority)
    
    with col2:
        medium_priority = len(reorder_items[reorder_items['Order_Priority'] == 'MEDIUM'])
        st.metric("🟡 Medium Priority", medium_priority)
    
    with col3:
        total_order_value = (reorder_items['Recommended_Order_Qty'] * reorder_items['Unit_Price']).sum()
        st.metric("💰 Total Order Value", f"${total_order_value:,.2f}")
    
    st.markdown("---")
    
    # Reorder recommendations table
    st.subheader("📋 Reorder Recommendations")
    
    if len(reorder_items) > 0:
        display_cols = ['Product', 'Category', 'Current_Stock', 'Days_Until_Stockout', 
                       'Recommended_Order_Qty', 'Unit_Price', 'Order_Priority']
        
        reorder_items['Order_Value'] = reorder_items['Recommended_Order_Qty'] * reorder_items['Unit_Price']
        
        # Create styled dataframe
        def highlight_priority(val):
            if val == 'HIGH':
                return 'background-color: #ffcccc'
            elif val == 'MEDIUM':
                return 'background-color: #fff4cc'
            else:
                return ''
        
        styled_df = reorder_items[display_cols + ['Order_Value']].style.format({
            'Current_Stock': '{:.0f}',
            'Days_Until_Stockout': '{:.1f}',
            'Recommended_Order_Qty': '{:.0f}',
            'Unit_Price': '${:.2f}',
            'Order_Value': '${:,.2f}'
        }).applymap(highlight_priority, subset=['Order_Priority'])
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Download recommendations
        csv = reorder_items[display_cols + ['Order_Value']].to_csv(index=False)
        st.download_button(
            label="📥 Download Reorder List",
            data=csv,
            file_name="reorder_recommendations.csv",
            mime="text/csv"
        )
    else:
        st.success("✅ All products are adequately stocked!")
    
    # EOQ Analysis
    st.subheader("📊 Economic Order Quantity Analysis")
    
    fig = px.scatter(inventory_df, 
                     x='Annual_Demand', 
                     y='EOQ',
                     size='Unit_Price',
                     color='Category',
                     hover_data=['Product'],
                     title='EOQ vs Annual Demand')
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💄 Beauty Products Inventory & Demand Forecasting System</p>
    <p>Powered by AI | Simple Moving Average | Machine Learning</p>
</div>
""", unsafe_allow_html=True)