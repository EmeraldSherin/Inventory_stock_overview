"""
Custom Data Preparation Script for Your Retail Dataset
Converts your dataset to beauty products inventory format
"""

import pandas as pd
import numpy as np
from datetime import datetime

print("="*70)
print("💄 PREPARING YOUR DATASET FOR BEAUTY INVENTORY SYSTEM")
print("="*70)

# STEP 1: Load your dataset
# CHANGE THIS to match your file name
input_file = "cosmetics_data.csv"  # Change to your actual filename

print(f"\n📂 Loading: {input_file}")

try:
    df = pd.read_csv(input_file, encoding='latin-1')  # Try latin-1 encoding
    print(f"✅ Loaded {len(df):,} rows")
except:
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
        print(f"✅ Loaded {len(df):,} rows")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        print("Please check:")
        print("1. File name is correct")
        print("2. File is in same folder as this script")
        print("3. File is not open in Excel")
        exit()

print(f"📊 Original columns: {list(df.columns)}")

# STEP 2: Create beauty product categories
print("\n🔄 Converting to beauty products format...")

# Map your products to beauty categories
def categorize_beauty_product(description):
    """Intelligently categorize products as beauty items"""
    if pd.isna(description):
        return 'General'
    
    desc = str(description).lower()
    
    # Beauty product keywords
    if any(word in desc for word in ['cream', 'serum', 'cleanser', 'moisturizer', 'lotion', 
                                      'toner', 'mask', 'oil', 'balm', 'treatment']):
        return 'Skincare'
    elif any(word in desc for word in ['lipstick', 'mascara', 'foundation', 'eyeshadow', 
                                        'makeup', 'blush', 'concealer', 'powder', 'eyeliner']):
        return 'Makeup'
    elif any(word in desc for word in ['shampoo', 'conditioner', 'hair', 'styling']):
        return 'Hair Care'
    elif any(word in desc for word in ['perfume', 'fragrance', 'cologne', 'scent']):
        return 'Fragrance'
    elif any(word in desc for word in ['body', 'bath', 'soap', 'scrub', 'shower']):
        return 'Body Care'
    elif any(word in desc for word in ['nail', 'polish', 'manicure']):
        return 'Nails'
    else:
        # For other products, assign to general beauty category
        categories = ['Skincare', 'Makeup', 'Hair Care', 'Body Care', 'Accessories']
        return np.random.choice(categories)

# Map your products to beauty brands
def assign_beauty_brand(description):
    """Assign beauty brand names"""
    brands = [
        'GlowLux', 'ColorPop', 'YouthGlow', 'LashMaster', 
        'PureEssence', 'SilkStrands', 'BeautyPro', 'RadiantSkin',
        'LuxeCosmetics', 'UrbanBeauty', 'NaturalGlow', 'EliteBeauty'
    ]
    
    # Hash description to consistently assign same brand to same product
    if pd.notna(description):
        idx = hash(str(description)) % len(brands)
        return brands[idx]
    return 'Unknown'

import re

def clean_product_name(product, brand=None):
    if pd.isna(product):
        return "Unknown Product"

    product = str(product).lower()

    # Remove brand name if already inside product text
    if brand and isinstance(brand, str):
        product = product.replace(brand.lower(), '')

    # Remove sizes & numbers (50ml, 4.5g, etc.)
    product = re.sub(r'\b\d+(ml|g|gm|kg|l|oz|pack)\b', '', product)
    product = re.sub(r'\b\d+\b', '', product)

    # Keep text only
    product = re.sub(r'[^a-zA-Z ]', ' ', product)

    # Remove filler words
    remove_words = ["for", "with", "and", "the", "of", "by"]
    for w in remove_words:
        product = product.replace(f" {w} ", " ")

    product = re.sub(r'\s+', ' ', product).strip()

    # Format name
    if brand and brand != "Unknown":
        product = f"{brand} - {product}"

    product = product.title()

    # Limit name length
    return product[:38]


# STEP 3: Rename columns to required format
print("✅ Mapping columns...")

df_prepared = pd.DataFrame()

# Map your columns to required format
df_prepared['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# ✅ PRODUCT NAME (short clean name)
raw_product = df.get('product_name')
raw_brand = df.get('brand')

df_prepared['Product'] = [
    clean_product_name(p, b) for p, b in zip(raw_product, raw_brand)
]


# ✅ BRAND
df_prepared['Brand'] = df.get('brand')

# ✅ CATEGORY (use Master Category if available, else category)
df_prepared['Category'] = df.get('category')

# ✅ QUANTITY SOLD
df_prepared['Quantity_Sold'] = pd.to_numeric(df.get('Qty'), errors='coerce')

# ✅ UNIT PRICE (prefer Price column, fallback to MRP)
df_prepared['Unit_Price'] = pd.to_numeric(df.get('Price'), errors='coerce')

df_prepared['Subcategory'] = df.get('subcategory')

# ✅ TOTAL SALES (if exists use, else calculate)
if 'Net Sales calculated' in df.columns:
    df_prepared['Total_Sales'] = pd.to_numeric(df['Net Sales calculated'], errors='coerce')
else:
    df_prepared['Total_Sales'] = df_prepared['Quantity_Sold'] * df_prepared['Unit_Price']

# ✅ Fill missing values
df_prepared.fillna({
    'Brand': 'Unknown',
    'Category': 'Unknown',
    'Product': 'Unknown Product'
}, inplace=True)

print("✅ Column mapping completed!")


# STEP 4: Data Cleaning
print("\n🧹 Cleaning data...")

# Remove rows with missing critical data
initial_rows = len(df_prepared)
df_prepared = df_prepared.dropna(subset=['Date', 'Product', 'Quantity_Sold', 'Unit_Price'])
print(f"✅ Removed {initial_rows - len(df_prepared)} rows with missing data")

# Remove negative quantities (returns/cancellations)
negative = (df_prepared['Quantity_Sold'] < 0).sum()
df_prepared = df_prepared[df_prepared['Quantity_Sold'] > 0]
print(f"✅ Removed {negative} negative quantity transactions")

# Remove zero or negative prices
invalid_price = (df_prepared['Unit_Price'] <= 0).sum()
df_prepared = df_prepared[df_prepared['Unit_Price'] > 0]
print(f"✅ Removed {invalid_price} invalid price records")

# Remove extreme outliers (keep 99% of data)
Q1_qty = df_prepared['Quantity_Sold'].quantile(0.01)
Q3_qty = df_prepared['Quantity_Sold'].quantile(0.99)
Q1_price = df_prepared['Unit_Price'].quantile(0.01)
Q3_price = df_prepared['Unit_Price'].quantile(0.99)

outliers = len(df_prepared)
df_prepared = df_prepared[
    (df_prepared['Quantity_Sold'] >= Q1_qty) & 
    (df_prepared['Quantity_Sold'] <= Q3_qty) &
    (df_prepared['Unit_Price'] >= Q1_price) & 
    (df_prepared['Unit_Price'] <= Q3_price)
]
print(f"✅ Removed {outliers - len(df_prepared)} extreme outliers")

# Clean product names
df_prepared['Product'] = df_prepared['Product'].str.strip()
df_prepared['Product'] = df_prepared['Product'].str.title()

# Recalculate Total_Sales after cleaning
df_prepared['Total_Sales'] = df_prepared['Quantity_Sold'] * df_prepared['Unit_Price']

# STEP 5: Sort by date
df_prepared = df_prepared.sort_values('Date').reset_index(drop=True)

# STEP 6: Generate Summary Statistics
print("\n" + "="*70)
print("📊 PREPARED DATASET SUMMARY")
print("="*70)

print(f"\n📅 Date Range: {df_prepared['Date'].min().date()} to {df_prepared['Date'].max().date()}")
print(f"📆 Total Days: {(df_prepared['Date'].max() - df_prepared['Date'].min()).days} days")

print(f"\n📦 Total Products: {df_prepared['Product'].nunique()}")
print(f"🏷️  Total Categories: {df_prepared['Category'].nunique()}")
print(f"🏢 Total Brands: {df_prepared['Brand'].nunique()}")

print(f"\n📈 Total Transactions: {len(df_prepared):,}")
print(f"💰 Total Revenue: ${df_prepared['Total_Sales'].sum():,.2f}")
print(f"📊 Average Transaction Value: ${df_prepared['Total_Sales'].mean():.2f}")
print(f"📊 Average Quantity per Order: {df_prepared['Quantity_Sold'].mean():.1f}")
print(f"💵 Average Unit Price: ${df_prepared['Unit_Price'].mean():.2f}")

print("\n📊 Category Distribution:")
category_counts = df_prepared['Category'].value_counts()
for cat, count in category_counts.items():
    pct = (count / len(df_prepared)) * 100
    print(f"   {cat}: {count:,} transactions ({pct:.1f}%)")

print("\n🏆 Top 10 Products by Revenue:")
top_products = df_prepared.groupby('Product')['Total_Sales'].sum().sort_values(ascending=False).head(10)
for i, (product, sales) in enumerate(top_products.items(), 1):
    print(f"   {i}. {product[:40]}: ${sales:,.2f}")

print("\n💰 Top 5 Brands by Revenue:")
top_brands = df_prepared.groupby('Brand')['Total_Sales'].sum().sort_values(ascending=False).head(5)
for i, (brand, sales) in enumerate(top_brands.items(), 1):
    print(f"   {i}. {brand}: ${sales:,.2f}")

print("\n📅 Monthly Sales Overview:")
df_prepared['Month'] = df_prepared['Date'].dt.to_period('M')
monthly_sales = df_prepared.groupby('Month')['Total_Sales'].sum().tail(6)
for month, sales in monthly_sales.items():
    print(f"   {month}: ${sales:,.2f}")

# STEP 7: Data Quality Check
print("\n" + "="*70)
print("✅ DATA QUALITY CHECK")
print("="*70)

required_cols = ['Date', 'Product', 'Category', 'Brand', 'Quantity_Sold', 'Unit_Price', 'Total_Sales']
missing = [col for col in required_cols if col not in df_prepared.columns]

if missing:
    print(f"❌ Missing columns: {missing}")
else:
    print("✅ All required columns present")

print(f"✅ No missing values in critical columns")
print(f"✅ All quantities positive: {(df_prepared['Quantity_Sold'] > 0).all()}")
print(f"✅ All prices positive: {(df_prepared['Unit_Price'] > 0).all()}")
print(f"✅ Date range valid: {df_prepared['Date'].min()} to {df_prepared['Date'].max()}")

# STEP 8: Save prepared dataset
output_file = "prepared_beauty_inventory.csv"
df_prepared.to_csv(output_file, index=False)

print("\n" + "="*70)
print("🎉 SUCCESS! YOUR DATASET IS READY!")
print("="*70)
print(f"\n📁 Output File: {output_file}")
print(f"📊 Rows Prepared: {len(df_prepared):,} (from original {initial_rows:,})")
print(f"💾 File Size: {df_prepared.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

print("\n🚀 NEXT STEPS:")
print("="*70)
print("\n✅ OPTION 1: Upload in Streamlit App (Easiest)")
print("   1. Run: streamlit run app.py")
print("   2. Look at left sidebar")
print("   3. Click 'Browse files' under '📁 Data Upload'")
print(f"   4. Select: {output_file}")
print("   5. Done! Your data will appear instantly")

print("\n✅ OPTION 2: Load Directly in Code")
print("   1. Open app.py")
print("   2. Find line ~200 (where it says 'Load data')")
print("   3. Change to:")
print(f"      sales_df = pd.read_csv('{output_file}')")
print("   4. Run: streamlit run app.py")

print("\n💡 TIP: Keep both files!")
print(f"   - Original: {input_file} (backup)")
print(f"   - Prepared: {output_file} (for app)")

print("\n" + "="*70)
print("✨ Your beauty products inventory system is ready to use!")
print("="*70)

# STEP 9: Create sample preview
print("\n📋 Sample of prepared data (first 5 rows):")
print(df_prepared.head().to_string())

print("\n✅ Preparation Complete! Ready to analyze! 🎯")