import pandas as pd
import numpy as np

pd.set_option('display.width', 160)
pd.set_option('display.max_columns', None)

INPUT_FILE = "supply_chain_dataset1.csv"
OUTPUT_FILE = "prepared_supply_chain.csv"

# The 15 original columns that must be preserved exactly as-is.
ORIGINAL_COLUMNS = [
    "Date", "SKU_ID", "Warehouse_ID", "Supplier_ID", "Region",
    "Units_Sold", "Inventory_Level", "Supplier_Lead_Time_Days",
    "Reorder_Point", "Order_Quantity", "Unit_Cost", "Unit_Price",
    "Promotion_Flag", "Stockout_Flag", "Demand_Forecast",
]

DERIVED_COLUMNS = [
    "Revenue", "Inventory_Value", "Gross_Profit", "Margin_Percent",
    "Avg_Daily_Demand_7D", "Avg_Daily_Demand_30D", "Demand_Std_30D",
    "Demand_CV", "Days_of_Inventory", "Distance_to_Reorder_Point",
    "Below_Reorder_Point", "Inventory_Status",
]


def main():
    print("=" * 70)
    print("SUPPLY CHAIN INVENTORY DATA PREPARATION — LEVEL 2")
    print("=" * 70)

    # ------------------------------------------------------------------
    # STEP 1: Load and validate
    # ------------------------------------------------------------------
    print(f"\n[STEP 1] Loading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    rows_before = len(df)
    cols_before = len(df.columns)
    print(f"Loaded {rows_before:,} rows, {cols_before} columns.")

    missing_cols = [c for c in ORIGINAL_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Expected original columns are missing from the CSV: {missing_cols}. "
            "Stopping rather than guessing or inventing them."
        )

    # Parse Date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    bad_dates = df["Date"].isna().sum()

    # Convert ID columns to string/category
    for col in ["SKU_ID", "Warehouse_ID", "Supplier_ID", "Region"]:
        df[col] = df[col].astype(str).astype("category")

    # Convert numeric columns to appropriate numeric types
    int_cols = [
        "Units_Sold", "Inventory_Level", "Supplier_Lead_Time_Days",
        "Reorder_Point", "Order_Quantity", "Promotion_Flag", "Stockout_Flag",
    ]
    float_cols = ["Unit_Cost", "Unit_Price", "Demand_Forecast"]

    for col in int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    coercion_nulls = df[int_cols + float_cols].isna().sum().sum()

    # --- Validation checks (report only, do not silently fix) ---
    print("\n[STEP 1] Validation report")
    print("-" * 70)

    dup_count = df.duplicated().sum()
    print(f"Duplicate rows (all columns identical): {dup_count}")

    missing_report = df[ORIGINAL_COLUMNS].isna().sum()
    missing_total = missing_report.sum()
    print(f"Missing values per column:")
    for col, n in missing_report.items():
        if n > 0:
            print(f"   - {col}: {n}")
    if missing_total == 0:
        print("   None found.")

    if bad_dates > 0:
        print(f"WARNING: {bad_dates} rows have an unparseable Date value.")

    if coercion_nulls > 0:
        print(f"WARNING: {coercion_nulls} numeric values could not be parsed "
              f"and became NaN during type conversion.")

    neg_units_sold = (df["Units_Sold"] < 0).sum()
    print(f"Negative Units_Sold: {neg_units_sold}")

    neg_inventory = (df["Inventory_Level"] < 0).sum()
    print(f"Negative Inventory_Level: {neg_inventory}")

    invalid_cost = (df["Unit_Cost"] <= 0).sum()
    print(f"Invalid Unit_Cost (<= 0): {invalid_cost}")

    invalid_price = (df["Unit_Price"] <= 0).sum()
    print(f"Invalid Unit_Price (<= 0): {invalid_price}")

    price_below_cost = (df["Unit_Price"] < df["Unit_Cost"]).sum()
    print(f"Rows where Unit_Price < Unit_Cost: {price_below_cost}")

    # Conservative handling: flag, don't fabricate corrections.
    # Only rows that are structurally unusable (unparseable date, or a
    # value so invalid that downstream division/ratio math would be
    # undefined, e.g. Unit_Price <= 0) are excluded, and each exclusion
    # is counted and reported explicitly below. No values are altered.
    exclusion_mask = df["Date"].isna() | (df["Unit_Price"] <= 0)
    n_excluded = exclusion_mask.sum()
    if n_excluded > 0:
        print(f"\nExcluding {n_excluded} row(s) that are structurally invalid "
              f"(unparseable Date and/or Unit_Price <= 0). No other rows are "
              f"modified or removed.")
        df = df[~exclusion_mask].copy()
    else:
        print("\nNo structurally invalid rows found. No rows excluded.")

    if dup_count > 0:
        print(f"NOTE: {dup_count} exact duplicate row(s) detected but NOT "
              f"removed automatically — review before deciding whether they "
              f"are genuine repeats or legitimate data.")

    # ------------------------------------------------------------------
    # STEP 2: Original columns preserved as-is (no renaming) — nothing to
    # do here beyond what's already been done: types were normalized in
    # place, values were not changed, and no column was renamed.
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # STEP 3: Add reliable derived features
    # ------------------------------------------------------------------
    print("\n[STEP 3] Adding derived features")
    print("-" * 70)

    # Sort chronologically within each SKU + Warehouse group so that all
    # rolling calculations below only ever see current and past rows for
    # that specific SKU/Warehouse — never a different group's data and
    # never a future date. This sort is required before any .rolling()
    # call to avoid future-data leakage.
    df = df.sort_values(["SKU_ID", "Warehouse_ID", "Date"]).reset_index(drop=True)
    group_key = ["SKU_ID", "Warehouse_ID"]

    # 1. Revenue
    df["Revenue"] = df["Units_Sold"] * df["Unit_Price"]

    # 2. Inventory Value
    df["Inventory_Value"] = df["Inventory_Level"] * df["Unit_Cost"]

    # 3. Gross Profit
    df["Gross_Profit"] = df["Units_Sold"] * (df["Unit_Price"] - df["Unit_Cost"])

    # 4. Margin Percent (safe division: 0 wherever Unit_Price is 0,
    # which by this point has already been excluded, but the guard is
    # kept in case Unit_Price is ever exactly 0 in future data refreshes)
    df["Margin_Percent"] = np.where(
        df["Unit_Price"] > 0,
        ((df["Unit_Price"] - df["Unit_Cost"]) / df["Unit_Price"]) * 100,
        np.nan,
    )

    # 5. 7-day rolling average demand per SKU + Warehouse.
    # groupby(...).rolling(window, min_periods=1) computed on data already
    # sorted ascending by Date only ever looks backward from each row —
    # this is standard trailing/causal rolling, not centered or forward.
    df["Avg_Daily_Demand_7D"] = (
        df.groupby(group_key, observed=True)["Units_Sold"]
        .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
    )

    # 6. 30-day rolling average demand per SKU + Warehouse
    df["Avg_Daily_Demand_30D"] = (
        df.groupby(group_key, observed=True)["Units_Sold"]
        .transform(lambda s: s.rolling(window=30, min_periods=1).mean())
    )

    # 7. 30-day rolling standard deviation of demand per SKU + Warehouse
    df["Demand_Std_30D"] = (
        df.groupby(group_key, observed=True)["Units_Sold"]
        .transform(lambda s: s.rolling(window=30, min_periods=1).std())
    )
    # First observation in each group has an undefined std (only one
    # point); leave as NaN rather than inventing a value.

    # 8. Demand coefficient of variation (safe division)
    df["Demand_CV"] = np.where(
        (df["Avg_Daily_Demand_30D"] > 0) & df["Demand_Std_30D"].notna(),
        df["Demand_Std_30D"] / df["Avg_Daily_Demand_30D"],
        np.nan,
    )

    # 9. Days of Inventory (safe division)
    df["Days_of_Inventory"] = np.where(
        df["Avg_Daily_Demand_30D"] > 0,
        df["Inventory_Level"] / df["Avg_Daily_Demand_30D"],
        np.nan,
    )

    # 10. Distance to Reorder Point
    df["Distance_to_Reorder_Point"] = df["Inventory_Level"] - df["Reorder_Point"]

    # 11. Below Reorder Point
    df["Below_Reorder_Point"] = df["Inventory_Level"] < df["Reorder_Point"]

    # 12. Inventory Status (transparent rule-based classification;
    # no "Overstock" tier — the dataset provides no defensible maximum
    # stock threshold to compare against)
    conditions = [
        df["Inventory_Level"] <= 0,
        df["Inventory_Level"] <= 0.5 * df["Reorder_Point"],
        df["Inventory_Level"] < df["Reorder_Point"],
    ]
    choices = ["Stockout", "Critical", "Below Reorder Point"]
    df["Inventory_Status"] = np.select(conditions, choices, default="Healthy")

    print(f"Added {len(DERIVED_COLUMNS)} derived columns: {DERIVED_COLUMNS}")

    # Note: original Stockout_Flag is left completely untouched — it is
    # not recalculated, relabeled, or used to derive Inventory_Status.
    # Inventory_Status is a separate, clearly-derived column based only
    # on Inventory_Level and Reorder_Point.

    # ------------------------------------------------------------------
    # STEP 5: Verify dataset grain preserved
    # ------------------------------------------------------------------
    rows_after = len(df)
    cols_after = len(df.columns)

    grain_check = df.duplicated(subset=["Date", "SKU_ID", "Warehouse_ID"]).sum()

    print("\n[STEP 5] Grain verification")
    print("-" * 70)
    print(f"Row count before cleaning:  {rows_before:,}")
    print(f"Row count after cleaning:   {rows_after:,}")
    print(f"Unique SKU_ID:              {df['SKU_ID'].nunique()}")
    print(f"Unique Warehouse_ID:        {df['Warehouse_ID'].nunique()}")
    print(f"Date range:                 {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Columns before:             {cols_before}")
    print(f"Columns after:              {cols_after}")
    print(f"Duplicate (Date, SKU_ID, Warehouse_ID) combinations: {grain_check}")
    if grain_check > 0:
        print("WARNING: grain is not unique per Date+SKU+Warehouse — investigate before use.")
    else:
        print("Grain confirmed: exactly one row per Date + SKU_ID + Warehouse_ID.")

    # ------------------------------------------------------------------
    # STEP 6: Output
    # ------------------------------------------------------------------
    print("\n[STEP 6] Summary")
    print("-" * 70)
    print(f"Original columns ({len(ORIGINAL_COLUMNS)}): {ORIGINAL_COLUMNS}")
    print(f"Derived columns ({len(DERIVED_COLUMNS)}): {DERIVED_COLUMNS}")
    print(f"Final shape: {df.shape}")

    print("\nFirst 5 rows:")
    preview_cols = ORIGINAL_COLUMNS + DERIVED_COLUMNS
    print(df[preview_cols].head().to_string())

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved prepared dataset to: {OUTPUT_FILE}")
    print("=" * 70)
    print("DONE — no ML model trained, no fabricated data introduced.")
    print("=" * 70)


if __name__ == "__main__":
    main()