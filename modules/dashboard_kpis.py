from __future__ import annotations

import pandas as pd
import numpy as np

from modules.column_map import get_col, has_col


def safe_sum(series: pd.Series) -> float | np.floating:
    """Safely sum a series, returning NaN if empty or all NaN.
    
    Handles numeric values, numeric strings (e.g., "125.50"), missing values,
    and invalid strings (e.g., "unknown", "N/A") by coercing invalid values to NaN.
    """
    if series is None or series.empty:
        return np.nan
    # Convert to numeric, coercing invalid values to NaN
    numeric_series = pd.to_numeric(series, errors="coerce")
    # Drop NaN values and sum
    valid = numeric_series.dropna()
    if valid.empty:
        return np.nan
    return float(valid.sum())


def safe_mean(series: pd.Series) -> float | np.floating:
    """Safely calculate mean, returning NaN if empty or all NaN.
    
    Handles numeric values, numeric strings (e.g., "125.50"), missing values,
    and invalid strings (e.g., "unknown", "N/A") by coercing invalid values to NaN.
    """
    if series is None or series.empty:
        return np.nan
    # Convert to numeric, coercing invalid values to NaN
    numeric_series = pd.to_numeric(series, errors="coerce")
    # Drop NaN values and calculate mean
    valid = numeric_series.dropna()
    if valid.empty:
        return np.nan
    return float(valid.mean())


def safe_nunique(series: pd.Series) -> int | np.floating:
    """Safely count unique values, returning NaN if empty."""
    if series is None or series.empty:
        return np.nan
    return int(series.nunique(dropna=True))


def safe_count(series: pd.Series) -> int | np.floating:
    """Safely count non-null values."""
    if series is None or series.empty:
        return np.nan
    return int(series.count())


def calculate_kpis(df: pd.DataFrame) -> dict:
    """
    Calculate executive KPIs from the dataset.
    
    Uses actual dataset column names via column_map.
    Returns NaN for missing/invalid calculations instead of 0.
    """
    kpis = {
        "total_sales": np.nan,
        "total_orders": np.nan,
        "avg_profit_per_order": np.nan,
        "late_delivery_risk_pct": np.nan,
        "avg_actual_shipping_days": np.nan,
    }

    if df is None or df.empty:
        return kpis

    # Total Sales
    sales_col = get_col(df, "sales")
    if sales_col:
        kpis["total_sales"] = safe_sum(df[sales_col])

    # Total Orders (distinct Order Id)
    order_col = get_col(df, "order_id")
    if order_col:
        kpis["total_orders"] = safe_nunique(df[order_col])
    else:
        # Fallback: count rows
        kpis["total_orders"] = safe_count(df.iloc[:, 0])

    # Average Profit per Order
    # Use "Order Profit Per Order" as primary, fallback to "Benefit per order"
    profit_col = get_col(df, "profit")
    if not profit_col:
        profit_col = get_col(df, "benefit_per_order")
    
    if profit_col and order_col:
        total_profit = safe_sum(df[profit_col])
        total_orders = kpis["total_orders"]
        if not pd.isna(total_orders) and total_orders > 0:
            kpis["avg_profit_per_order"] = total_profit / total_orders
    elif profit_col:
        kpis["avg_profit_per_order"] = safe_mean(df[profit_col])

    # Late-Delivery Risk Percentage
    # Late_delivery_risk is 0/1 flag - calculate percentage of 1s
    risk_col = get_col(df, "late_delivery_risk")
    if risk_col:
        series = df[risk_col]
        valid = series.dropna()
        if len(valid) > 0:
            # Count where value == 1 (at risk)
            at_risk = (valid == 1).sum()
            kpis["late_delivery_risk_pct"] = float(at_risk / len(valid) * 100)

    # Average Actual Shipping Days
    shipping_col = get_col(df, "actual_shipping_days")
    if shipping_col:
        kpis["avg_actual_shipping_days"] = safe_mean(df[shipping_col])

    return kpis


def format_currency(value: float | np.floating) -> str:
    """Format currency value with compact notation."""
    if pd.isna(value):
        return "N/A"
    abs_val = abs(value)
    if abs_val >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    elif abs_val >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    elif abs_val >= 1_000:
        return f"${value / 1_000:.2f}K"
    else:
        return f"${value:,.2f}"


def format_count(value: float | np.floating) -> str:
    """Format count value with compact notation."""
    if pd.isna(value):
        return "N/A"
    abs_val = abs(value)
    if abs_val >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif abs_val >= 1_000:
        return f"{value / 1_000:.1f}K"
    else:
        return f"{int(value):,}"


def format_percentage(value: float | np.floating) -> str:
    """Format percentage value with one decimal place."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f}%"


def format_days(value: float | np.floating) -> str:
    """Format days value with one decimal place and unit."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.1f} days"


def format_kpi_value(value: float | np.floating, kpi_type: str) -> str:
    """Format KPI value for display based on type."""
    if kpi_type == "currency":
        return format_currency(value)
    elif kpi_type == "count":
        return format_count(value)
    elif kpi_type == "percentage":
        return format_percentage(value)
    elif kpi_type == "days":
        return format_days(value)
    else:
        return str(value)


def get_kpi_definitions() -> list[dict]:
    """Return KPI definitions for rendering cards."""
    return [
        {
            "key": "total_sales",
            "label": "Total Sales",
            "type": "currency",
            "icon": "💰",
            "help": "Sum of all sales amounts",
        },
        {
            "key": "total_orders",
            "label": "Total Orders",
            "type": "count",
            "icon": "📦",
            "help": "Count of distinct Order IDs",
        },
        {
            "key": "avg_profit_per_order",
            "label": "Avg Profit per Order",
            "type": "currency",
            "icon": "📈",
            "help": "Total profit divided by order count",
        },
        {
            "key": "late_delivery_risk_pct",
            "label": "Late-Delivery Risk %",
            "type": "percentage",
            "icon": "⚠️",
            "help": "Percentage of orders with Late_delivery_risk = 1",
        },
        {
            "key": "avg_actual_shipping_days",
            "label": "Avg Shipping Days",
            "type": "days",
            "icon": "🚚",
            "help": "Mean actual shipping days per order (Days for shipping (real))",
        },
    ]