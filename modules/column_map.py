from __future__ import annotations

import pandas as pd
from typing import Optional


# Actual column names from the dataset
COLUMN_MAP = {
    "sales": "Sales",
    "order_date": "order date (DateOrders)",
    "market": "Market",
    "category": "Category Name",
    "late_delivery_risk": "Late_delivery_risk",
    "actual_shipping_days": "Days for shipping (real)",
    "scheduled_shipping_days": "Days for shipment (scheduled)",
    "order_id": "Order Id",
    "profit": "Order Profit Per Order",
    "benefit_per_order": "Benefit per order",
    "shipping_mode": "Shipping Mode",
    "order_status": "Order Status",
    "region": "Region",
}


def get_col(df, logical_name: str) -> Optional[str]:
    """
    Get the actual column name from the DataFrame for a logical column name.
    Returns None if column doesn't exist.
    """
    actual_name = COLUMN_MAP.get(logical_name)
    if actual_name and actual_name in df.columns:
        return actual_name
    # Fallback: try case-insensitive match
    for col in df.columns:
        if col.lower() == logical_name.lower():
            return col
    return None


def get_col_required(df, logical_name: str) -> str:
    """
    Get the actual column name, raise KeyError if not found.
    """
    col = get_col(df, logical_name)
    if col is None:
        raise KeyError(f"Required column '{logical_name}' (mapped to '{COLUMN_MAP.get(logical_name)}') not found in DataFrame")
    return col


def has_col(df, logical_name: str) -> bool:
    """Check if a logical column exists in the DataFrame."""
    return get_col(df, logical_name) is not None


def ensure_datetime(df, logical_name: str):
    """
    Ensure a column is datetime type. Returns the column name if successful,
    None if column doesn't exist or conversion fails.
    """
    col = get_col(df, logical_name)
    if col is None:
        return None
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            return None
    return col