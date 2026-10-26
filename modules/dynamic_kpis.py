from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from modules.semantic_model import (
    ColumnRole,
    BusinessDomain,
    DatasetProfile,
    SemanticColumn,
)
from modules.column_map import get_col


@dataclass
class KPISpec:
    """Specification for a dynamically generated KPI."""
    key: str
    label: str
    type: str  # "currency", "count", "percentage", "days", "ratio"
    icon: str
    formula: str
    source_columns: list[str]
    domain: BusinessDomain
    is_valid: bool
    validity_reason: str = ""
    value: float | np.floating = np.nan


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


def get_column(df: pd.DataFrame, name: str) -> Optional[str]:
    """Get actual column name, trying exact match first then normalized."""
    if name in df.columns:
        return name
    # Try normalized match
    from modules.semantic_model import normalize_column_name
    norm_target = normalize_column_name(name)
    for col in df.columns:
        if normalize_column_name(col) == norm_target:
            return col
    return None


def generate_general_kpis(df: pd.DataFrame, profile: DatasetProfile) -> list[KPISpec]:
    """Generate general KPIs applicable to any dataset."""
    kpis = []
    
    # Total records
    kpis.append(KPISpec(
        key="total_records",
        label="Total Records",
        type="count",
        icon="📋",
        formula="COUNT(*)",
        source_columns=[],
        domain=BusinessDomain.GENERAL,
        is_valid=True,
        value=float(len(df)),
    ))
    
    # Distinct entities (using primary identifier)
    primary_id = profile.get_primary_identifier()
    if primary_id:
        distinct_count = safe_nunique(df[primary_id])
        kpis.append(KPISpec(
            key="distinct_entities",
            label=f"Distinct {primary_id}",
            type="count",
            icon="🔑",
            formula=f"COUNT(DISTINCT {primary_id})",
            source_columns=[primary_id],
            domain=BusinessDomain.GENERAL,
            is_valid=not pd.isna(distinct_count),
            validity_reason="Primary identifier found" if not pd.isna(distinct_count) else "Identifier column has no valid values",
            value=distinct_count,
        ))
    
    # Missing data percentage
    total_cells = df.size
    missing_cells = int(df.isna().sum().sum())
    missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0
    kpis.append(KPISpec(
        key="missing_data_pct",
        label="Missing Data %",
        type="percentage",
        icon="🕳️",
        formula="Missing cells / Total cells × 100",
        source_columns=list(df.columns),
        domain=BusinessDomain.GENERAL,
        is_valid=True,
        value=float(missing_pct),
    ))
    
    # Duplicate rows
    dup_count = int(df.duplicated().sum())
    kpis.append(KPISpec(
        key="duplicate_rows",
        label="Duplicate Rows",
        type="count",
        icon="📋",
        formula="COUNT(*) WHERE row is duplicate",
        source_columns=[],
        domain=BusinessDomain.GENERAL,
        is_valid=True,
        value=float(dup_count),
    ))
    
    return kpis


def generate_numeric_kpis(df: pd.DataFrame, profile: DatasetProfile) -> list[KPISpec]:
    """Generate KPIs for all numeric measure columns in any dataset."""
    kpis = []
    
    # For each measure column, generate sum, mean, min, max
    for col in profile.measures + profile.currencies:
        if col not in df.columns:
            continue
        
        # Sum
        total = safe_sum(df[col])
        if not pd.isna(total):
            kpis.append(KPISpec(
                key=f"sum_{col}",
                label=f"Sum of {col}",
                type="currency" if col in profile.currencies else "count",
                icon="💰" if col in profile.currencies else "📊",
                formula=f"SUM({col})",
                source_columns=[col],
                domain=BusinessDomain.GENERAL,
                is_valid=True,
                value=total,
            ))
        
        # Mean
        mean_val = safe_mean(df[col])
        if not pd.isna(mean_val):
            kpis.append(KPISpec(
                key=f"mean_{col}",
                label=f"Average {col}",
                type="currency" if col in profile.currencies else "count",
                icon="📈" if col in profile.currencies else "📊",
                formula=f"AVG({col})",
                source_columns=[col],
                domain=BusinessDomain.GENERAL,
                is_valid=True,
                value=mean_val,
            ))
        
        # Min/Max
        numeric_series = pd.to_numeric(df[col], errors="coerce").dropna()
        if not numeric_series.empty:
            min_val = float(numeric_series.min())
            max_val = float(numeric_series.max())
            kpis.append(KPISpec(
                key=f"min_{col}",
                label=f"Min {col}",
                type="currency" if col in profile.currencies else "count",
                icon="📉",
                formula=f"MIN({col})",
                source_columns=[col],
                domain=BusinessDomain.GENERAL,
                is_valid=True,
                value=min_val,
            ))
            kpis.append(KPISpec(
                key=f"max_{col}",
                label=f"Max {col}",
                type="currency" if col in profile.currencies else "count",
                icon="📈",
                formula=f"MAX({col})",
                source_columns=[col],
                domain=BusinessDomain.GENERAL,
                is_valid=True,
                value=max_val,
            ))
    
    return kpis


def generate_sales_retail_kpis(df: pd.DataFrame, profile: DatasetProfile) -> list[KPISpec]:
    """Generate sales/retail specific KPIs."""
    kpis = []
    
    # Revenue (primary currency measure) - check both currencies and measures
    revenue_col = None
    for col in profile.currencies + profile.measures:
        if any(kw in col.lower() for kw in ["revenue", "sales", "amount", "total", "price"]):
            revenue_col = col
            break
    if not revenue_col and profile.currencies:
        revenue_col = profile.currencies[0]
    
    if revenue_col:
        total_revenue = safe_sum(df[revenue_col])
        kpis.append(KPISpec(
            key="total_revenue",
            label="Total Revenue",
            type="currency",
            icon="💰",
            formula=f"SUM({revenue_col})",
            source_columns=[revenue_col],
            domain=BusinessDomain.SALES_RETAIL,
            is_valid=not pd.isna(total_revenue),
            validity_reason="Revenue column found and has valid values" if not pd.isna(total_revenue) else "Revenue column has no valid values",
            value=total_revenue,
        ))
    
    # Orders count (using identifier or date)
    order_col = None
    for col in profile.identifiers:
        if any(kw in col.lower() for kw in ["order", "transaction", "invoice", "receipt"]):
            order_col = col
            break
    if not order_col:
        order_col = profile.get_primary_identifier()
    
    if order_col:
        order_count = safe_nunique(df[order_col])
        kpis.append(KPISpec(
            key="total_orders",
            label="Total Orders",
            type="count",
            icon="📦",
            formula=f"COUNT(DISTINCT {order_col})",
            source_columns=[order_col],
            domain=BusinessDomain.SALES_RETAIL,
            is_valid=not pd.isna(order_count),
            validity_reason="Order identifier found" if not pd.isna(order_count) else "Order identifier has no valid values",
            value=order_count,
        ))
        
        # Average Order Value (if revenue exists)
        if revenue_col and not pd.isna(order_count) and order_count > 0:
            total_revenue = safe_sum(df[revenue_col])
            if not pd.isna(total_revenue):
                kpis.append(KPISpec(
                    key="avg_order_value",
                    label="Avg Order Value",
                    type="currency",
                    icon="📊",
                    formula=f"SUM({revenue_col}) / COUNT(DISTINCT {order_col})",
                    source_columns=[revenue_col, order_col],
                    domain=BusinessDomain.SALES_RETAIL,
                    is_valid=True,
                    value=total_revenue / order_count,
                ))
    
    # Units sold (quantity measure)
    qty_col = None
    for col in profile.measures:
        if any(kw in col.lower() for kw in ["quantity", "qty", "units", "volume"]):
            qty_col = col
            break
    if qty_col:
        total_qty = safe_sum(df[qty_col])
        kpis.append(KPISpec(
            key="total_units",
            label="Total Units Sold",
            type="count",
            icon="📦",
            formula=f"SUM({qty_col})",
            source_columns=[qty_col],
            domain=BusinessDomain.SALES_RETAIL,
            is_valid=not pd.isna(total_qty),
            validity_reason="Quantity column found" if not pd.isna(total_qty) else "Quantity column has no valid values",
            value=total_qty,
        ))
    
    # Profit margin (if profit column exists)
    profit_col = None
    for col in profile.currencies + profile.measures:
        if any(kw in col.lower() for kw in ["profit", "margin", "gain"]):
            profit_col = col
            break
    if profit_col and revenue_col:
        total_profit = safe_sum(df[profit_col])
        total_revenue = safe_sum(df[revenue_col])
        if not pd.isna(total_profit) and not pd.isna(total_revenue) and total_revenue != 0:
            margin_pct = (total_profit / total_revenue) * 100
            kpis.append(KPISpec(
                key="profit_margin",
                label="Profit Margin %",
                type="percentage",
                icon="📈",
                formula=f"SUM({profit_col}) / SUM({revenue_col}) × 100",
                source_columns=[profit_col, revenue_col],
                domain=BusinessDomain.SALES_RETAIL,
                is_valid=True,
                value=margin_pct,
            ))
    
    return kpis


def generate_churn_kpis(df: pd.DataFrame, profile: DatasetProfile) -> list[KPISpec]:
    """Generate churn-specific KPIs."""
    kpis = []
    
    # Find churn indicator (boolean column related to churn)
    churn_col = None
    for col in profile.booleans:
        if any(kw in col.lower() for kw in ["churn", "retained", "active", "cancel"]):
            churn_col = col
            break
    
    if not churn_col:
        # Look for categorical churn column
        for col in profile.dimensions:
            if any(kw in col.lower() for kw in ["churn", "status", "state"]):
                # Check if values suggest churn
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 10:
                    str_vals = [str(v).lower() for v in unique_vals]
                    if any("churn" in v or "cancel" in v or "lost" in v for v in str_vals):
                        churn_col = col
                        break
    
    if churn_col:
        churn_series = df[churn_col].dropna()
        if len(churn_series) > 0:
            if churn_series.dtype == bool or set(churn_series.unique()).issubset({0, 1, True, False}):
                # Binary churn flag
                churned = (churn_series == 1).sum() if churn_series.dtype != bool else churn_series.sum()
                total = len(churn_series)
                churn_rate = (churned / total * 100) if total > 0 else 0
                kpis.append(KPISpec(
                    key="churn_rate",
                    label="Churn Rate %",
                    type="percentage",
                    icon="📉",
                    formula=f"COUNT({churn_col}=1) / COUNT({churn_col}) × 100",
                    source_columns=[churn_col],
                    domain=BusinessDomain.CHURN,
                    is_valid=True,
                    value=churn_rate,
                ))
            else:
                # Categorical - find churn-like values
                str_vals = churn_series.astype(str).str.lower()
                churn_mask = str_vals.str.contains("churn|cancel|lost|inactive", na=False)
                churned = churn_mask.sum()
                total = len(churn_series)
                churn_rate = (churned / total * 100) if total > 0 else 0
                kpis.append(KPISpec(
                    key="churn_rate",
                    label="Churn Rate %",
                    type="percentage",
                    icon="📉",
                    formula=f"COUNT(churn-like status) / COUNT({churn_col}) × 100",
                    source_columns=[churn_col],
                    domain=BusinessDomain.CHURN,
                    is_valid=True,
                    value=churn_rate,
                ))
    
    # Customer count
    customer_col = profile.get_primary_identifier()
    if customer_col:
        cust_count = safe_nunique(df[customer_col])
        kpis.append(KPISpec(
            key="total_customers",
            label="Total Customers",
            type="count",
            icon="👥",
            formula=f"COUNT(DISTINCT {customer_col})",
            source_columns=[customer_col],
            domain=BusinessDomain.CHURN,
            is_valid=not pd.isna(cust_count),
            validity_reason="Customer identifier found" if not pd.isna(cust_count) else "Customer identifier has no valid values",
            value=cust_count,
        ))
    
    return kpis


def generate_supply_chain_kpis(df: pd.DataFrame, profile: DatasetProfile) -> list[KPISpec]:
    """Generate supply chain specific KPIs."""
    kpis = []
    
    # Orders/shipments count
    order_col = None
    for col in profile.identifiers:
        if any(kw in col.lower() for kw in ["order", "shipment", "delivery", "po", "purchase"]):
            order_col = col
            break
    if not order_col:
        order_col = profile.get_primary_identifier()
    
    if order_col:
        order_count = safe_nunique(df[order_col])
        kpis.append(KPISpec(
            key="total_shipments",
            label="Total Shipments",
            type="count",
            icon="🚚",
            formula=f"COUNT(DISTINCT {order_col})",
            source_columns=[order_col],
            domain=BusinessDomain.SUPPLY_CHAIN,
            is_valid=not pd.isna(order_count),
            validity_reason="Shipment identifier found" if not pd.isna(order_count) else "Shipment identifier has no valid values",
            value=order_count,
        ))
    
    # Shipping days (actual)
    shipping_col = None
    for col in profile.measures:
        if any(kw in col.lower() for kw in ["ship", "delivery", "transit", "lead_time", "days"]):
            shipping_col = col
            break
    
    if shipping_col:
        avg_shipping = safe_mean(df[shipping_col])
        kpis.append(KPISpec(
            key="avg_shipping_days",
            label="Avg Shipping Days",
            type="days",
            icon="📦",
            formula=f"AVG({shipping_col})",
            source_columns=[shipping_col],
            domain=BusinessDomain.SUPPLY_CHAIN,
            is_valid=not pd.isna(avg_shipping),
            validity_reason="Shipping days column found" if not pd.isna(avg_shipping) else "Shipping days column has no valid values",
            value=avg_shipping,
        ))
    
    # Late delivery risk
    risk_col = None
    for col in profile.booleans:
        if any(kw in col.lower() for kw in ["late", "risk", "delay", "on_time"]):
            risk_col = col
            break
    if not risk_col:
        for col in profile.measures:
            if any(kw in col.lower() for kw in ["late", "risk", "delay", "on_time"]):
                risk_col = col
                break
    
    if risk_col:
        risk_series = df[risk_col].dropna()
        if len(risk_series) > 0:
            if risk_series.dtype == bool or set(risk_series.unique()).issubset({0, 1}):
                late_count = (risk_series == 1).sum() if risk_series.dtype != bool else risk_series.sum()
                late_pct = (late_count / len(risk_series) * 100)
                kpis.append(KPISpec(
                    key="late_delivery_rate",
                    label="Late Delivery Rate %",
                    type="percentage",
                    icon="⚠️",
                    formula=f"COUNT({risk_col}=1) / COUNT({risk_col}) × 100",
                    source_columns=[risk_col],
                    domain=BusinessDomain.SUPPLY_CHAIN,
                    is_valid=True,
                    value=late_pct,
                ))
    
    return kpis


def generate_hr_kpis(df: pd.DataFrame, profile: DatasetProfile) -> list[KPISpec]:
    """Generate HR specific KPIs."""
    kpis = []
    
    # Headcount
    emp_col = None
    for col in profile.identifiers:
        if any(kw in col.lower() for kw in ["employee", "emp", "staff", "personnel", "worker"]):
            emp_col = col
            break
    if not emp_col:
        emp_col = profile.get_primary_identifier()
    
    if emp_col:
        headcount = safe_nunique(df[emp_col])
        kpis.append(KPISpec(
            key="headcount",
            label="Headcount",
            type="count",
            icon="👥",
            formula=f"COUNT(DISTINCT {emp_col})",
            source_columns=[emp_col],
            domain=BusinessDomain.HR,
            is_valid=not pd.isna(headcount),
            validity_reason="Employee identifier found" if not pd.isna(headcount) else "Employee identifier has no valid values",
            value=headcount,
        ))
    
    # Salary statistics
    salary_col = None
    for col in profile.currencies:
        if any(kw in col.lower() for kw in ["salary", "wage", "compensation", "pay", "income"]):
            salary_col = col
            break
    
    if salary_col:
        avg_salary = safe_mean(df[salary_col])
        kpis.append(KPISpec(
            key="avg_salary",
            label="Average Salary",
            type="currency",
            icon="💵",
            formula=f"AVG({salary_col})",
            source_columns=[salary_col],
            domain=BusinessDomain.HR,
            is_valid=not pd.isna(avg_salary),
            validity_reason="Salary column found" if not pd.isna(avg_salary) else "Salary column has no valid values",
            value=avg_salary,
        ))
    
    return kpis


def generate_loan_finance_kpis(df: pd.DataFrame, profile: DatasetProfile) -> list[KPISpec]:
    """Generate loan/finance specific KPIs."""
    kpis = []
    
    # Loan portfolio total
    loan_col = None
    for col in profile.currencies:
        if any(kw in col.lower() for kw in ["loan", "principal", "balance", "amount", "credit"]):
            loan_col = col
            break
    
    if loan_col:
        total_loans = safe_sum(df[loan_col])
        kpis.append(KPISpec(
            key="total_loan_balance",
            label="Total Loan Balance",
            type="currency",
            icon="🏦",
            formula=f"SUM({loan_col})",
            source_columns=[loan_col],
            domain=BusinessDomain.LOAN_FINANCE,
            is_valid=not pd.isna(total_loans),
            validity_reason="Loan amount column found" if not pd.isna(total_loans) else "Loan amount column has no valid values",
            value=total_loans,
        ))
        
        # Average loan size
        loan_count = safe_nunique(df[loan_col]) if df[loan_col].dtype in ["object", "string"] else safe_count(df[loan_col])
        if not pd.isna(loan_count) and loan_count > 0 and not pd.isna(total_loans):
            kpis.append(KPISpec(
                key="avg_loan_size",
                label="Average Loan Size",
                type="currency",
                icon="💰",
                formula=f"SUM({loan_col}) / COUNT({loan_col})",
                source_columns=[loan_col],
                domain=BusinessDomain.LOAN_FINANCE,
                is_valid=True,
                value=total_loans / loan_count,
            ))
    
    # Default rate
    default_col = None
    for col in profile.booleans:
        if any(kw in col.lower() for kw in ["default", "delinqu", "npl", "bad", "charge_off"]):
            default_col = col
            break
    if not default_col:
        for col in profile.dimensions:
            if any(kw in col.lower() for kw in ["default", "status", "stage", "risk"]):
                unique_vals = df[col].dropna().unique()
                str_vals = [str(v).lower() for v in unique_vals]
                if any("default" in v or "bad" in v or "charge" in v or "npl" in v for v in str_vals):
                    default_col = col
                    break
    
    if default_col:
        default_series = df[default_col].dropna()
        if len(default_series) > 0:
            if default_series.dtype == bool or set(default_series.unique()).issubset({0, 1}):
                default_count = (default_series == 1).sum() if default_series.dtype != bool else default_series.sum()
                default_rate = (default_count / len(default_series) * 100)
                kpis.append(KPISpec(
                    key="default_rate",
                    label="Default Rate %",
                    type="percentage",
                    icon="⚠️",
                    formula=f"COUNT({default_col}=1) / COUNT({default_col}) × 100",
                    source_columns=[default_col],
                    domain=BusinessDomain.LOAN_FINANCE,
                    is_valid=True,
                    value=default_rate,
                ))
    
    return kpis


def generate_dynamic_kpis(df: pd.DataFrame, profile: DatasetProfile) -> list[KPISpec]:
    """Generate all relevant KPIs based on dataset profile and domain."""
    all_kpis = []
    
    # Always include general KPIs
    all_kpis.extend(generate_general_kpis(df, profile))
    
    # Always include numeric KPIs for any dataset with measures/currencies
    all_kpis.extend(generate_numeric_kpis(df, profile))
    
    # Add domain-specific KPIs based on detected domain confidence
    # Only add if domain confidence is reasonably high (> 0.3)
    if profile.primary_domain == BusinessDomain.SALES_RETAIL and profile.domain_confidence > 0.3:
        all_kpis.extend(generate_sales_retail_kpis(df, profile))
    elif profile.primary_domain == BusinessDomain.CHURN and profile.domain_confidence > 0.3:
        all_kpis.extend(generate_churn_kpis(df, profile))
    elif profile.primary_domain == BusinessDomain.SUPPLY_CHAIN and profile.domain_confidence > 0.3:
        all_kpis.extend(generate_supply_chain_kpis(df, profile))
    elif profile.primary_domain == BusinessDomain.HR and profile.domain_confidence > 0.3:
        all_kpis.extend(generate_hr_kpis(df, profile))
    elif profile.primary_domain == BusinessDomain.LOAN_FINANCE and profile.domain_confidence > 0.3:
        all_kpis.extend(generate_loan_finance_kpis(df, profile))
    # For GENERAL, AMBIGUOUS, or low-confidence domains: only general + numeric KPIs
    # Do NOT try all domain KPIs as this produces misleading metrics
    
    # Deduplicate by key (in case multiple domains generate same KPI)
    seen = set()
    unique_kpis = []
    for kpi in all_kpis:
        if kpi.key not in seen:
            seen.add(kpi.key)
            unique_kpis.append(kpi)
    
    return unique_kpis


def format_kpi_value(value: float | np.floating, kpi_type: str) -> str:
    """Format KPI value for display based on type."""
    if pd.isna(value):
        return "N/A"
    
    if kpi_type == "currency":
        # Full currency value with comma separators and 2 decimal places
        return f"${value:,.2f}"
    elif kpi_type == "count":
        # Full integer with comma separators
        return f"{int(value):,}"
    elif kpi_type == "percentage":
        return f"{value:.1f}%"
    elif kpi_type == "days":
        return f"{value:.1f} days"
    elif kpi_type == "ratio":
        return f"{value:.2f}"
    else:
        return str(value)


def kpi_specs_to_dict(kpis: list[KPISpec]) -> dict:
    """Convert KPISpec list to dict for compatibility with existing dashboard code."""
    return {kpi.key: kpi.value for kpi in kpis}


def kpi_specs_to_definitions(kpis: list[KPISpec]) -> list[dict]:
    """Convert KPISpec list to definitions format for dashboard rendering."""
    return [
        {
            "key": kpi.key,
            "label": kpi.label,
            "type": kpi.type,
            "icon": kpi.icon,
            "help": f"{kpi.formula} | {kpi.validity_reason}" if not kpi.is_valid else kpi.formula,
            "domain": kpi.domain.value,
            "is_valid": kpi.is_valid,
        }
        for kpi in kpis
    ]