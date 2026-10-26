from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict:
    """Comprehensive data profiling including memory usage and type analysis."""
    # Helper to safely get column as Series (handles duplicate column names)
    def get_series(col):
        series = df[col]
        if isinstance(series, pd.DataFrame):
            return series.iloc[:, 0]
        return series
    
    missing_by_column = pd.DataFrame({
        "Column": df.columns.astype(str),
        "Missing values": [int(get_series(col).isna().sum()) for col in df.columns],
        "Missing %": [
            round(float(get_series(col).isna().mean() * 100), 2) if len(df) else 0.0
            for col in df.columns
        ],
    })
    missing_by_column = missing_by_column[missing_by_column["Missing values"] > 0]
    missing_by_column = missing_by_column.sort_values(
        ["Missing values", "Column"], ascending=[False, True]
    ).reset_index(drop=True)

    # Memory usage per column (approximate)
    memory_usage = df.memory_usage(deep=True, index=False)
    memory_by_column = pd.DataFrame({
        "Column": df.columns.astype(str),
        "Memory (bytes)": memory_usage.values,
        "Memory (MB)": [round(v / (1024 * 1024), 3) for v in memory_usage.values],
    })

    # Data type analysis
    dtype_info = pd.DataFrame({
        "Column": df.columns.astype(str),
        "Data type": [str(get_series(col).dtype) for col in df.columns],
        "Non-null values": [int(get_series(col).notna().sum()) for col in df.columns],
        "Unique values": [int(get_series(col).nunique(dropna=True)) for col in df.columns],
    })

    # Detect potential date columns (object/string columns that look like dates)
    date_candidates = detect_date_columns(df)

    # Identify inconsistent text categories (low cardinality text columns with fuzzy duplicates)
    text_inconsistencies = detect_text_inconsistencies(df)

    # Numeric column suspicious values
    numeric_issues = detect_numeric_issues(df)

    total_memory_mb = round(float(memory_usage.sum() / (1024 * 1024)), 3)

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "empty_columns": [str(col) for col in df.columns if get_series(col).isna().all()],
        "missing_by_column": missing_by_column,
        "memory_by_column": memory_by_column,
        "total_memory_mb": total_memory_mb,
        "dtype_info": dtype_info,
        "date_candidates": date_candidates,
        "text_inconsistencies": text_inconsistencies,
        "numeric_issues": numeric_issues,
    }


# Regex pattern -> (strftime format, description)
DATE_PATTERNS = [
    (r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d", "ISO (YYYY-MM-DD)"),
    (r"^\d{2}/\d{2}/\d{4}$", "%m/%d/%Y", "US (MM/DD/YYYY)"),
    (r"^\d{2}-\d{2}-\d{4}$", "%d-%m-%Y", "EU (DD-MM-YYYY)"),
    (r"^\d{4}/\d{2}/\d{2}$", "%Y/%m/%d", "YYYY/MM/DD"),
    (r"^\d{2}\.\d{2}.\d{4}$", "%d.%m.%Y", "DD.MM.YYYY"),
    (r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", "%Y-%m-%dT%H:%M:%S", "ISO datetime"),
    (r"^\d{2}/\d{2}/\d{2}$", "%m/%d/%y", "Short (MM/DD/YY)"),
]


def detect_date_columns(df: pd.DataFrame) -> list[dict]:
    """Detect columns that appear to contain date-like strings but are stored as object/string."""
    candidates = []

    for col in df.select_dtypes(include=["object", "string"]).columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        # Sample up to 100 non-null values
        sample = non_null.head(100).astype(str)
        for pattern, fmt, desc in DATE_PATTERNS:
            matches = sample.str.match(pattern).sum()
            if matches / len(sample) >= 0.8:  # 80% match threshold
                # Try parsing to confirm
                parse_result = try_parse_dates(df[col], fmt)
                candidates.append({
                    "column": str(col),
                    "suspected_format": desc,
                    "pattern": pattern,
                    "strftime_format": fmt,
                    "sample_values": sample.head(5).tolist(),
                    "parse_success_rate": parse_result["success_rate"],
                    "parse_errors": parse_result["errors"],
                })
                break  # Only report first matching pattern

    return candidates


def try_parse_dates(series: pd.Series, strftime_format: str) -> dict:
    """Attempt to parse a series as dates using strftime format, return success rate and any errors."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return {"success_rate": 0.0, "errors": []}

    sample = non_null.head(200).astype(str)
    try:
        parsed = pd.to_datetime(sample, format=strftime_format, errors="coerce")
        success = int(parsed.notna().sum())
        errors = sample[parsed.isna()].head(5).tolist()
        return {
            "success_rate": round(success / len(sample) * 100, 1),
            "errors": errors,
        }
    except Exception as exc:
        return {"success_rate": 0.0, "errors": [str(exc)]}


def detect_text_inconsistencies(df: pd.DataFrame, max_cardinality: int = 50) -> list[dict]:
    """Identify text columns with potential inconsistent categories (fuzzy duplicates)."""
    inconsistencies = []

    for col in df.select_dtypes(include=["object", "string"]).columns:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue

        unique_vals = non_null.unique()
        if len(unique_vals) > max_cardinality:
            continue  # Skip high-cardinality columns

        # Normalize for comparison: lowercase, strip, collapse whitespace
        normalized_map = {}
        for val in unique_vals:
            if not isinstance(val, str):
                continue
            norm = re.sub(r"\s+", " ", val.strip().lower())
            if norm not in normalized_map:
                normalized_map[norm] = []
            normalized_map[norm].append(val)

        # Find groups with multiple original values mapping to same normalized value
        fuzzy_groups = {k: v for k, v in normalized_map.items() if len(v) > 1}
        if fuzzy_groups:
            total_affected = sum(len(v) for v in fuzzy_groups.values())
            inconsistencies.append({
                "column": str(col),
                "cardinality": int(len(unique_vals)),
                "fuzzy_groups": fuzzy_groups,
                "affected_unique_values": total_affected,
            })

    return inconsistencies


def detect_numeric_issues(df: pd.DataFrame) -> list[dict]:
    """Detect suspicious numeric values: extreme outliers, unexpected negatives, zeros where not expected."""
    issues = []

    for col in df.select_dtypes(include="number").columns:
        series = df[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        series = series.dropna()
        if len(series) == 0:
            continue

        col_issues = {"column": str(col), "issues": []}

        # Negative values in columns that might be counts/amounts
        neg_count = int((series < 0).sum())
        if neg_count > 0:
            col_issues["issues"].append({
                "type": "negative_values",
                "count": neg_count,
                "percentage": round(neg_count / len(series) * 100, 2),
                "min_value": float(series.min()),
                "message": f"{neg_count:,} negative value(s) (min: {series.min()})",
            })

        # Zero values - check if unusually high proportion
        zero_count = int((series == 0).sum())
        if zero_count > 0 and zero_count / len(series) > 0.5:
            col_issues["issues"].append({
                "type": "excessive_zeros",
                "count": zero_count,
                "percentage": round(zero_count / len(series) * 100, 2),
                "message": f"{zero_count:,} zero values ({zero_count / len(series) * 100:.1f}% of column)",
            })

        # Extreme outliers using IQR method
        if len(series) > 3:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = series[(series < lower_bound) | (series > upper_bound)]
                outlier_count = len(outliers)
                # Use a more permissive threshold for small datasets
                max_outlier_ratio = 0.25 if len(series) < 20 else 0.1
                if outlier_count > 0 and outlier_count / len(series) < max_outlier_ratio:
                    col_issues["issues"].append({
                        "type": "outliers_iqr",
                        "count": outlier_count,
                        "percentage": round(outlier_count / len(series) * 100, 2),
                        "lower_bound": float(lower_bound),
                        "upper_bound": float(upper_bound),
                        "message": f"{outlier_count:,} potential outlier(s) by IQR method (bounds: {lower_bound:.2f}–{upper_bound:.2f})",
                    })

        if col_issues["issues"]:
            issues.append(col_issues)

    return issues


def build_quality_warnings(df: pd.DataFrame) -> list[dict]:
    warnings: list[dict] = []
    if df.empty:
        warnings.append({"level": "error", "message": "The dataset has no data rows."})
    if len(df.columns) == 0:
        warnings.append({"level": "error", "message": "The dataset has no columns."})
    if df.columns.duplicated().any():
        warnings.append({"level": "warning", "message": "Duplicate column names were detected."})
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        warnings.append({
            "level": "warning",
            "message": f"{duplicate_count:,} exact duplicate row(s) were detected.",
        })
    # Handle duplicate column names: use iloc to check each column position
    empty_cols = []
    for i, col_name in enumerate(df.columns):
        col_series = df.iloc[:, i]
        if col_series.isna().all():
            empty_cols.append(str(col_name))
    if empty_cols:
        shown = ", ".join(empty_cols[:8])
        suffix = "…" if len(empty_cols) > 8 else ""
        warnings.append({
            "level": "warning",
            "message": f"{len(empty_cols)} entirely empty column(s): {shown}{suffix}",
        })
    # Handle duplicate column names for missing values check
    missing_cols = []
    for i, col_name in enumerate(df.columns):
        col_series = df.iloc[:, i]
        if col_series.isna().any():
            missing_cols.append(str(col_name))
    if missing_cols:
        warnings.append({
            "level": "info",
            "message": f"{len(missing_cols)} column(s) contain missing values. Review the missing-value table before cleaning.",
        })
    for col in df.select_dtypes(include="number").columns:
        # Handle duplicate column names - get first occurrence
        col_series = df[col]
        if isinstance(col_series, pd.DataFrame):
            col_series = col_series.iloc[:, 0]
        negative_count = int((col_series < 0).sum())
        if negative_count:
            warnings.append({
                "level": "info",
                "message": f"Numeric column '{col}' contains {negative_count:,} negative value(s). Confirm whether these are valid for this business context.",
            })
    if len(df) > 0 and len(df.columns) == 1:
        warnings.append({
            "level": "info",
            "message": "Only one column was detected. If this is a CSV, verify that the delimiter was parsed correctly.",
        })
    return warnings