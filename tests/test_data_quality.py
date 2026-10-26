import pandas as pd
import numpy as np
from modules.data_quality import (
    profile_dataset,
    build_quality_warnings,
    detect_date_columns,
    detect_text_inconsistencies,
    detect_numeric_issues,
    try_parse_dates,
)
from modules.cleaning import create_cleaning_preview


def test_profile_counts():
    df = pd.DataFrame({"name": [" A ", "B", "B"], "amount": [1, None, 2]})
    profile = profile_dataset(df)
    assert profile["rows"] == 3
    assert profile["columns"] == 2
    assert profile["missing_cells"] == 1
    assert profile["duplicate_rows"] == 0


def test_cleaning_does_not_mutate_original():
    df = pd.DataFrame({"name": [" A ", "A"], "value": [1, 1]})
    cleaned, changes = create_cleaning_preview(
        df, trim_strings=True, drop_duplicates=True
    )
    assert df.iloc[0, 0] == " A "
    assert len(cleaned) == 1
    assert changes


def test_quality_warnings_for_missing_and_duplicates():
    df = pd.DataFrame({"x": [1, 1, None]})
    messages = [item["message"] for item in build_quality_warnings(df)]
    assert any("duplicate" in message.lower() for message in messages)
    assert any("missing" in message.lower() for message in messages)


def test_profile_memory_usage():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    profile = profile_dataset(df)
    assert "total_memory_mb" in profile
    assert profile["total_memory_mb"] >= 0
    assert "memory_by_column" in profile
    assert len(profile["memory_by_column"]) == 2


def test_profile_dtype_info():
    df = pd.DataFrame({"int_col": [1, 2], "float_col": [1.0, 2.0], "str_col": ["a", "b"]})
    profile = profile_dataset(df)
    assert "dtype_info" in profile
    assert len(profile["dtype_info"]) == 3
    assert set(profile["dtype_info"]["Column"]) == {"int_col", "float_col", "str_col"}


def test_detect_date_columns_iso():
    df = pd.DataFrame({"date_col": ["2024-01-15", "2024-02-20", "2024-03-10"]})
    candidates = detect_date_columns(df)
    assert len(candidates) == 1
    assert candidates[0]["column"] == "date_col"
    assert "ISO" in candidates[0]["suspected_format"]
    assert candidates[0]["parse_success_rate"] == 100.0


def test_detect_date_columns_mixed_formats():
    df = pd.DataFrame({"date_col": ["2024-01-15", "2024-02-20", "2024-03-10", "2024-04-05", "not-a-date"]})
    candidates = detect_date_columns(df)
    # Should still detect but with lower success rate
    assert len(candidates) == 1
    assert candidates[0]["parse_success_rate"] < 100.0


def test_detect_date_columns_no_match():
    df = pd.DataFrame({"text_col": ["hello", "world", "foo"]})
    candidates = detect_date_columns(df)
    assert len(candidates) == 0


def test_detect_text_inconsistencies_case_whitespace():
    df = pd.DataFrame({"category": ["A", " a ", "A", "B", "b"]})
    issues = detect_text_inconsistencies(df)
    assert len(issues) == 1
    assert issues[0]["column"] == "category"
    assert issues[0]["affected_unique_values"] >= 2
    # Should have fuzzy groups for "a" and "b"
    assert "a" in issues[0]["fuzzy_groups"]
    assert "b" in issues[0]["fuzzy_groups"]


def test_detect_text_inconsistencies_high_cardinality_skipped():
    df = pd.DataFrame({"id": [f"id_{i}" for i in range(100)]})
    issues = detect_text_inconsistencies(df, max_cardinality=50)
    assert len(issues) == 0


def test_detect_numeric_issues_negatives():
    df = pd.DataFrame({"amount": [10, -5, 20, -1, 30]})
    issues = detect_numeric_issues(df)
    assert len(issues) == 1
    assert issues[0]["column"] == "amount"
    neg_issue = next(i for i in issues[0]["issues"] if i["type"] == "negative_values")
    assert neg_issue["count"] == 2


def test_detect_numeric_issues_excessive_zeros():
    df = pd.DataFrame({"count": [0, 0, 0, 0, 1, 2]})
    issues = detect_numeric_issues(df)
    assert len(issues) == 1
    zero_issue = next(i for i in issues[0]["issues"] if i["type"] == "excessive_zeros")
    assert zero_issue["count"] == 4


def test_detect_numeric_issues_outliers():
    # Clear outliers: 1000 and -500 in a series centered around 10
    # Need at least 9 values for 2 outliers to be < 25%
    df = pd.DataFrame({"value": [10, 11, 9, 10, 12, 10, 11, 1000, -500]})
    issues = detect_numeric_issues(df)
    assert len(issues) == 1
    outlier_issue = next(i for i in issues[0]["issues"] if i["type"] == "outliers_iqr")
    assert outlier_issue["count"] >= 2


def test_empty_dataset_warnings():
    df = pd.DataFrame()
    warnings = build_quality_warnings(df)
    assert any(w["level"] == "error" and "no data rows" in w["message"].lower() for w in warnings)


def test_empty_dataset_profile():
    df = pd.DataFrame()
    profile = profile_dataset(df)
    assert profile["rows"] == 0
    assert profile["columns"] == 0
    assert profile["missing_cells"] == 0
    assert profile["duplicate_rows"] == 0


def test_single_column_warning():
    df = pd.DataFrame({"only_col": [1, 2, 3]})
    warnings = build_quality_warnings(df)
    assert any("only one column" in w["message"].lower() for w in warnings)


def test_duplicate_column_names_warning():
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "A"])
    warnings = build_quality_warnings(df)
    assert any("duplicate column" in w["message"].lower() for w in warnings)


def test_mixed_data_types_profile():
    df = pd.DataFrame({
        "int_col": [1, 2, None],
        "float_col": [1.5, None, 2.5],
        "str_col": ["a", "b", "c"],
        "bool_col": [True, False, True],
    })
    profile = profile_dataset(df)
    assert profile["rows"] == 3
    assert profile["columns"] == 4
    assert profile["missing_cells"] == 2
    # Check dtype info captures all types
    cols = set(profile["dtype_info"]["Column"])
    assert cols == {"int_col", "float_col", "str_col", "bool_col"}


def test_unusual_numeric_values_large_numbers():
    # Use values that clearly show outliers with IQR
    # Need at least 9 values for 2 outliers to be < 25%
    df = pd.DataFrame({"big_nums": [100, 105, 95, 102, 98, 100, 101, 10000, -5000]})
    issues = detect_numeric_issues(df)
    # Should detect outliers
    assert len(issues) == 1
    outlier_issue = next(i for i in issues[0]["issues"] if i["type"] == "outliers_iqr")
    assert outlier_issue["count"] >= 2


def test_cleaning_trim_strings():
    df = pd.DataFrame({"name": ["  Alice  ", "Bob ", "  Carol"]})
    cleaned, changes = create_cleaning_preview(df, trim_strings=True)
    assert cleaned["name"].tolist() == ["Alice", "Bob", "Carol"]
    assert any("Trimmed" in c for c in changes)


def test_cleaning_normalize_spaces():
    df = pd.DataFrame({"name": ["Alice   Smith", "Bob\tJones", "Carol\nWilliams"]})
    cleaned, changes = create_cleaning_preview(df, normalize_spaces=True)
    assert all("  " not in v for v in cleaned["name"])
    assert any("Normalized" in c for c in changes)


def test_cleaning_drop_duplicates():
    df = pd.DataFrame({"x": [1, 1, 2], "y": ["a", "a", "b"]})
    cleaned, changes = create_cleaning_preview(df, drop_duplicates=True)
    assert len(cleaned) == 2
    assert any("Removed" in c and "duplicate" in c for c in changes)


def test_cleaning_drop_empty_columns():
    df = pd.DataFrame({"a": [1, 2], "b": [None, None], "c": [3, 4]})
    cleaned, changes = create_cleaning_preview(df, drop_empty_columns=True)
    assert "b" not in cleaned.columns
    assert any("Removed" in c and "empty column" in c for c in changes)


def test_try_parse_dates_success():
    series = pd.Series(["2024-01-01", "2024-02-02", "2024-03-03"])
    result = try_parse_dates(series, "%Y-%m-%d")
    assert result["success_rate"] == 100.0
    assert result["errors"] == []


def test_try_parse_dates_partial_failure():
    series = pd.Series(["2024-01-01", "not-a-date", "2024-03-03"])
    result = try_parse_dates(series, r"^\d{4}-\d{2}-\d{2}$")
    assert result["success_rate"] < 100.0
    assert "not-a-date" in result["errors"]


def test_profile_empty_columns_detected():
    df = pd.DataFrame({"a": [1, 2], "b": [None, None], "c": [3, 4]})
    profile = profile_dataset(df)
    assert "b" in profile["empty_columns"]


def test_duplicate_rows_count():
    df = pd.DataFrame({"x": [1, 1, 2, 2], "y": ["a", "a", "b", "b"]})
    profile = profile_dataset(df)
    # All rows are duplicates of previous rows, so 2 duplicate rows
    assert profile["duplicate_rows"] == 2


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))