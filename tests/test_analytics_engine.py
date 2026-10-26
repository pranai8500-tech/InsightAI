from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from unittest.mock import Mock, patch

from modules.analytics_engine import (
    AnalyticsEngine,
    AnalysisPlan,
    AnalysisResult,
    OperationType,
    FilterCondition,
    FilterOperator,
    GroupBySpec,
    AggregateSpec,
    AggregationFunction,
    SortSpec,
    DateAnalysisSpec,
    MoMComparisonSpec,
    OutlierDetectionSpec,
    create_engine,
)
from modules.analysis_planner import (
    parse_analysis_plan,
    build_analysis_system_prompt,
    sanitize_for_ai_context,
    _fix_common_json_issues,
    _normalize_aggregation_synonyms,
)


class TestAnalyticsEngine:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "category": ["A", "B", "A", "B", "A"],
            "value": [10.0, 20.0, 30.0, 40.0, 50.0],
            "date": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15", "2024-03-01"]),
            "nullable": [1, None, 3, 4, None],
        })

    def test_engine_creation(self, sample_df):
        engine = create_engine(sample_df)
        assert isinstance(engine, AnalyticsEngine)

    def test_empty_df_raises(self):
        with pytest.raises(ValueError, match="empty"):
            create_engine(pd.DataFrame())

    def test_no_columns_raises(self):
        with pytest.raises(ValueError, match="empty"):
            create_engine(pd.DataFrame(index=[0, 1]))

    def test_row_count(self, sample_df):
        engine = create_engine(sample_df)
        plan = AnalysisPlan(operations=[OperationType.ROW_COUNT])
        result = engine.execute(plan)
        assert result.success
        assert result.data["row_count"] == 5

    def test_missing_summary(self, sample_df):
        engine = create_engine(sample_df)
        plan = AnalysisPlan(operations=[OperationType.MISSING_SUMMARY])
        result = engine.execute(plan)
        assert result.success
        missing_df = result.data["missing_summary"]
        assert len(missing_df) == 5
        nullable_row = missing_df[missing_df["column"] == "nullable"].iloc[0]
        assert nullable_row["missing_count"] == 2
        assert nullable_row["missing_percentage"] == 40.0

    def test_column_stats_numeric(self, sample_df):
        engine = create_engine(sample_df)
        plan = AnalysisPlan(operations=[OperationType.COLUMN_STATS], column_stats=["value"])
        result = engine.execute(plan)
        assert result.success
        stats_df = result.data["column_stats"]
        assert len(stats_df) == 1
        row = stats_df.iloc[0]
        assert row["column"] == "value"
        assert row["mean"] == 30.0
        assert row["min"] == 10.0
        assert row["max"] == 50.0

    def test_column_stats_invalid_column(self, sample_df):
        engine = create_engine(sample_df)
        plan = AnalysisPlan(operations=[OperationType.COLUMN_STATS], column_stats=["nonexistent"])
        result = engine.execute(plan)
        assert not result.success
        assert "not found" in result.error.lower()

    def test_value_counts(self, sample_df):
        engine = create_engine(sample_df)
        plan = AnalysisPlan(operations=[OperationType.VALUE_COUNTS], value_counts=["category"])
        result = engine.execute(plan)
        assert result.success
        vc = result.data["value_counts"]["category"]
        assert len(vc) == 2
        assert set(vc["value"]) == {"A", "B"}

    def test_filter_equals(self, sample_df):
        engine = create_engine(sample_df)
        plan = AnalysisPlan(
            operations=[OperationType.ROW_COUNT],
            filters=[FilterCondition(column="category", operator=FilterOperator.EQ, value="A")],
        )
        result = engine.execute(plan)
        assert result.success
        assert result.data["row_count"] == 3

    def test_filter_numeric_gt(self, sample_df):
        engine = create_engine(sample_df)
        plan = AnalysisPlan(
            operations=[OperationType.ROW_COUNT],
            filters=[FilterCondition(column="value", operator=FilterOperator.GT, value=25)],
        )
        result = engine.execute(plan)
        assert result.success
        assert result.data["row_count"] == 3

    def test_filter_invalid_column(self, sample_df):
        engine = create_engine(sample_df)
        plan = AnalysisPlan(
            operations=[OperationType.ROW_COUNT],
            filters=[FilterCondition(column="nonexistent", operator=FilterOperator.EQ, value="A")],
        )
        result = engine.execute(plan)
        assert not result.success

    def test_filter_contains_string(self, sample_df):
        engine = create_engine(sample_df)
        plan = AnalysisPlan(
            operations=[OperationType.ROW_COUNT],
            filters=[FilterCondition(column="category", operator=FilterOperator.CONTAINS, value="A")],
        )
        result = engine.execute(plan)
        assert result.success
        assert result.data["row_count"] == 3

    def test_group_by_with_aggregation(self, sample_df):
        engine = create_engine(sample_df)
        plan = AnalysisPlan(
            operations=[OperationType.GROUP_BY],
            group_by=GroupBySpec(
                columns=["category"],
                aggregations={"value": [AggregationFunction.SUM, AggregationFunction.MEAN]},
            ),
        )
        result = engine.execute(plan)
        assert result.success
        gb_df = result.data["group_by"]
        assert len(gb_df) == 2
        assert "value_sum" in gb_df.columns
        assert "value_mean" in gb_df.columns

    def test_aggregate_without_group_by(self, sample_df):
        engine = create_engine(sample_df)
        plan = AnalysisPlan(
            operations=[OperationType.AGGREGATE],
            aggregate=AggregateSpec(columns=["value"], functions=[AggregationFunction.SUM, AggregationFunction.MEAN]),
        )
        result = engine.execute(plan)
        assert result.success
        agg_df = result.data["aggregate"]
        # pandas agg returns one row per function
        assert len(agg_df) == 2
        assert "index" in agg_df.columns or "value" in agg_df.columns

    def test_aggregate_without_group_by_column_structure(self, sample_df):
        """Regression test: verify AGGREGATE without group_by column structure.
        
        The result has columns: "index" (function names) and original column names.
        This is different from GROUP BY which produces {col}_{func} column names.
        """
        engine = create_engine(sample_df)
        plan = AnalysisPlan(
            operations=[OperationType.AGGREGATE],
            aggregate=AggregateSpec(columns=["value"], functions=[AggregationFunction.MEAN]),
        )
        result = engine.execute(plan)
        assert result.success
        agg_df = result.data["aggregate"]
        
        # Verify column structure
        assert "index" in agg_df.columns
        assert "value" in agg_df.columns
        assert "value_mean" not in agg_df.columns  # This is the key difference from GROUP BY
        
        # Verify we can extract the mean value correctly
        mean_row = agg_df[agg_df["index"] == "mean"]
        assert not mean_row.empty
        mean_val = mean_row["value"].iloc[0]
        assert mean_val == 30.0  # (10+20+30+40+50)/5

    def test_aggregate_with_group_by(self, sample_df):
        engine = create_engine(sample_df)
        plan = AnalysisPlan(
            operations=[OperationType.AGGREGATE],
            aggregate=AggregateSpec(
                columns=["value"],
                functions=[AggregationFunction.SUM],
                group_by=["category"],
            ),
        )
        result = engine.execute(plan)
        assert result.success
        agg_df = result.data["aggregate"]
        assert len(agg_df) == 2

    def test_sort(self, sample_df):
        engine = create_engine(sample_df)
        plan = AnalysisPlan(
            operations=[OperationType.SORT, OperationType.ROW_COUNT],
            sort=SortSpec(columns=["value"], ascending=[False]),
        )
        result = engine.execute(plan)
        assert result.success

    def test_group_by_mean_with_sort_on_generated_column(self, sample_df):
        """Regression test: group_by with mean, then sort by generated column (e.g., Sales_mean)."""
        engine = create_engine(sample_df)
        plan = AnalysisPlan(
            operations=[OperationType.GROUP_BY, OperationType.SORT],
            group_by=GroupBySpec(
                columns=["category"],
                aggregations={"value": [AggregationFunction.MEAN]},
            ),
            sort=SortSpec(columns=["value_mean"], ascending=[False]),
        )
        result = engine.execute(plan)
        assert result.success, f"Failed: {result.error}"
        gb_df = result.data["group_by"]
        assert len(gb_df) == 2
        assert "value_mean" in gb_df.columns
        # Verify sorted descending by value_mean
        means = gb_df["value_mean"].tolist()
        assert means == sorted(means, reverse=True)

    def test_aggregate_with_group_by_and_sort(self, sample_df):
        """Regression test: aggregate with group_by, then sort by generated column."""
        engine = create_engine(sample_df)
        plan = AnalysisPlan(
            operations=[OperationType.AGGREGATE, OperationType.SORT],
            aggregate=AggregateSpec(
                columns=["value"],
                functions=[AggregationFunction.MEAN],
                group_by=["category"],
            ),
            sort=SortSpec(columns=["value_mean"], ascending=[False]),
        )
        result = engine.execute(plan)
        assert result.success, f"Failed: {result.error}"
        agg_df = result.data["aggregate"]
        assert len(agg_df) == 2
        assert "value_mean" in agg_df.columns
        means = agg_df["value_mean"].tolist()
        assert means == sorted(means, reverse=True)

    def test_sort_invalid_column_on_grouped_result(self, sample_df):
        """Sort should fail for invalid column names even after group_by."""
        engine = create_engine(sample_df)
        plan = AnalysisPlan(
            operations=[OperationType.GROUP_BY, OperationType.SORT],
            group_by=GroupBySpec(
                columns=["category"],
                aggregations={"value": [AggregationFunction.MEAN]},
            ),
            sort=SortSpec(columns=["nonexistent_column"], ascending=[False]),
        )
        result = engine.execute(plan)
        assert not result.success
        assert "not found" in result.error.lower()

    def test_date_analysis(self, sample_df):
        engine = create_engine(sample_df)
        plan = AnalysisPlan(
            operations=[OperationType.DATE_ANALYSIS],
            date_analysis=DateAnalysisSpec(
                date_column="date",
                value_column="value",
                frequency="M",
                aggregation=AggregationFunction.SUM,
            ),
        )
        result = engine.execute(plan)
        assert result.success
        da_df = result.data["date_analysis"]
        assert len(da_df) == 3
        assert "period" in da_df.columns
        assert "value_sum" in da_df.columns

    def test_correlation(self, sample_df):
        engine = create_engine(sample_df)
        plan = AnalysisPlan(
            operations=[OperationType.CORRELATION],
            correlation_columns=["value", "nullable"],
        )
        result = engine.execute(plan)
        assert result.success
        corr_df = result.data["correlation"]
        assert "index" in corr_df.columns or "value" in corr_df.columns

    def test_correlation_needs_two_numeric(self, sample_df):
        engine = create_engine(sample_df)
        plan = AnalysisPlan(
            operations=[OperationType.CORRELATION],
            correlation_columns=["category"],
        )
        result = engine.execute(plan)
        assert not result.success
        assert "at least 2 numeric" in result.error.lower()

    def test_limit_applied(self, sample_df):
        engine = create_engine(sample_df)
        plan = AnalysisPlan(
            operations=[OperationType.VALUE_COUNTS],
            value_counts=["category"],
            limit=1,
        )
        result = engine.execute(plan)
        assert result.success
        vc = result.data["value_counts"]["category"]
        assert len(vc) == 1

    def test_empty_result_after_filter(self, sample_df):
        engine = create_engine(sample_df)
        plan = AnalysisPlan(
            operations=[OperationType.ROW_COUNT],
            filters=[FilterCondition(column="category", operator=FilterOperator.EQ, value="Z")],
        )
        result = engine.execute(plan)
        assert result.success
        assert result.data["row_count"] == 0
        assert result.row_count == 0

    def test_group_by_result_contains_calculated_values(self, sample_df):
        """Regression test: grouped result includes actual calculated aggregation values."""
        engine = create_engine(sample_df)
        plan = AnalysisPlan(
            operations=[OperationType.GROUP_BY],
            group_by=GroupBySpec(
                columns=["category"],
                aggregations={"value": [AggregationFunction.MEAN, AggregationFunction.SUM]},
            ),
        )
        result = engine.execute(plan)
        assert result.success, f"Failed: {result.error}"
        gb_df = result.data["group_by"]
        assert len(gb_df) == 2
        # Verify generated columns exist with actual calculated values
        assert "value_mean" in gb_df.columns
        assert "value_sum" in gb_df.columns
        # Verify values are calculated correctly (A: 10, 30, 50 -> mean 30, sum 90; B: 20, 40 -> mean 30, sum 60)
        a_row = gb_df[gb_df["category"] == "A"].iloc[0]
        b_row = gb_df[gb_df["category"] == "B"].iloc[0]
        assert a_row["value_mean"] == 30.0
        assert a_row["value_sum"] == 90.0
        assert b_row["value_mean"] == 30.0
        assert b_row["value_sum"] == 60.0

    def test_date_analysis_with_string_dates(self):
        """Regression test: date_analysis parses string dates in ISO format."""
        df = pd.DataFrame({
            "date_str": ["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15", "2024-03-01"],
            "sales": [100.0, 200.0, 150.0, 250.0, 300.0],
        })
        engine = create_engine(df)
        plan = AnalysisPlan(
            operations=[OperationType.DATE_ANALYSIS],
            date_analysis=DateAnalysisSpec(
                date_column="date_str",
                value_column="sales",
                frequency="M",
                aggregation=AggregationFunction.SUM,
            ),
        )
        result = engine.execute(plan)
        assert result.success, f"Failed: {result.error}"
        da_df = result.data["date_analysis"]
        assert len(da_df) == 3  # 3 months
        assert "period" in da_df.columns
        assert "sales_sum" in da_df.columns
        # Verify chronological order
        periods = da_df["period"].tolist()
        assert periods == sorted(periods)
        # Verify calculated values
        jan_row = da_df[da_df["period"] == "2024-01"].iloc[0]
        assert jan_row["sales_sum"] == 300.0  # 100 + 200

    def test_date_analysis_with_us_format_dates(self):
        """Regression test: date_analysis parses US format dates (MM/DD/YYYY)."""
        df = pd.DataFrame({
            "date_str": ["01/15/2024", "01/30/2024", "02/15/2024", "03/01/2024"],
            "sales": [100.0, 200.0, 150.0, 300.0],
        })
        engine = create_engine(df)
        plan = AnalysisPlan(
            operations=[OperationType.DATE_ANALYSIS],
            date_analysis=DateAnalysisSpec(
                date_column="date_str",
                value_column="sales",
                frequency="M",
                aggregation=AggregationFunction.SUM,
            ),
        )
        result = engine.execute(plan)
        assert result.success, f"Failed: {result.error}"
        da_df = result.data["date_analysis"]
        assert len(da_df) == 3
        assert "sales_sum" in da_df.columns

    def test_date_analysis_with_invalid_dates(self):
        """Regression test: date_analysis handles invalid/missing dates gracefully."""
        df = pd.DataFrame({
            "date_str": ["2024-01-01", "invalid", "2024-02-01", None, "2024-03-01"],
            "sales": [100.0, 200.0, 150.0, 250.0, 300.0],
        })
        engine = create_engine(df)
        plan = AnalysisPlan(
            operations=[OperationType.DATE_ANALYSIS],
            date_analysis=DateAnalysisSpec(
                date_column="date_str",
                value_column="sales",
                frequency="M",
                aggregation=AggregationFunction.SUM,
            ),
        )
        result = engine.execute(plan)
        assert result.success, f"Failed: {result.error}"
        da_df = result.data["date_analysis"]
        # Should have 3 valid months (invalid and None excluded)
        assert len(da_df) == 3
        assert "sales_sum" in da_df.columns

    def test_date_analysis_all_invalid_dates_fails(self):
        """Regression test: date_analysis fails when no valid dates found."""
        df = pd.DataFrame({
            "date_str": ["invalid", "also invalid", "not a date"],
            "sales": [100.0, 200.0, 150.0],
        })
        engine = create_engine(df)
        plan = AnalysisPlan(
            operations=[OperationType.DATE_ANALYSIS],
            date_analysis=DateAnalysisSpec(
                date_column="date_str",
                value_column="sales",
                frequency="M",
                aggregation=AggregationFunction.SUM,
            ),
        )
        result = engine.execute(plan)
        assert not result.success
        assert "could not be parsed" in result.error.lower()

    def test_date_analysis_chronological_sort(self):
        """Regression test: date_analysis returns months in chronological order."""
        df = pd.DataFrame({
            "date_str": ["2024-03-01", "2024-01-01", "2024-02-01"],  # Out of order
            "sales": [300.0, 100.0, 200.0],
        })
        engine = create_engine(df)
        plan = AnalysisPlan(
            operations=[OperationType.DATE_ANALYSIS],
            date_analysis=DateAnalysisSpec(
                date_column="date_str",
                value_column="sales",
                frequency="M",
                aggregation=AggregationFunction.SUM,
            ),
        )
        result = engine.execute(plan)
        assert result.success, f"Failed: {result.error}"
        da_df = result.data["date_analysis"]
        periods = da_df["period"].tolist()
        assert periods == ["2024-01", "2024-02", "2024-03"]  # Chronological

    def test_date_analysis_with_existing_datetime(self, sample_df):
        """Regression test: date_analysis works with already-parsed datetime columns."""
        engine = create_engine(sample_df)
        plan = AnalysisPlan(
            operations=[OperationType.DATE_ANALYSIS],
            date_analysis=DateAnalysisSpec(
                date_column="date",
                value_column="value",
                frequency="M",
                aggregation=AggregationFunction.SUM,
            ),
        )
        result = engine.execute(plan)
        assert result.success, f"Failed: {result.error}"
        da_df = result.data["date_analysis"]
        assert len(da_df) == 3
        assert "value_sum" in da_df.columns

    def test_date_analysis_with_logical_column_name(self):
        """Regression test: date_analysis resolves logical column names to actual column names.
        
        This tests the fix for the 'Monthly sales trends' bug where the AI planner
        uses logical names like 'order_date' instead of actual names like 'order date (DateOrders)'.
        """
        df = pd.DataFrame({
            "order date (DateOrders)": ["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15", "2024-03-01"],
            "Sales": [100.0, 200.0, 150.0, 250.0, 300.0],
        })
        engine = create_engine(df)
        plan = AnalysisPlan(
            operations=[OperationType.DATE_ANALYSIS],
            date_analysis=DateAnalysisSpec(
                date_column="order_date",  # Logical name
                value_column="Sales",
                frequency="M",
                aggregation=AggregationFunction.SUM,
            ),
        )
        result = engine.execute(plan)
        assert result.success, f"Failed: {result.error}"
        da_df = result.data["date_analysis"]
        assert len(da_df) == 3
        assert "Sales_sum" in da_df.columns
        # Verify calculated values
        jan_row = da_df[da_df["period"] == "2024-01"].iloc[0]
        assert jan_row["Sales_sum"] == 300.0  # 100 + 200

    def test_group_by_with_logical_column_names(self):
        """Regression test: group_by resolves logical column names to actual column names.
        
        This tests the fix for the 'Average late-delivery risk by market' bug where the AI planner
        uses logical names like 'late_delivery_risk' instead of actual names like 'Late_delivery_risk'.
        """
        df = pd.DataFrame({
            "Market": ["A", "B", "A", "B", "A"],
            "Late_delivery_risk": [1, 0, 1, 0, 1],
        })
        engine = create_engine(df)
        plan = AnalysisPlan(
            operations=[OperationType.GROUP_BY],
            group_by=GroupBySpec(
                columns=["Market"],
                aggregations={"late_delivery_risk": [AggregationFunction.MEAN]},  # Logical name
            ),
        )
        result = engine.execute(plan)
        assert result.success, f"Failed: {result.error}"
        gb_df = result.data["group_by"]
        assert len(gb_df) == 2
        assert "Late_delivery_risk_mean" in gb_df.columns
        # Verify calculated values: A has 1,1 -> mean 1.0; B has 0,0 -> mean 0.0
        a_row = gb_df[gb_df["Market"] == "A"].iloc[0]
        b_row = gb_df[gb_df["Market"] == "B"].iloc[0]
        assert a_row["Late_delivery_risk_mean"] == 1.0
        assert b_row["Late_delivery_risk_mean"] == 0.0

    def test_date_analysis_missing_column_error(self):
        """Regression test: date_analysis gives clear error when date column is missing.
        
        The error should list available columns to help diagnose the issue.
        """
        df = pd.DataFrame({
            "some_other_date": ["2024-01-01", "2024-01-15"],
            "Sales": [100.0, 200.0],
        })
        engine = create_engine(df)
        plan = AnalysisPlan(
            operations=[OperationType.DATE_ANALYSIS],
            date_analysis=DateAnalysisSpec(
                date_column="order date (DateOrders)",  # Column that doesn't exist
                value_column="Sales",
                frequency="M",
                aggregation=AggregationFunction.SUM,
            ),
        )
        result = engine.execute(plan)
        assert not result.success
        assert "order date (DateOrders)" in result.error
        assert "not found in working DataFrame" in result.error
        assert "Available columns" in result.error
        assert "some_other_date" in result.error
        assert "Sales" in result.error

    def test_date_analysis_after_group_by_uses_original_data(self):
        """Test that DATE_ANALYSIS uses original filtered data even after GROUP_BY.
        
        Previously, GROUP_BY would transform result_df and DATE_ANALYSIS would fail
        because the date column was no longer in the transformed DataFrame.
        Now, DATE_ANALYSIS uses the original filtered data, so it works correctly.
        """
        df = pd.DataFrame({
            "order date (DateOrders)": ["2024-01-01", "2024-01-15", "2024-02-01"],
            "Sales": [100.0, 200.0, 150.0],
            "Market": ["A", "B", "A"],
        })
        engine = create_engine(df)
        plan = AnalysisPlan(
            operations=[OperationType.GROUP_BY, OperationType.DATE_ANALYSIS],
            group_by=GroupBySpec(
                columns=["Market"],
                aggregations={"Sales": [AggregationFunction.SUM]},
            ),
            date_analysis=DateAnalysisSpec(
                date_column="order date (DateOrders)",
                value_column="Sales",
                frequency="M",
                aggregation=AggregationFunction.SUM,
            ),
        )
        result = engine.execute(plan)
        assert result.success, f"Expected success but got: {result.error}"
        # Should have both group_by and date_analysis results
        assert "group_by" in result.data
        assert "date_analysis" in result.data
        # date_analysis should have monthly aggregated data from original dates
        date_analysis = result.data["date_analysis"]
        assert len(date_analysis) == 2  # Jan and Feb
        assert "period" in date_analysis.columns
        assert "Sales_sum" in date_analysis.columns


    def test_date_analysis_missing_column_still_fails(self):
        """Test that DATE_ANALYSIS still fails when column is truly missing from original data."""
        df = pd.DataFrame({
            "order date (DateOrders)": ["2024-01-01", "2024-01-15", "2024-02-01"],
            "Sales": [100.0, 200.0, 150.0],
            "Market": ["A", "B", "A"],
        })
        engine = create_engine(df)
        plan = AnalysisPlan(
            operations=[OperationType.GROUP_BY, OperationType.DATE_ANALYSIS],
            group_by=GroupBySpec(
                columns=["Market"],
                aggregations={"Sales": [AggregationFunction.SUM]},
            ),
            date_analysis=DateAnalysisSpec(
                date_column="nonexistent_date",  # This column doesn't exist in original data
                value_column="Sales",
                frequency="M",
                aggregation=AggregationFunction.SUM,
            ),
        )
        result = engine.execute(plan)
        assert not result.success
        assert "nonexistent_date" in result.error
        assert "not found" in result.error.lower()


    # --- MoM Comparison Tests ---
    def test_mom_comparison_monthly(self):
        """Test month-over-month comparison with monthly frequency."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15", "2024-03-01"]),
            "value": [10.0, 30.0, 20.0, 40.0, 50.0],
            "category": ["A", "B", "A", "B", "A"],
        })
        engine = create_engine(df)
        plan = AnalysisPlan(
            operations=[OperationType.MOM_COMPARISON],
            mom_comparison=MoMComparisonSpec(
                date_column="date",
                value_column="value",
                frequency="M",
                aggregation=AggregationFunction.SUM,
                periods=3,
            ),
        )
        result = engine.execute(plan)
        assert result.success, f"Failed: {result.error}"
        mom_df = result.data["mom_comparison"]
        assert len(mom_df) == 3
        assert "period" in mom_df.columns
        assert "value_sum" in mom_df.columns
        assert "value_sum_mom_pct_change" in mom_df.columns
        assert "value_sum_mom_abs_change" in mom_df.columns
        # First period should have NaN for MoM change
        assert pd.isna(mom_df.iloc[0]["value_sum_mom_pct_change"])
        # Verify calculated values
        # Jan: 10+30=40, Feb: 20+40=60, Mar: 50
        jan_row = mom_df[mom_df["period"] == "2024-01"].iloc[0]
        feb_row = mom_df[mom_df["period"] == "2024-02"].iloc[0]
        mar_row = mom_df[mom_df["period"] == "2024-03"].iloc[0]
        assert jan_row["value_sum"] == 40.0
        assert feb_row["value_sum"] == 60.0
        assert mar_row["value_sum"] == 50.0
        # Feb MoM change: (60-40)/40 * 100 = 50%
        assert abs(feb_row["value_sum_mom_pct_change"] - 50.0) < 0.01
        # Mar MoM change: (50-60)/60 * 100 = -16.67%
        assert abs(mar_row["value_sum_mom_pct_change"] - (-16.666666666666664)) < 0.01

    def test_mom_comparison_quarterly(self):
        """Test quarter-over-quarter comparison."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01", "2024-06-01"]),
            "sales": [100.0, 150.0, 200.0, 250.0, 300.0, 350.0],
        })
        engine = create_engine(df)
        plan = AnalysisPlan(
            operations=[OperationType.MOM_COMPARISON],
            mom_comparison=MoMComparisonSpec(
                date_column="date",
                value_column="sales",
                frequency="Q",
                aggregation=AggregationFunction.SUM,
                periods=2,
            ),
        )
        result = engine.execute(plan)
        assert result.success, f"Failed: {result.error}"
        mom_df = result.data["mom_comparison"]
        assert len(mom_df) == 2
        assert "period" in mom_df.columns
        assert "sales_sum" in mom_df.columns
        assert "sales_sum_mom_pct_change" in mom_df.columns
        # Q1: 100+150+200=450, Q2: 250+300+350=900
        q1_row = mom_df[mom_df["period"] == "2024Q1"].iloc[0]
        q2_row = mom_df[mom_df["period"] == "2024Q2"].iloc[0]
        assert q1_row["sales_sum"] == 450.0
        assert q2_row["sales_sum"] == 900.0
        # Q2 QoQ change: (900-450)/450 * 100 = 100%
        assert abs(q2_row["sales_sum_mom_pct_change"] - 100.0) < 0.01

    def test_mom_comparison_with_string_dates(self):
        """Test MoM comparison parses string dates."""
        df = pd.DataFrame({
            "date_str": ["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15", "2024-03-01"],
            "sales": [100.0, 200.0, 150.0, 250.0, 300.0],
        })
        engine = create_engine(df)
        plan = AnalysisPlan(
            operations=[OperationType.MOM_COMPARISON],
            mom_comparison=MoMComparisonSpec(
                date_column="date_str",
                value_column="sales",
                frequency="M",
                aggregation=AggregationFunction.SUM,
                periods=3,
            ),
        )
        result = engine.execute(plan)
        assert result.success, f"Failed: {result.error}"
        mom_df = result.data["mom_comparison"]
        assert len(mom_df) == 3
        assert "sales_sum" in mom_df.columns
        assert "sales_sum_mom_pct_change" in mom_df.columns

    def test_mom_comparison_missing_columns_error(self):
        """Test MoM comparison gives clear error for missing columns."""
        df = pd.DataFrame({"date": ["2024-01-01"], "sales": [100.0]})
        engine = create_engine(df)
        plan = AnalysisPlan(
            operations=[OperationType.MOM_COMPARISON],
            mom_comparison=MoMComparisonSpec(
                date_column="nonexistent_date",
                value_column="sales",
                frequency="M",
                aggregation=AggregationFunction.SUM,
            ),
        )
        result = engine.execute(plan)
        assert not result.success
        assert "nonexistent_date" in result.error
        assert "not found" in result.error.lower()

    # --- Outlier Detection Tests ---
    def test_outlier_detection_iqr(self):
        """Test outlier detection using IQR method."""
        df = pd.DataFrame({
            "value": [10.0, 20.0, 30.0, 40.0, 50.0, 1000.0],  # 1000 is outlier
            "category": ["A", "B", "A", "B", "A", "B"],
        })
        engine = create_engine(df)
        plan = AnalysisPlan(
            operations=[OperationType.OUTLIER_DETECTION],
            outlier_detection=OutlierDetectionSpec(
                column="value",
                method="iqr",
                threshold=1.5,
            ),
        )
        result = engine.execute(plan)
        assert result.success, f"Failed: {result.error}"
        outlier_data = result.data["outlier_detection"]
        assert "summary" in outlier_data
        assert "outliers" in outlier_data
        summary = outlier_data["summary"][0]
        assert summary["method"] == "iqr"
        assert summary["threshold"] == 1.5
        assert summary["outlier_count"] == 1
        assert summary["outlier_percentage"] > 0
        # Check outlier details
        outliers = outlier_data["outliers"]
        assert len(outliers) == 1
        assert outliers[0]["value"] == 1000.0

    def test_outlier_detection_zscore(self):
        """Test outlier detection using z-score method."""
        df = pd.DataFrame({
            "value": [10.0, 20.0, 30.0, 40.0, 50.0, 1000.0],
            "category": ["A", "B", "A", "B", "A", "B"],
        })
        engine = create_engine(df)
        plan = AnalysisPlan(
            operations=[OperationType.OUTLIER_DETECTION],
            outlier_detection=OutlierDetectionSpec(
                column="value",
                method="zscore",
                threshold=2.0,
            ),
        )
        result = engine.execute(plan)
        assert result.success, f"Failed: {result.error}"
        outlier_data = result.data["outlier_detection"]
        summary = outlier_data["summary"][0]
        assert summary["method"] == "zscore"
        assert summary["threshold"] == 2.0
        assert "mean" in summary
        assert "std" in summary

    def test_outlier_detection_no_outliers(self):
        """Test outlier detection when no outliers exist."""
        df = pd.DataFrame({
            "value": [10.0, 20.0, 30.0, 40.0, 50.0],
            "category": ["A", "B", "A", "B", "A"],
        })
        engine = create_engine(df)
        plan = AnalysisPlan(
            operations=[OperationType.OUTLIER_DETECTION],
            outlier_detection=OutlierDetectionSpec(
                column="value",
                method="iqr",
                threshold=1.5,
            ),
        )
        result = engine.execute(plan)
        assert result.success, f"Failed: {result.error}"
        outlier_data = result.data["outlier_detection"]
        summary = outlier_data["summary"][0]
        assert summary["outlier_count"] == 0
        assert summary["outlier_percentage"] == 0.0
        assert outlier_data["outliers"] == []

    def test_outlier_detection_zero_std_fails(self):
        """Test outlier detection fails with zero standard deviation (zscore)."""
        df = pd.DataFrame({"value": [10.0, 10.0, 10.0, 10.0]})
        engine = create_engine(df)
        plan = AnalysisPlan(
            operations=[OperationType.OUTLIER_DETECTION],
            outlier_detection=OutlierDetectionSpec(
                column="value",
                method="zscore",
                threshold=2.0,
            ),
        )
        result = engine.execute(plan)
        assert not result.success
        assert "zero standard deviation" in result.error.lower()

    def test_outlier_detection_missing_column_error(self):
        """Test outlier detection gives clear error for missing column."""
        df = pd.DataFrame({"value": [10.0, 20.0, 30.0]})
        engine = create_engine(df)
        plan = AnalysisPlan(
            operations=[OperationType.OUTLIER_DETECTION],
            outlier_detection=OutlierDetectionSpec(
                column="nonexistent",
                method="iqr",
                threshold=1.5,
            ),
        )
        result = engine.execute(plan)
        assert not result.success
        assert "nonexistent" in result.error
        assert "not found" in result.error.lower()

    def test_outlier_detection_non_numeric_column_error(self):
        """Test outlier detection fails for non-numeric column."""
        df = pd.DataFrame({"category": ["A", "B", "A"], "value": [10.0, 20.0, 30.0]})
        engine = create_engine(df)
        plan = AnalysisPlan(
            operations=[OperationType.OUTLIER_DETECTION],
            outlier_detection=OutlierDetectionSpec(
                column="category",  # string column
                method="iqr",
                threshold=1.5,
            ),
        )
        result = engine.execute(plan)
        assert not result.success
        assert "type" in result.error.lower() or "numeric" in result.error.lower()


class TestAnalysisPlanner:
    def test_parse_valid_plan(self):
        json_str = '''
        {
            "operations": ["row_count", "missing_summary"],
            "filters": null,
            "group_by": null,
            "aggregate": null,
            "sort": null,
            "date_analysis": null,
            "column_stats": ["value"],
            "value_counts": ["category"],
            "correlation_columns": null,
            "limit": 100
        }
        '''
        plan, error = parse_analysis_plan(json_str)
        assert plan is not None, f"Parse failed: {error}"
        assert error is None
        assert OperationType.ROW_COUNT in plan.operations
        assert OperationType.MISSING_SUMMARY in plan.operations
        assert plan.column_stats == ["value"]
        assert plan.value_counts == ["category"]
        assert plan.limit == 100

    def test_parse_plan_with_filters(self):
        json_str = '''
        {
            "operations": ["row_count"],
            "filters": [{"column": "category", "operator": "==", "value": "A"}],
            "group_by": null,
            "aggregate": null,
            "sort": null,
            "date_analysis": null,
            "column_stats": null,
            "value_counts": null,
            "correlation_columns": null,
            "limit": null
        }
        '''
        plan, error = parse_analysis_plan(json_str)
        assert plan is not None, f"Parse failed: {error}"
        assert error is None
        assert len(plan.filters) == 1
        assert plan.filters[0].column == "category"
        assert plan.filters[0].operator == FilterOperator.EQ
        assert plan.filters[0].value == "A"

    def test_parse_plan_with_group_by(self):
        json_str = '''
        {
            "operations": ["group_by"],
            "filters": null,
            "group_by": {"columns": ["category"], "aggregations": {"value": ["sum", "mean"]}},
            "aggregate": null,
            "sort": null,
            "date_analysis": null,
            "column_stats": null,
            "value_counts": null,
            "correlation_columns": null,
            "limit": null
        }
        '''
        plan, error = parse_analysis_plan(json_str)
        assert plan is not None, f"Parse failed: {error}"
        assert error is None
        assert plan.group_by is not None
        assert plan.group_by.columns == ["category"]
        assert AggregationFunction.SUM in plan.group_by.aggregations["value"]

    def test_parse_plan_with_aggregate(self):
        json_str = '''
        {
            "operations": ["aggregate"],
            "filters": null,
            "group_by": null,
            "aggregate": {"columns": ["value"], "functions": ["sum", "mean"], "group_by": ["category"]},
            "sort": null,
            "date_analysis": null,
            "column_stats": null,
            "value_counts": null,
            "correlation_columns": null,
            "limit": null
        }
        '''
        plan, error = parse_analysis_plan(json_str)
        assert plan is not None, f"Parse failed: {error}"
        assert error is None
        assert plan.aggregate is not None
        assert plan.aggregate.columns == ["value"]
        assert AggregationFunction.SUM in plan.aggregate.functions
        assert plan.aggregate.group_by == ["category"]

    def test_parse_invalid_json(self):
        plan, error = parse_analysis_plan("not valid json")
        assert plan is None
        assert error is not None
        assert "valid JSON" in error or "JSON parsing failed" in error

    def test_parse_missing_operations(self):
        json_str = '{"filters": null}'
        plan, error = parse_analysis_plan(json_str)
        assert plan is None
        assert error is not None
        assert "operations" in error.lower()

    def test_parse_json_with_markdown_code_block(self):
        """Regression test: parse JSON wrapped in markdown code block."""
        json_str = '''```json
        {
            "operations": ["group_by", "sort"],
            "filters": null,
            "group_by": {"columns": ["category"], "aggregations": {"value": ["sum", "mean"]}},
            "aggregate": null,
            "sort": {"columns": ["value_sum"], "ascending": [false]},
            "date_analysis": null,
            "column_stats": null,
            "value_counts": null,
            "correlation_columns": null,
            "limit": 100
        }
        ```'''
        plan, error = parse_analysis_plan(json_str)
        assert plan is not None, f"Parse failed: {error}"
        assert error is None
        assert OperationType.GROUP_BY in plan.operations
        assert OperationType.SORT in plan.operations
        assert plan.group_by is not None
        assert plan.group_by.columns == ["category"]
        assert AggregationFunction.SUM in plan.group_by.aggregations["value"]
        assert plan.sort is not None
        assert plan.sort.columns == ["value_sum"]
        assert plan.sort.ascending == [False]

    def test_parse_json_with_extra_text(self):
        """Regression test: parse JSON with extra explanatory text."""
        json_str = '''Here is the analysis plan:
        {
            "operations": ["row_count", "missing_summary"],
            "filters": null,
            "group_by": null,
            "aggregate": null,
            "sort": null,
            "date_analysis": null,
            "column_stats": ["value"],
            "value_counts": ["category"],
            "correlation_columns": null,
            "limit": 100
        }
        This plan counts rows and shows missing values.'''
        plan, error = parse_analysis_plan(json_str)
        assert plan is not None, f"Parse failed: {error}"
        assert error is None
        assert OperationType.ROW_COUNT in plan.operations
        assert OperationType.MISSING_SUMMARY in plan.operations
        assert plan.column_stats == ["value"]
        assert plan.value_counts == ["category"]

    def test_parse_json_with_unsupported_aggregation_fails(self):
        """Regression test: unsupported aggregation function should fail parsing."""
        json_str = '''{
            "operations": ["group_by"],
            "filters": null,
            "group_by": {"columns": ["category"], "aggregations": {"value": ["percentage"]}},
            "aggregate": null,
            "sort": null,
            "date_analysis": null,
            "column_stats": null,
            "value_counts": null,
            "correlation_columns": null,
            "limit": 100
        }'''
        plan, error = parse_analysis_plan(json_str)
        assert plan is None
        assert error is not None
        assert "Schema validation failed" in error

    def test_parse_json_with_trailing_comma(self):
        """Regression test: handle trailing commas in JSON."""
        json_str = '''{
            "operations": ["row_count",],
            "filters": null,
            "group_by": null,
            "aggregate": null,
            "sort": null,
            "date_analysis": null,
            "column_stats": null,
            "value_counts": null,
            "correlation_columns": null,
            "limit": 100,
        }'''
        plan, error = parse_analysis_plan(json_str)
        assert plan is not None, f"Parse failed: {error}"
        assert error is None
        assert OperationType.ROW_COUNT in plan.operations

    def test_parse_json_with_single_quotes(self):
        """Regression test: handle single quotes in JSON."""
        json_str = """{
            'operations': ['row_count'],
            'filters': null,
            'group_by': null,
            'aggregate': null,
            'sort': null,
            'date_analysis': null,
            'column_stats': null,
            'value_counts': null,
            'correlation_columns': null,
            'limit': 100
        }"""
        plan, error = parse_analysis_plan(json_str)
        assert plan is not None, f"Parse failed: {error}"
        assert error is None
        assert OperationType.ROW_COUNT in plan.operations

    def test_parse_json_with_python_none(self):
        """Regression test: handle Python None/True/False in JSON."""
        json_str = """{
            "operations": ["row_count"],
            "filters": None,
            "group_by": null,
            "aggregate": null,
            "sort": null,
            "date_analysis": null,
            "column_stats": null,
            "value_counts": null,
            "correlation_columns": null,
            "limit": 100
        }"""
        plan, error = parse_analysis_plan(json_str)
        assert plan is not None, f"Parse failed: {error}"
        assert error is None
        assert OperationType.ROW_COUNT in plan.operations

    def test_parse_json_with_aggregation_synonyms(self):
        """Regression test: normalize common AI synonyms for aggregation functions."""
        # "total" -> "sum", "average" -> "mean"
        json_str = """{
            "operations": ["group_by"],
            "filters": null,
            "group_by": {
                "columns": ["Market"],
                "aggregations": {
                    "Sales": ["total"],
                    "Late_delivery_risk": ["average"]
                }
            },
            "aggregate": null,
            "sort": null,
            "date_analysis": null,
            "column_stats": null,
            "value_counts": null,
            "correlation_columns": null,
            "limit": 100
        }"""
        plan, error = parse_analysis_plan(json_str)
        assert plan is not None, f"Parse failed: {error}"
        assert error is None
        assert OperationType.GROUP_BY in plan.operations
        assert plan.group_by is not None
        assert "Sales" in plan.group_by.aggregations
        assert "Late_delivery_risk" in plan.group_by.aggregations
        assert AggregationFunction.SUM in plan.group_by.aggregations["Sales"]
        assert AggregationFunction.MEAN in plan.group_by.aggregations["Late_delivery_risk"]

    def test_parse_json_with_aggregate_synonyms(self):
        """Regression test: normalize synonyms in aggregate functions too."""
        json_str = """{
            "operations": ["aggregate"],
            "filters": null,
            "group_by": null,
            "aggregate": {
                "columns": ["Sales", "Risk"],
                "functions": ["total", "avg"],
                "group_by": ["Market"]
            },
            "sort": null,
            "date_analysis": null,
            "column_stats": null,
            "value_counts": null,
            "correlation_columns": null,
            "limit": 100
        }"""
        plan, error = parse_analysis_plan(json_str)
        assert plan is not None, f"Parse failed: {error}"
        assert error is None
        assert OperationType.AGGREGATE in plan.operations
        assert plan.aggregate is not None
        assert AggregationFunction.SUM in plan.aggregate.functions
        assert AggregationFunction.MEAN in plan.aggregate.functions

    def test_sanitize_email(self):
        text = "Contact user@example.com for info"
        result = sanitize_for_ai_context(text)
        assert "[REDACTED]" in result
        assert "user@example.com" not in result

    def test_sanitize_phone(self):
        text = "Call 555-123-4567"
        result = sanitize_for_ai_context(text)
        assert "[REDACTED]" in result

    def test_sanitize_ssn(self):
        text = "SSN: 123-45-6789"
        result = sanitize_for_ai_context(text)
        assert "[REDACTED]" in result

    def test_sanitize_credit_card(self):
        text = "Card: 1234567890123456"
        result = sanitize_for_ai_context(text)
        assert "[REDACTED]" in result

    def test_build_system_prompt(self):
        columns = ["id", "value", "category"]
        dtypes = {"id": "int64", "value": "float64", "category": "object"}
        prompt = build_analysis_system_prompt(columns, dtypes)
        assert "id" in prompt
        assert "value" in prompt
        assert "int64" in prompt
        assert "float64" in prompt


class TestJSONParsingRobustness:
    """Tests for robust JSON parsing fixes."""

    def test_parse_json_with_missing_commas_in_array(self):
        """Test parsing JSON with missing commas between array elements."""
        # This simulates AI output like: "operations": ["group_by" "sort"]
        json_str = '''{
            "operations": ["group_by" "sort"],
            "filters": null,
            "group_by": {"columns": ["category"], "aggregations": {"value": ["sum"]}},
            "aggregate": null,
            "sort": {"columns": ["value_sum"], "ascending": [false]},
            "date_analysis": null,
            "column_stats": null,
            "value_counts": null,
            "correlation_columns": null,
            "limit": 100
        }'''
        plan, error = parse_analysis_plan(json_str)
        assert plan is not None, f"Parse failed: {error}"
        assert error is None
        assert OperationType.GROUP_BY in plan.operations
        assert OperationType.SORT in plan.operations

    def test_parse_json_with_missing_commas_in_object(self):
        """Test parsing JSON with missing commas between object properties."""
        # Missing comma between filters and group_by
        json_str = '''{
            "operations": ["group_by"]
            "filters": null
            "group_by": {"columns": ["category"], "aggregations": {"value": ["sum"]}}
            "aggregate": null
            "sort": null
            "date_analysis": null
            "column_stats": null
            "value_counts": null
            "correlation_columns": null
            "limit": 100
        }'''
        plan, error = parse_analysis_plan(json_str)
        assert plan is not None, f"Parse failed: {error}"
        assert error is None
        assert OperationType.GROUP_BY in plan.operations

    def test_parse_json_with_missing_comma_after_array(self):
        """Test parsing JSON with missing comma after array closing."""
        json_str = '''{
            "operations": ["group_by", "sort"]
            "filters": null,
            "group_by": {"columns": ["category"], "aggregations": {"value": ["sum"]}},
            "aggregate": null,
            "sort": {"columns": ["value_sum"], "ascending": [false]},
            "date_analysis": null,
            "column_stats": null,
            "value_counts": null,
            "correlation_columns": null,
            "limit": 100
        }'''
        plan, error = parse_analysis_plan(json_str)
        assert plan is not None, f"Parse failed: {error}"
        assert error is None
        assert OperationType.GROUP_BY in plan.operations
        assert OperationType.SORT in plan.operations

    def test_parse_json_with_python_comments(self):
        """Test parsing JSON with Python-style comments."""
        json_str = '''{
            "operations": ["group_by"],  // This is a comment
            "filters": null,
            "group_by": {"columns": ["category"], "aggregations": {"value": ["sum"]}},
            "aggregate": null,
            "sort": null,
            "date_analysis": null,
            "column_stats": null,
            "value_counts": null,
            "correlation_columns": null,
            "limit": 100
        }'''
        plan, error = parse_analysis_plan(json_str)
        assert plan is not None, f"Parse failed: {error}"
        assert error is None
        assert OperationType.GROUP_BY in plan.operations

    def test_parse_json_with_c_style_comments(self):
        """Test parsing JSON with C-style comments."""
        json_str = '''{
            "operations": ["group_by"],
            "filters": null,
            "group_by": {"columns": ["category"], "aggregations": {"value": ["sum"]}}, /* comment */
            "aggregate": null,
            "sort": null,
            "date_analysis": null,
            "column_stats": null,
            "value_counts": null,
            "correlation_columns": null,
            "limit": 100
        }'''
        plan, error = parse_analysis_plan(json_str)
        assert plan is not None, f"Parse failed: {error}"
        assert error is None
        assert OperationType.GROUP_BY in plan.operations

    def test_parse_json_with_unclosed_brace_recovery(self):
        """Test that we handle unclosed braces gracefully."""
        # Missing closing brace for group_by
        json_str = '''{
            "operations": ["group_by"],
            "filters": null,
            "group_by": {"columns": ["category"], "aggregations": {"value": ["sum"]}},
            "aggregate": null,
            "sort": null,
            "date_analysis": null,
            "column_stats": null,
            "value_counts": null,
            "correlation_columns": null,
            "limit": 100
        '''
        plan, error = parse_analysis_plan(json_str)
        # Should still extract the JSON object up to the last }
        assert plan is not None, f"Parse failed: {error}"
        assert error is None
        assert OperationType.GROUP_BY in plan.operations


class TestTopNProductsByRevenue:
    """Tests for deterministic top N products by revenue analysis."""

    @pytest.fixture
    def retail_df(self):
        """Sample retail dataset with product and revenue columns."""
        return pd.DataFrame({
            "Product": ["Widget A", "Widget B", "Widget A", "Widget C", "Widget B", "Widget A"],
            "Revenue": [100.0, 200.0, 150.0, 300.0, 250.0, 50.0],
            "Quantity": [1, 2, 1, 3, 2, 1],
            "Unit_Price": [100.0, 100.0, 150.0, 100.0, 125.0, 50.0],
            "Category": ["Electronics", "Electronics", "Electronics", "Home", "Electronics", "Home"],
            "Order_Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-06"]),
        })

    @pytest.fixture
    def supply_chain_df(self):
        """Sample supply chain dataset with item and sales columns."""
        return pd.DataFrame({
            "Item_Description": ["Part A", "Part B", "Part A", "Part C", "Part B"],
            "Sales_Amount": [500.0, 750.0, 600.0, 1000.0, 800.0],
            "Units_Sold": [10, 15, 12, 20, 16],
            "Unit_Cost": [50.0, 50.0, 50.0, 50.0, 50.0],
            "Warehouse": ["WH1", "WH2", "WH1", "WH3", "WH2"],
        })

    def test_is_top_n_products_question_detection(self):
        """Test detection of various top N product question phrasings."""
        from modules.analysis_planner import is_top_n_products_question
        
        questions = [
            "Show the top 10 products by revenue",
            "What are the top 5 products by sales?",
            "Top 3 items by amount",
            "Show me the top 10 skus by total",
            "Best 5 products by revenue",
            "Highest 10 items by sales",
            "Top 10 by revenue",  # generic pattern
        ]
        
        for q in questions:
            result = is_top_n_products_question(q)
            assert result is not None, f"Failed to detect: {q}"
            assert result[0] in ("product", "dimension")
            assert isinstance(result[1], int)
            assert 1 <= result[1] <= 100
        
        # Non-matching questions
        non_matching = [
            "Show me all products",
            "What is the total revenue?",
            "Show revenue by category",
            "List all items",
        ]
        for q in non_matching:
            result = is_top_n_products_question(q)
            assert result is None, f"Should not match: {q}"

    def test_build_top_n_products_plan_with_revenue_column(self, retail_df):
        """Test building plan when revenue column exists."""
        from modules.analysis_planner import build_top_n_products_plan
        from modules.semantic_model import build_semantic_profile
        
        sem_profile = build_semantic_profile(retail_df)
        plan, exec_df, error = build_top_n_products_plan(
            "Show top 10 products by revenue",
            retail_df,
            sem_profile,
            limit=10
        )
        
        assert plan is not None, f"Plan failed: {error}"
        assert error is None
        assert OperationType.GROUP_BY in plan.operations
        assert OperationType.SORT in plan.operations
        assert plan.limit == 10
        assert plan.group_by is not None
        assert "Product" in plan.group_by.columns or "product" in [c.lower() for c in plan.group_by.columns]
        assert "Revenue" in plan.group_by.aggregations or "revenue" in [c.lower() for c in plan.group_by.aggregations.keys()]
        assert AggregationFunction.SUM in list(plan.group_by.aggregations.values())[0]

    def test_build_top_n_products_plan_with_sales_column(self, supply_chain_df):
        """Test building plan with sales column (supply chain terminology)."""
        from modules.analysis_planner import build_top_n_products_plan
        from modules.semantic_model import build_semantic_profile
        
        sem_profile = build_semantic_profile(supply_chain_df)
        plan, exec_df, error = build_top_n_products_plan(
            "Show top 5 items by sales",
            supply_chain_df,
            sem_profile,
            limit=5
        )
        
        assert plan is not None, f"Plan failed: {error}"
        assert error is None
        assert plan.limit == 5
        assert plan.group_by is not None

    def test_build_top_n_products_plan_computed_revenue(self):
        """Test building plan when revenue must be computed from quantity * price."""
        from modules.analysis_planner import build_top_n_products_plan
        from modules.semantic_model import build_semantic_profile
        
        # Dataset with quantity and unit_price but no direct revenue column
        df = pd.DataFrame({
            "Product_Name": ["Widget A", "Widget B", "Widget A", "Widget C"],
            "Qty": [10, 5, 20, 3],
            "Unit_Price": [10.0, 20.0, 15.0, 25.0],
        })
        
        sem_profile = build_semantic_profile(df)
        plan, exec_df, error = build_top_n_products_plan(
            "Show top 10 products by revenue",
            df,
            sem_profile,
            limit=10
        )
        
        assert plan is not None, f"Plan failed: {error}"
        assert error is None
        # Should use computed revenue
        assert plan.group_by is not None

    def test_build_top_n_products_plan_no_product_column(self):
        """Test error when no product dimension column exists."""
        from modules.analysis_planner import build_top_n_products_plan
        from modules.semantic_model import build_semantic_profile
        
        # Dataset with only measures, no dimensions
        df = pd.DataFrame({
            "Revenue": [100.0, 200.0, 150.0],
            "Cost": [50.0, 75.0, 60.0],
        })
        
        sem_profile = build_semantic_profile(df)
        plan, exec_df, error = build_top_n_products_plan(
            "Show top 10 products by revenue",
            df,
            sem_profile,
            limit=10
        )
        
        assert plan is None
        assert error is not None
        assert "product dimension" in error.lower() or "dimension" in error.lower()

    def test_build_top_n_products_plan_no_revenue_column(self):
        """Test error when no revenue column exists and cannot be computed."""
        from modules.analysis_planner import build_top_n_products_plan
        from modules.semantic_model import build_semantic_profile
        
        # Dataset with product but no revenue/quantity/price
        df = pd.DataFrame({
            "Product": ["A", "B", "C"],
            "Category": ["X", "Y", "Z"],
            "Description": ["Desc A", "Desc B", "Desc C"],
        })
        
        sem_profile = build_semantic_profile(df)
        plan, exec_df, error = build_top_n_products_plan(
            "Show top 10 products by revenue",
            df,
            sem_profile,
            limit=10
        )
        
        assert plan is None
        assert error is not None
        assert "revenue" in error.lower()

    def test_top_n_products_end_to_end_execution(self, retail_df):
        """Test full execution of top N products plan."""
        from modules.analysis_planner import build_top_n_products_plan, execute_analysis
        from modules.semantic_model import build_semantic_profile
        
        sem_profile = build_semantic_profile(retail_df)
        plan, exec_df, error = build_top_n_products_plan(
            "Show top 10 products by revenue",
            retail_df,
            sem_profile,
            limit=3
        )
        
        assert plan is not None, f"Plan failed: {error}"
        
        result = execute_analysis(exec_df, plan, sem_profile)
        assert result.success, f"Execution failed: {result.error}"
        
        gb_df = result.data["group_by"]
        assert len(gb_df) == 3  # Top 3
        assert "Product" in gb_df.columns
        assert "Revenue_sum" in gb_df.columns
        
        # Verify sorted descending by revenue
        revenues = gb_df["Revenue_sum"].tolist()
        assert revenues == sorted(revenues, reverse=True)
        
        # Widget B: 200 + 250 = 450 (highest)
        # Widget A: 100 + 150 + 50 = 300
        # Widget C: 300
        top_product = gb_df.iloc[0]["Product"]
        assert top_product == "Widget B"


class TestAnalysisResult:
    def test_success_result(self):
        result = AnalysisResult(success=True, data={"test": "data"}, plan=None, row_count=10)
        assert result.success
        assert result.data["test"] == "data"
        assert result.row_count == 10

    def test_error_result(self):
        result = AnalysisResult(success=False, error="Something went wrong", plan=None)
        assert not result.success
        assert result.error == "Something went wrong"


class TestDashboardKPIs:
    """Regression tests for dashboard KPI calculations and formatting."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "Sales": [100.0, 200.0, 150.0, 250.0, 300.0],
            "Order Id": ["ORD-001", "ORD-002", "ORD-003", "ORD-004", "ORD-005"],
            "Order Profit Per Order": [10.0, 15.0, -5.0, 20.0, 8.0],
            "Late_delivery_risk": [1, 0, 1, 0, 1],
            "Days for shipping (real)": [3, 5, 2, 4, 3],
            "Market": ["A", "B", "A", "B", "A"],
            "Category Name": ["Electronics", "Furniture", "Electronics", "Office", "Electronics"],
        })
    
    def test_calculate_kpis_returns_expected_keys(self, sample_df):
        from modules.dashboard_kpis import calculate_kpis
        kpis = calculate_kpis(sample_df)
        assert set(kpis.keys()) == {
            "total_sales",
            "total_orders",
            "avg_profit_per_order",
            "late_delivery_risk_pct",
            "avg_actual_shipping_days",
        }
    
    def test_calculate_kpis_total_sales(self, sample_df):
        from modules.dashboard_kpis import calculate_kpis
        kpis = calculate_kpis(sample_df)
        # Sales: 100+200+150+250+300 = 1000
        assert kpis["total_sales"] == 1000.0
    
    def test_calculate_kpis_total_orders_distinct(self, sample_df):
        from modules.dashboard_kpis import calculate_kpis
        kpis = calculate_kpis(sample_df)
        # 5 distinct Order Ids
        assert kpis["total_orders"] == 5
    
    def test_calculate_kpis_avg_profit_per_order(self, sample_df):
        from modules.dashboard_kpis import calculate_kpis
        kpis = calculate_kpis(sample_df)
        # Total profit: 10+15-5+20+8 = 48, Orders: 5, Avg: 9.6
        assert kpis["avg_profit_per_order"] == 9.6
    
    def test_calculate_kpis_late_delivery_risk_pct(self, sample_df):
        from modules.dashboard_kpis import calculate_kpis
        kpis = calculate_kpis(sample_df)
        # 3 out of 5 have risk=1 -> 60%
        assert kpis["late_delivery_risk_pct"] == 60.0
    
    def test_calculate_kpis_avg_shipping_days(self, sample_df):
        from modules.dashboard_kpis import calculate_kpis
        kpis = calculate_kpis(sample_df)
        # (3+5+2+4+3)/5 = 3.4
        assert kpis["avg_actual_shipping_days"] == 3.4
    
    def test_calculate_kpis_returns_nan_for_missing_columns(self):
        from modules.dashboard_kpis import calculate_kpis
        df = pd.DataFrame({"Other": [1, 2, 3]})
        kpis = calculate_kpis(df)
        # total_orders has fallback to row count, so check others
        assert pd.isna(kpis["total_sales"])
        assert pd.isna(kpis["avg_profit_per_order"])
        assert pd.isna(kpis["late_delivery_risk_pct"])
        assert pd.isna(kpis["avg_actual_shipping_days"])
        # total_orders falls back to row count
        assert kpis["total_orders"] == 3
    
    def test_format_currency(self):
        from modules.dashboard_kpis import format_currency
        import numpy as np
        assert format_currency(1000000000) == "$1.00B"
        assert format_currency(1500000000) == "$1.50B"
        assert format_currency(1000000) == "$1.00M"
        assert format_currency(1500000) == "$1.50M"
        assert format_currency(1000) == "$1.00K"
        assert format_currency(1500) == "$1.50K"
        assert format_currency(100) == "$100.00"
        assert format_currency(99.99) == "$99.99"
        assert format_currency(np.nan) == "N/A"
        assert format_currency(-1000000) == "$-1.00M"
    
    def test_format_count(self):
        from modules.dashboard_kpis import format_count
        import numpy as np
        assert format_count(1000000) == "1.00M"
        assert format_count(1500000) == "1.50M"
        assert format_count(1000) == "1.0K"
        assert format_count(1500) == "1.5K"
        assert format_count(100) == "100"
        assert format_count(999) == "999"
        assert format_count(np.nan) == "N/A"
    
    def test_format_percentage(self):
        from modules.dashboard_kpis import format_percentage
        import numpy as np
        assert format_percentage(50.0) == "50.0%"
        assert format_percentage(60.5) == "60.5%"
        assert format_percentage(0.0) == "0.0%"
        assert format_percentage(100.0) == "100.0%"
        assert format_percentage(np.nan) == "N/A"
    
    def test_format_days(self):
        from modules.dashboard_kpis import format_days
        import numpy as np
        assert format_days(3.4) == "3.4 days"
        assert format_days(1.0) == "1.0 days"
        assert format_days(np.nan) == "N/A"
    
    def test_format_kpi_value_dispatch(self):
        from modules.dashboard_kpis import format_kpi_value
        import numpy as np
        assert format_kpi_value(1000000, "currency") == "$1.00M"
        assert format_kpi_value(1000000, "count") == "1.00M"
        assert format_kpi_value(50.0, "percentage") == "50.0%"
        assert format_kpi_value(3.4, "days") == "3.4 days"
        assert format_kpi_value(np.nan, "currency") == "N/A"
    
    def test_get_kpi_definitions(self):
        from modules.dashboard_kpis import get_kpi_definitions
        defs = get_kpi_definitions()
        assert len(defs) == 5
        keys = [d["key"] for d in defs]
        assert keys == [
            "total_sales",
            "total_orders",
            "avg_profit_per_order",
            "late_delivery_risk_pct",
            "avg_actual_shipping_days",
        ]
        for d in defs:
            assert "label" in d
            assert "type" in d
            assert "icon" in d
            assert "help" in d


class TestDashboardCharts:
    """Regression tests for dashboard chart utilities."""
    
    def test_create_bar_chart(self):
        from modules.dashboard_charts import create_bar_chart
        import plotly.graph_objects as go
        df = pd.DataFrame({"x": ["A", "B"], "y": [100, 200]})
        fig = create_bar_chart(df, x="x", y="y", title="Test Chart")
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Test Chart"
        assert fig.layout.height == 300
    
    def test_create_line_chart(self):
        from modules.dashboard_charts import create_line_chart
        import plotly.graph_objects as go
        df = pd.DataFrame({"x": ["2024-01", "2024-02"], "y": [100, 200]})
        fig = create_line_chart(df, x="x", y="y", title="Test Line")
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Test Line"
        assert fig.layout.height == 300
    
    def test_format_currency_axis(self):
        from modules.dashboard_charts import format_currency_axis
        assert format_currency_axis(1_000_000_000) == "$1.0B"
        assert format_currency_axis(1_500_000_000) == "$1.5B"
        assert format_currency_axis(1_000_000) == "$1.0M"
        assert format_currency_axis(1_500_000) == "$1.5M"
        assert format_currency_axis(1_000) == "$1K"
        assert format_currency_axis(100) == "$100"
    
    def test_format_percentage_axis(self):
        from modules.dashboard_charts import format_percentage_axis
        assert format_percentage_axis(0.5) == "50%"
        assert format_percentage_axis(1.0) == "100%"
        assert format_percentage_axis(0.0) == "0%"
    
    def test_format_compact_number(self):
        from modules.dashboard_charts import format_compact_number
        assert format_compact_number(1_000_000_000) == "1.0B"
        assert format_compact_number(1_000_000) == "1.0M"
        assert format_compact_number(1_000) == "1K"
        assert format_compact_number(100) == "100"


class TestSemanticModel:
    """Tests for the semantic model module."""
    
    @pytest.fixture
    def sales_df(self):
        return pd.DataFrame({
            "Order Id": ["ORD-001", "ORD-002", "ORD-003", "ORD-004", "ORD-005"],
            "Sales": [100.0, 200.0, 150.0, 250.0, 300.0],
            "Order Date": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15", "2024-03-01"]),
            "Category": ["A", "B", "A", "B", "A"],
            "Region": ["North", "South", "North", "South", "North"],
            "Churned": [0, 1, 0, 1, 0],
        })
    
    def test_build_semantic_profile(self, sales_df):
        from modules.semantic_model import build_semantic_profile, ColumnRole
        profile = build_semantic_profile(sales_df)
        
        assert profile.row_count == 5
        assert profile.column_count == 6
        assert "Order Id" in profile.original_columns
        assert "Sales" in profile.original_columns
        assert "Order Date" in profile.original_columns
        
        # Check role assignments
        assert profile.semantic_columns["Order Id"].role == ColumnRole.IDENTIFIER
        assert profile.semantic_columns["Sales"].role == ColumnRole.CURRENCY
        assert profile.semantic_columns["Order Date"].role == ColumnRole.DATE
        assert profile.semantic_columns["Category"].role == ColumnRole.DIMENSION
        assert profile.semantic_columns["Region"].role == ColumnRole.DIMENSION
        assert profile.semantic_columns["Churned"].role == ColumnRole.BOOLEAN
        
        # Check groupings
        assert "Order Id" in profile.identifiers
        assert "Sales" in profile.currencies
        assert "Order Date" in profile.dates
        assert "Category" in profile.dimensions
        assert "Region" in profile.dimensions
        assert "Churned" in profile.booleans
        
        # Check primary columns
        assert profile.get_primary_measure() == "Sales"
        assert profile.get_primary_date() == "Order Date"
        assert profile.get_primary_identifier() == "Order Id"
    
    def test_semantic_profile_with_string_dates(self):
        """Test that string dates are detected as DATE role."""
        df = pd.DataFrame({
            "date_str": ["2024-01-01", "2024-01-15", "2024-02-01"],
            "value": [100.0, 200.0, 150.0],
        })
        from modules.semantic_model import build_semantic_profile, ColumnRole
        profile = build_semantic_profile(df)
        
        # String dates should be detected as DATE due to name pattern
        assert "date_str" in profile.dates
        assert profile.semantic_columns["date_str"].role == ColumnRole.DATE
    
    def test_normalize_column_name(self):
        from modules.semantic_model import normalize_column_name
        assert normalize_column_name("Order Date") == "order_date"
        assert normalize_column_name("order date (DateOrders)") == "order_date_dateorders"
        assert normalize_column_name("  Sales  ") == "sales"
        assert normalize_column_name("Customer_ID") == "customer_id"


class TestSemanticModelIntegration:
    """Tests for semantic model integration with analytics engine."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "Order Id": ["ORD-001", "ORD-002", "ORD-003", "ORD-004", "ORD-005"],
            "Sales Amount": [100.0, 200.0, 150.0, 250.0, 300.0],
            "Order Date": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15", "2024-03-01"]),
            "Category": ["A", "B", "A", "B", "A"],
            "Region": ["North", "South", "North", "South", "North"],
            "Churned": [0, 1, 0, 1, 0],
            "Quantity": [1, 2, 3, 4, 5],
        })
    
    def test_exact_column_name_match(self, sample_df):
        """Test exact column name resolution."""
        from modules.analytics_engine import create_engine, AnalysisPlan, OperationType, GroupBySpec, AggregationFunction
        from modules.semantic_model import build_semantic_profile
        
        profile = build_semantic_profile(sample_df)
        engine = create_engine(sample_df, profile)
        
        plan = AnalysisPlan(
            operations=[OperationType.GROUP_BY],
            group_by=GroupBySpec(
                columns=["Category"],
                aggregations={"Sales Amount": [AggregationFunction.SUM]}
            )
        )
        result = engine.execute(plan)
        assert result.success
        assert "Sales Amount_sum" in result.data["group_by"].columns
    
    def test_normalized_column_name_match(self, sample_df):
        """Test normalized column name resolution (case/space differences)."""
        from modules.analytics_engine import create_engine, AnalysisPlan, OperationType, GroupBySpec, AggregationFunction
        from modules.semantic_model import build_semantic_profile
        
        profile = build_semantic_profile(sample_df)
        engine = create_engine(sample_df, profile)
        
        # Use normalized names: lowercase, underscores
        plan = AnalysisPlan(
            operations=[OperationType.GROUP_BY],
            group_by=GroupBySpec(
                columns=["category"],  # lowercase
                aggregations={"sales_amount": [AggregationFunction.SUM]}  # normalized
            )
        )
        result = engine.execute(plan)
        assert result.success
        assert "Sales Amount_sum" in result.data["group_by"].columns
    
    def test_ambiguous_column_resolution_raises_error(self, sample_df):
        """Test that truly ambiguous column names (multiple same-role columns) raise clear errors."""
        from modules.analytics_engine import create_engine, AnalysisPlan, OperationType, GroupBySpec, AggregationFunction
        from modules.semantic_model import build_semantic_profile
        
        # Add another currency column that could match "sales"
        df = sample_df.copy()
        df["Revenue Amount"] = [1000.0, 2000.0, 1500.0, 2500.0, 3000.0]
        
        profile = build_semantic_profile(df)
        engine = create_engine(df, profile)
        
        # "sales" could match both "Sales Amount" and "Revenue Amount" (both CURRENCY)
        plan = AnalysisPlan(
            operations=[OperationType.GROUP_BY],
            group_by=GroupBySpec(
                columns=["Category"],
                aggregations={"sales": [AggregationFunction.SUM]}  # ambiguous - two currency columns
            )
        )
        result = engine.execute(plan)
        assert not result.success
        assert "ambiguous" in result.error.lower() or "not found" in result.error.lower()
    
    def test_missing_column_raises_clear_error(self, sample_df):
        """Test that missing columns produce helpful error messages."""
        from modules.analytics_engine import create_engine, AnalysisPlan, OperationType, GroupBySpec, AggregationFunction
        from modules.semantic_model import build_semantic_profile
        
        profile = build_semantic_profile(sample_df)
        engine = create_engine(sample_df, profile)
        
        plan = AnalysisPlan(
            operations=[OperationType.GROUP_BY],
            group_by=GroupBySpec(
                columns=["Category"],
                aggregations={"Nonexistent Column": [AggregationFunction.SUM]}
            )
        )
        result = engine.execute(plan)
        assert not result.success
        assert "not found" in result.error.lower()
        assert "available columns" in result.error.lower()
    
    def test_numeric_conversion_validation(self, sample_df):
        """Test that numeric column validation works."""
        from modules.analytics_engine import create_engine, AnalysisPlan, OperationType, GroupBySpec, AggregationFunction
        from modules.semantic_model import build_semantic_profile
        
        profile = build_semantic_profile(sample_df)
        engine = create_engine(sample_df, profile)
        
        # Try to aggregate a non-numeric column
        plan = AnalysisPlan(
            operations=[OperationType.GROUP_BY],
            group_by=GroupBySpec(
                columns=["Category"],
                aggregations={"Category": [AggregationFunction.SUM]}  # string column
            )
        )
        result = engine.execute(plan)
        assert not result.success
        assert "type" in result.error.lower() or "numeric" in result.error.lower()
    
    def test_date_conversion_from_string(self, sample_df):
        """Test date parsing from string columns in date_analysis."""
        from modules.analytics_engine import create_engine, AnalysisPlan, OperationType, DateAnalysisSpec, AggregationFunction
        from modules.semantic_model import build_semantic_profile
        
        # Create a copy with string dates
        df = sample_df.copy()
        df["Order Date Str"] = df["Order Date"].dt.strftime("%Y-%m-%d")
        df = df.drop(columns=["Order Date"])
        
        profile = build_semantic_profile(df)
        engine = create_engine(df, profile)
        
        plan = AnalysisPlan(
            operations=[OperationType.DATE_ANALYSIS],
            date_analysis=DateAnalysisSpec(
                date_column="order_date_str",  # normalized logical name
                value_column="Sales Amount",
                frequency="M",
                aggregation=AggregationFunction.SUM
            )
        )
        result = engine.execute(plan)
        assert result.success
        assert len(result.data["date_analysis"]) == 3
        assert "Sales Amount_sum" in result.data["date_analysis"].columns
    
    def test_date_conversion_invalid_format_fails_gracefully(self, sample_df):
        """Test that invalid date formats produce clear errors."""
        from modules.analytics_engine import create_engine, AnalysisPlan, OperationType, DateAnalysisSpec, AggregationFunction
        from modules.semantic_model import build_semantic_profile
        
        df = sample_df.copy()
        df["Bad Dates"] = ["not-a-date", "also-bad", "nope", "invalid", "fail"]
        df = df.drop(columns=["Order Date"])
        
        profile = build_semantic_profile(df)
        engine = create_engine(df, profile)
        
        plan = AnalysisPlan(
            operations=[OperationType.DATE_ANALYSIS],
            date_analysis=DateAnalysisSpec(
                date_column="bad_dates",
                value_column="Sales Amount",
                frequency="M",
                aggregation=AggregationFunction.SUM
            )
        )
        result = engine.execute(plan)
        assert not result.success
        assert "could not be parsed" in result.error.lower() or "no valid dates" in result.error.lower()
    
    def test_consistent_column_resolution_across_operations(self, sample_df):
        """Test that resolved column names are used consistently across filtering, aggregation, sorting, and date analysis."""
        from modules.analytics_engine import create_engine, AnalysisPlan, OperationType, GroupBySpec, AggregateSpec, SortSpec, DateAnalysisSpec, AggregationFunction, FilterCondition, FilterOperator
        from modules.semantic_model import build_semantic_profile
        
        profile = build_semantic_profile(sample_df)
        engine = create_engine(sample_df, profile)
        
        # Complex plan with multiple operations using normalized names
        plan = AnalysisPlan(
            operations=[OperationType.FILTER, OperationType.GROUP_BY, OperationType.SORT],
            filters=[FilterCondition(column="region", operator=FilterOperator.EQ, value="North")],
            group_by=GroupBySpec(
                columns=["category"],
                aggregations={"sales_amount": [AggregationFunction.SUM, AggregationFunction.MEAN]}
            ),
            sort=SortSpec(columns=["Sales Amount_sum"], ascending=[False])
        )
        result = engine.execute(plan)
        assert result.success
        gb_df = result.data["group_by"]
        # After filtering for North region, only Category A remains (3 rows: A, A, A)
        assert len(gb_df) == 1
        assert "Sales Amount_sum" in gb_df.columns
        assert "Sales Amount_mean" in gb_df.columns
    
    def test_semantic_profile_drives_primary_column_selection(self, sample_df):
        """Test that semantic profile correctly identifies primary measure, date, identifier."""
        from modules.semantic_model import build_semantic_profile, ColumnRole
        
        profile = build_semantic_profile(sample_df)
        
        # Primary measure should be the currency column
        assert profile.get_primary_measure() == "Sales Amount"
        
        # Primary date should be Order Date
        assert profile.get_primary_date() == "Order Date"
        
        # Primary identifier should be Order Id
        assert profile.get_primary_identifier() == "Order Id"
        
        # Check role assignments
        assert profile.semantic_columns["Sales Amount"].role == ColumnRole.CURRENCY
        assert profile.semantic_columns["Quantity"].role == ColumnRole.MEASURE
        assert profile.semantic_columns["Order Date"].role == ColumnRole.DATE
        assert profile.semantic_columns["Order Id"].role == ColumnRole.IDENTIFIER
        assert profile.semantic_columns["Category"].role == ColumnRole.DIMENSION
        assert profile.semantic_columns["Churned"].role == ColumnRole.BOOLEAN


class TestColumnResolver:
    """Tests for the unified column resolver."""
    
    @pytest.fixture
    def sales_df(self):
        return pd.DataFrame({
            "Order Id": ["ORD-001", "ORD-002", "ORD-003"],
            "Sales": [100.0, 200.0, 150.0],
            "Order Date": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"]),
            "Category": ["A", "B", "A"],
            "Region": ["North", "South", "North"],
        })
    
    def test_resolver_exact_match(self, sales_df):
        from modules.column_resolver import create_resolver
        resolver = create_resolver(sales_df)
        assert resolver.resolve("Sales") == "Sales"
    
    def test_resolver_normalized_match(self, sales_df):
        from modules.column_resolver import create_resolver
        resolver = create_resolver(sales_df)
        assert resolver.resolve("sales") == "Sales"
        assert resolver.resolve("SALES") == "Sales"
        assert resolver.resolve("Order_Date") == "Order Date"
        assert resolver.resolve("order date") == "Order Date"
    
    def test_resolver_with_semantic_profile(self, sales_df):
        from modules.column_resolver import create_resolver
        from modules.semantic_model import build_semantic_profile
        
        profile = build_semantic_profile(sales_df)
        resolver = create_resolver(sales_df, profile)
        
        # Logical names should resolve via semantic profile
        assert resolver.resolve("order_date") == "Order Date"
        assert resolver.resolve("revenue") == "Sales"
        assert resolver.resolve("amount") == "Sales"
    
    def test_resolver_alias_match(self, sales_df):
        from modules.column_resolver import create_resolver
        resolver = create_resolver(sales_df)
        
        # Common aliases
        assert resolver.resolve("order_date") == "Order Date"
        assert resolver.resolve("order date") == "Order Date"
    
    def test_resolver_required_raises(self, sales_df):
        from modules.column_resolver import create_resolver
        resolver = create_resolver(sales_df)
        
        with pytest.raises(ValueError, match="not found"):
            resolver.resolve_required("nonexistent")
    
    def test_resolver_suggest_columns(self, sales_df):
        from modules.column_resolver import create_resolver
        resolver = create_resolver(sales_df)
        
        suggestions = resolver.suggest_columns("sale")
        assert "Sales" in suggestions
    
    def test_resolver_get_primary_measure(self, sales_df):
        from modules.column_resolver import create_resolver
        from modules.semantic_model import build_semantic_profile
        
        profile = build_semantic_profile(sales_df)
        resolver = create_resolver(sales_df, profile)
        
        assert resolver.get_primary_measure() == "Sales"
        assert resolver.get_primary_date() == "Order Date"
        assert resolver.get_primary_identifier() == "Order Id"


class TestDynamicKPIs:
    """Tests for dynamic KPI generation."""
    
    @pytest.fixture
    def sales_df(self):
        return pd.DataFrame({
            "Order Id": ["ORD-001", "ORD-002", "ORD-003", "ORD-004", "ORD-005"],
            "Sales": [100.0, 200.0, 150.0, 250.0, 300.0],
            "Order Profit Per Order": [10.0, 15.0, -5.0, 20.0, 8.0],
            "Order Date": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15", "2024-03-01"]),
            "Category": ["A", "B", "A", "B", "A"],
            "Churned": [0, 1, 0, 1, 0],
            "Days for shipping (real)": [3, 5, 2, 4, 3],
            "Late_delivery_risk": [1, 0, 1, 0, 1],
        })
    
    def test_generate_general_kpis(self, sales_df):
        from modules.dynamic_kpis import generate_general_kpis
        from modules.semantic_model import build_semantic_profile
        
        profile = build_semantic_profile(sales_df)
        kpis = generate_general_kpis(sales_df, profile)
        
        # Should have at least total_records
        keys = {k.key for k in kpis}
        assert "total_records" in keys
        assert "distinct_entities" in keys  # Order Id is identifier
        assert "missing_data_pct" in keys
        assert "duplicate_rows" in keys
        
        # Check values
        total_records = next(k for k in kpis if k.key == "total_records")
        assert total_records.value == 5
        assert total_records.is_valid
    
    def test_generate_sales_retail_kpis(self, sales_df):
        from modules.dynamic_kpis import generate_sales_retail_kpis
        from modules.semantic_model import build_semantic_profile
        
        profile = build_semantic_profile(sales_df)
        kpis = generate_sales_retail_kpis(sales_df, profile)
        
        keys = {k.key for k in kpis}
        assert "total_revenue" in keys
        assert "total_orders" in keys
        assert "avg_order_value" in keys
        
        total_revenue = next(k for k in kpis if k.key == "total_revenue")
        assert total_revenue.value == 1000.0  # 100+200+150+250+300
        assert total_revenue.is_valid
    
    def test_generate_dynamic_kpis_sales_domain(self, sales_df):
        from modules.dynamic_kpis import generate_dynamic_kpis
        from modules.semantic_model import build_semantic_profile
        
        profile = build_semantic_profile(sales_df)
        kpis = generate_dynamic_kpis(sales_df, profile)
        
        keys = {k.key for k in kpis}
        # General KPIs
        assert "total_records" in keys
        assert "missing_data_pct" in keys
        # Sales retail KPIs
        assert "total_revenue" in keys
        assert "total_orders" in keys
        assert "avg_order_value" in keys
        
        # All KPIs should have validity info
        for kpi in kpis:
            assert hasattr(kpi, 'is_valid')
            assert hasattr(kpi, 'formula')
            assert hasattr(kpi, 'source_columns')
    
    def test_format_functions(self):
        from modules.dashboard_kpis import format_currency, format_count, format_percentage, format_days
        import numpy as np
        
        assert format_currency(1_500_000) == "$1.50M"
        assert format_currency(np.nan) == "N/A"
        assert format_count(1_500_000) == "1.50M"
        assert format_count(np.nan) == "N/A"
        assert format_percentage(60.5) == "60.5%"
        assert format_percentage(np.nan) == "N/A"
        assert format_days(3.4) == "3.4 days"
        assert format_days(np.nan) == "N/A"

    def test_safe_sum_numeric_strings(self):
        """Test safe_sum handles numeric strings correctly."""
        import pandas as pd
        import numpy as np
        from modules.dynamic_kpis import safe_sum
        
        # Numeric strings
        s = pd.Series(["100.5", "200.0", "150.25"])
        assert safe_sum(s) == 450.75
        
        # Mixed floats and strings
        s = pd.Series([100.5, "200.0", None, "unknown"])
        assert safe_sum(s) == 300.5
        
        # With None
        s = pd.Series([100.0, None, 200.0])
        assert safe_sum(s) == 300.0
        
        # All invalid strings -> NaN
        s = pd.Series(["unknown", "N/A", "invalid"])
        assert np.isnan(safe_sum(s))
        
        # Empty series -> NaN
        s = pd.Series([], dtype=object)
        assert np.isnan(safe_sum(s))
        
        # Integers as strings
        s = pd.Series(["100", "200", "300"])
        assert safe_sum(s) == 600.0
        
        # Negative numbers as strings
        s = pd.Series(["-50.0", "100.0", "-25.5"])
        assert safe_sum(s) == 24.5
    
    def test_safe_mean_numeric_strings(self):
        """Test safe_mean handles numeric strings correctly."""
        import pandas as pd
        import numpy as np
        from modules.dynamic_kpis import safe_mean
        
        # Numeric strings
        s = pd.Series(["100.5", "200.0", "150.25"])
        assert safe_mean(s) == 150.25
        
        # Mixed floats and strings
        s = pd.Series([100.5, "200.0", None, "unknown"])
        assert safe_mean(s) == 150.25
        
        # All invalid strings -> NaN
        s = pd.Series(["unknown", "N/A"])
        assert np.isnan(safe_mean(s))
        
        # Empty series -> NaN
        s = pd.Series([], dtype=object)
        assert np.isnan(safe_mean(s))
    
    def test_safe_sum_preserves_zero(self):
        """Test that safe_sum correctly handles genuine zero values vs unavailable."""
        import pandas as pd
        import numpy as np
        from modules.dynamic_kpis import safe_sum
        
        # All zeros -> should return 0.0, not NaN
        s = pd.Series([0.0, 0.0, 0.0])
        assert safe_sum(s) == 0.0
        
        # Mix of zeros and valid numbers
        s = pd.Series([0.0, 100.0, 0.0])
        assert safe_sum(s) == 100.0
        
        # Zeros as strings
        s = pd.Series(["0", "0", "100"])
        assert safe_sum(s) == 100.0
    
    def test_kpi_generation_with_string_revenue(self):
        """Test KPI generation works when revenue column contains string numbers."""
        import pandas as pd
        from modules.semantic_model import build_semantic_profile
        from modules.dynamic_kpis import generate_sales_retail_kpis
        
        df = pd.DataFrame({
            "order_id": ["ORD-001", "ORD-002", "ORD-003"],
            "revenue": ["100.0", "200.0", "150.0"],  # String numbers
            "category": ["A", "B", "A"],
        })
        
        profile = build_semantic_profile(df)
        kpis = generate_sales_retail_kpis(df, profile)
        
        # Should find revenue KPI and calculate correctly
        revenue_kpi = next((k for k in kpis if k.key == "total_revenue"), None)
        assert revenue_kpi is not None
        assert revenue_kpi.is_valid
        assert revenue_kpi.value == 450.0
        assert revenue_kpi.validity_reason == "Revenue column found and has valid values"
    
    def test_kpi_generation_with_mixed_revenue(self):
        """Test KPI generation with mixed numeric and string revenue."""
        import pandas as pd
        from modules.semantic_model import build_semantic_profile
        from modules.dynamic_kpis import generate_sales_retail_kpis
        
        df = pd.DataFrame({
            "order_id": ["ORD-001", "ORD-002", "ORD-003", "ORD-004"],
            "revenue": [100.0, "200.0", None, "unknown"],  # Mixed types
            "category": ["A", "B", "A", "B"],
        })
        
        profile = build_semantic_profile(df)
        kpis = generate_sales_retail_kpis(df, profile)
        
        revenue_kpi = next((k for k in kpis if k.key == "total_revenue"), None)
        assert revenue_kpi is not None
        assert revenue_kpi.is_valid
        assert revenue_kpi.value == 300.0  # 100 + 200
    
    def test_kpi_generation_with_all_invalid_revenue(self):
        """Test KPI generation when revenue column has no valid numeric values."""
        import pandas as pd
        from modules.semantic_model import build_semantic_profile
        from modules.dynamic_kpis import generate_sales_retail_kpis
        
        df = pd.DataFrame({
            "order_id": ["ORD-001", "ORD-002", "ORD-003"],
            "revenue": ["unknown", "N/A", "invalid"],  # All invalid
            "category": ["A", "B", "A"],
        })
        
        profile = build_semantic_profile(df)
        kpis = generate_sales_retail_kpis(df, profile)
        
        revenue_kpi = next((k for k in kpis if k.key == "total_revenue"), None)
        assert revenue_kpi is not None
        assert not revenue_kpi.is_valid
        assert "no valid values" in revenue_kpi.validity_reason.lower()


class TestDynamicCharts:
    """Tests for dynamic chart selection."""
    
    @pytest.fixture
    def sales_df(self):
        return pd.DataFrame({
            "Order Id": ["ORD-001", "ORD-002", "ORD-003", "ORD-004", "ORD-005"],
            "Sales": [100.0, 200.0, 150.0, 250.0, 300.0],
            "Order Date": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15", "2024-03-01"]),
            "Category": ["A", "B", "A", "B", "A"],
            "Churned": [0, 1, 0, 1, 0],
        })
    
    def test_infer_chart_type(self, sales_df):
        from modules.dynamic_charts import infer_chart_type, ChartType
        from modules.semantic_model import build_semantic_profile
        
        profile = build_semantic_profile(sales_df)
        
        # Date + Measure -> LINE
        assert infer_chart_type(
            profile.semantic_columns["Order Date"].role,
            profile.semantic_columns["Sales"].role
        ) == ChartType.LINE
        
        # Dimension + Measure -> BAR
        assert infer_chart_type(
            profile.semantic_columns["Category"].role,
            profile.semantic_columns["Sales"].role
        ) == ChartType.BAR
        
        # Dimension + Boolean -> BAR
        assert infer_chart_type(
            profile.semantic_columns["Category"].role,
            profile.semantic_columns["Churned"].role
        ) == ChartType.BAR
    
    def test_build_chart_spec_from_result_group_by(self, sales_df):
        from modules.dynamic_charts import build_chart_spec_from_result, ChartType
        from modules.semantic_model import build_semantic_profile
        from modules.analytics_engine import create_engine, AnalysisPlan, OperationType, GroupBySpec, AggregationFunction
        
        profile = build_semantic_profile(sales_df)
        engine = create_engine(sales_df, profile)
        
        plan = AnalysisPlan(
            operations=[OperationType.GROUP_BY],
            group_by=GroupBySpec(
                columns=["Category"],
                aggregations={"Sales": [AggregationFunction.SUM]}
            )
        )
        result = engine.execute(plan)
        
        spec = build_chart_spec_from_result(
            result.data, "group_by",
            {"group_by": {"columns": ["Category"], "aggregations": {"Sales": ["sum"]}}},
            profile
        )
        
        assert spec is not None
        assert spec.chart_type == ChartType.BAR
        assert spec.x_column == "Category"
        assert spec.y_column == "Sales_sum"
        assert "Sales" in spec.title
        assert "Category" in spec.title
    
    def test_build_chart_spec_from_result_date_analysis(self, sales_df):
        from modules.dynamic_charts import build_chart_spec_from_result, ChartType
        from modules.semantic_model import build_semantic_profile
        from modules.analytics_engine import create_engine, AnalysisPlan, OperationType, DateAnalysisSpec, AggregationFunction
        
        profile = build_semantic_profile(sales_df)
        engine = create_engine(sales_df, profile)
        
        plan = AnalysisPlan(
            operations=[OperationType.DATE_ANALYSIS],
            date_analysis=DateAnalysisSpec(
                date_column="Order Date",
                value_column="Sales",
                frequency="M",
                aggregation=AggregationFunction.SUM
            )
        )
        result = engine.execute(plan)
        
        spec = build_chart_spec_from_result(
            result.data, "date_analysis",
            {"date_analysis": {"date_column": "Order Date", "value_column": "Sales"}},
            profile
        )
        
        assert spec is not None
        assert spec.chart_type == ChartType.LINE
        assert spec.x_column == "period"
        assert spec.y_column == "Sales_sum"
        assert "Sales" in spec.title
    
    def test_create_chart_from_spec(self, sales_df):
        from modules.dynamic_charts import create_chart_from_spec, ChartSpec, ChartType
        from modules.semantic_model import build_semantic_profile
        
        profile = build_semantic_profile(sales_df)
        
        gb_df = pd.DataFrame({"Category": ["A", "B"], "Sales_sum": [550.0, 450.0]})
        
        spec = ChartSpec(
            chart_type=ChartType.BAR,
            x_column="Category",
            y_column="Sales_sum",
            title="Sales by Category",
            y_format="$,.0f",
            height=320,
        )
        
        fig = create_chart_from_spec(gb_df, spec, profile)
        assert fig is not None
        import plotly.graph_objects as go
        assert isinstance(fig, go.Figure)
    
    def test_select_chart_for_question(self, sales_df):
        from modules.dynamic_charts import select_chart_for_question, ChartType
        from modules.semantic_model import build_semantic_profile
        
        profile = build_semantic_profile(sales_df)
        
        # Trend question
        spec = select_chart_for_question("Show monthly sales trend", profile, sales_df)
        assert spec is not None
        assert spec.chart_type == ChartType.LINE
        
        # Comparison question
        spec = select_chart_for_question("Show sales by category", profile, sales_df)
        assert spec is not None
        assert spec.chart_type == ChartType.BAR
        
        # Unsupported question
        spec = select_chart_for_question("What is the meaning of life?", profile, sales_df)
        # May return None or a default spec


class TestInsightsGeneration:
    """Tests for evidence-based insights generation."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "order_id": ["ORD-001", "ORD-002", "ORD-003", "ORD-004", "ORD-005"],
            "revenue": [100.0, 200.0, 150.0, 250.0, 300.0],
            "quantity": [1, 2, 3, 4, 5],
            "order_date": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15", "2024-03-01"]),
            "category": ["A", "B", "A", "B", "A"],
            "customer_id": ["CUST-1", "CUST-2", "CUST-1", "CUST-3", "CUST-2"],
            "is_churned": [0, 1, 0, 1, 0],
            "profit": [10.0, 20.0, -5.0, 30.0, 8.0],
        })
    
    def test_generate_insights_top_dimension(self, sample_df):
        """Test that insights include top dimension by measure."""
        from modules.semantic_model import build_semantic_profile
        from modules.analytics_engine import create_engine, AnalysisPlan, OperationType, GroupBySpec, AggregateSpec, AggregationFunction
        
        profile = build_semantic_profile(sample_df)
        engine = create_engine(sample_df, profile)
        
        primary_measure = profile.get_primary_measure()
        primary_dimension = profile.dimensions[0] if profile.dimensions else None
        
        assert primary_measure == "revenue"
        assert primary_dimension == "category"
        
        plan = AnalysisPlan(
            operations=[OperationType.GROUP_BY],
            group_by=GroupBySpec(
                columns=[primary_dimension],
                aggregations={primary_measure: [AggregationFunction.SUM]}
            )
        )
        result = engine.execute(plan)
        assert result.success
        gb_df = result.data["group_by"].sort_values(f"{primary_measure}_sum", ascending=False)
        top_dim = gb_df.iloc[0][primary_dimension]
        top_val = gb_df.iloc[0][f"{primary_measure}_sum"]
        
        # Category A has 100+150+300=550, Category B has 200+250=450
        assert top_dim == "A"
        assert top_val == 550.0
    
    def test_generate_insights_date_range(self, sample_df):
        """Test that insights include date range."""
        from modules.semantic_model import build_semantic_profile
        
        profile = build_semantic_profile(sample_df)
        primary_date = profile.get_primary_date()
        
        assert primary_date == "order_date"
        
        date_series = sample_df[primary_date]
        valid_dates = date_series.dropna()
        assert valid_dates.min().date() == pd.Timestamp("2024-01-01").date()
        assert valid_dates.max().date() == pd.Timestamp("2024-03-01").date()
    
    def test_generate_insights_measure_statistics(self, sample_df):
        """Test that insights include measure statistics."""
        from modules.semantic_model import build_semantic_profile
        from modules.analytics_engine import create_engine, AnalysisPlan, OperationType, AggregateSpec, AggregationFunction
        
        profile = build_semantic_profile(sample_df)
        engine = create_engine(sample_df, profile)
        
        primary_measure = profile.get_primary_measure()
        
        plan = AnalysisPlan(
            operations=[OperationType.AGGREGATE],
            aggregate=AggregateSpec(
                columns=[primary_measure],
                functions=[AggregationFunction.MEAN, AggregationFunction.MEDIAN, AggregationFunction.STD]
            )
        )
        result = engine.execute(plan)
        assert result.success
        agg_df = result.data["aggregate"]
        
        mean_row = agg_df[agg_df["index"] == "mean"]
        median_row = agg_df[agg_df["index"] == "median"]
        std_row = agg_df[agg_df["index"] == "std"]
        
        assert not mean_row.empty
        assert not median_row.empty
        
        mean_val = mean_row[primary_measure].iloc[0]
        median_val = median_row[primary_measure].iloc[0]
        std_val = std_row[primary_measure].iloc[0] if not std_row.empty else None
        
        # Revenue values: 100, 200, 150, 250, 300 -> mean=200, median=200
        assert mean_val == 200.0
        assert median_val == 200.0
        assert std_val is not None
        assert abs(std_val - 79.06) < 0.1  # Approximate std
    
    def test_generate_insights_churn_rate(self, sample_df):
        """Test that insights include churn/risk rate for boolean columns."""
        from modules.semantic_model import build_semantic_profile
        from modules.analytics_engine import create_engine, AnalysisPlan, OperationType, AggregateSpec, AggregationFunction
        
        profile = build_semantic_profile(sample_df)
        engine = create_engine(sample_df, profile)
        
        # Find churn/risk column
        risk_col = None
        for col in profile.booleans:
            if any(kw in col.lower() for kw in ["risk", "late", "churn", "default", "delay"]):
                risk_col = col
                break
        
        assert risk_col == "is_churned"
        
        plan = AnalysisPlan(
            operations=[OperationType.AGGREGATE],
            aggregate=AggregateSpec(columns=[risk_col], functions=[AggregationFunction.MEAN])
        )
        result = engine.execute(plan)
        assert result.success
        agg_df = result.data["aggregate"]
        
        mean_row = agg_df[agg_df["index"] == "mean"]
        assert not mean_row.empty
        
        rate = mean_row[risk_col].iloc[0] * 100
        # is_churned: 0, 1, 0, 1, 0 -> 2/5 = 40%
        assert rate == 40.0
    
    def test_insights_no_false_claims_for_missing_metrics(self, sample_df):
        """Test that insights don't generate false claims for unavailable metrics."""
        from modules.semantic_model import build_semantic_profile
        from modules.analytics_engine import create_engine
        
        # Remove churn column
        df_no_churn = sample_df.drop(columns=["is_churned"])
        profile = build_semantic_profile(df_no_churn)
        engine = create_engine(df_no_churn, profile)
        
        # Should not find any risk column
        risk_col = None
        for col in profile.booleans:
            if any(kw in col.lower() for kw in ["risk", "late", "churn", "default", "delay"]):
                risk_col = col
                break
        
        assert risk_col is None  # No churn/risk column available
        
        # Generate insights should not crash or produce false claims
        # (We just verify the risk_col detection works correctly)


class TestDatasetSwitching:
    """Tests for dataset switching without stale state."""
    
    def test_dataset_identity_changes_with_content(self):
        """Test that dataset identity changes when content changes but schema stays same."""
        import hashlib
        import pandas as pd
        
        df1 = pd.DataFrame({
            "order_id": ["ORD-001", "ORD-002", "ORD-003"],
            "revenue": [100.0, 200.0, 150.0],
            "category": ["A", "B", "A"],
        })
        
        df2 = pd.DataFrame({
            "order_id": ["ORD-001", "ORD-002", "ORD-003"],
            "revenue": [500.0, 600.0, 700.0],  # Different values
            "category": ["A", "B", "A"],
        })
        
        # Content hash should differ
        hash1 = hashlib.md5(df1.to_csv(index=False).encode()).hexdigest()
        hash2 = hashlib.md5(df2.to_csv(index=False).encode()).hexdigest()
        
        assert hash1 != hash2, "Same schema, different content should have different hashes"
        
        # File size might be the same (this is the bug in current implementation)
        import io
        buf1 = io.BytesIO()
        df1.to_csv(buf1, index=False)
        buf2 = io.BytesIO()
        df2.to_csv(buf2, index=False)
        
        size1 = len(buf1.getvalue())
        size2 = len(buf2.getvalue())
        
        # Note: sizes may be equal, which is why content hash is better
        # This test documents the expected behavior
    
    def test_dataset_identity_changes_with_schema(self):
        """Test that dataset identity changes when schema changes."""
        import hashlib
        import pandas as pd
        
        df1 = pd.DataFrame({
            "order_id": ["ORD-001", "ORD-002", "ORD-003"],
            "revenue": [100.0, 200.0, 150.0],
            "category": ["A", "B", "A"],
        })
        
        df2 = pd.DataFrame({
            "order_id": ["ORD-001", "ORD-002", "ORD-003"],
            "revenue": [100.0, 200.0, 150.0],
            "category": ["A", "B", "A"],
            "new_col": [1, 2, 3],  # Additional column
        })
        
        hash1 = hashlib.md5(df1.to_csv(index=False).encode()).hexdigest()
        hash2 = hashlib.md5(df2.to_csv(index=False).encode()).hexdigest()
        
        assert hash1 != hash2, "Different schema should have different hashes"
    
    def test_semantic_profile_rebuilds_on_dataset_change(self):
        """Test that semantic profile is rebuilt when dataset changes."""
        from modules.semantic_model import build_semantic_profile
        
        df1 = pd.DataFrame({
            "order_id": ["ORD-001", "ORD-002"],
            "revenue": [100.0, 200.0],
            "category": ["A", "B"],
        })
        
        df2 = pd.DataFrame({
            "order_id": ["ORD-001", "ORD-002"],
            "revenue": [500.0, 600.0],
            "category": ["A", "B"],
        })
        
        profile1 = build_semantic_profile(df1)
        profile2 = build_semantic_profile(df2)
        
        # Primary measure should be same column but different values
        assert profile1.get_primary_measure() == "revenue"
        assert profile2.get_primary_measure() == "revenue"
        
        # But the actual values would differ when computed
        assert profile1.semantic_columns["revenue"].sample_values != profile2.semantic_columns["revenue"].sample_values
    
    def test_column_resolver_clears_cache_on_new_dataset(self):
        """Test that column resolver works correctly with new dataset."""
        from modules.column_resolver import create_resolver
        from modules.semantic_model import build_semantic_profile
        
        df1 = pd.DataFrame({
            "order_id": ["ORD-001", "ORD-002"],
            "revenue": [100.0, 200.0],
            "category": ["A", "B"],
        })
        
        df2 = pd.DataFrame({
            "order_id": ["ORD-001", "ORD-002"],
            "sales": [500.0, 600.0],  # Different column name
            "category": ["A", "B"],
        })
        
        profile1 = build_semantic_profile(df1)
        profile2 = build_semantic_profile(df2)
        
        resolver1 = create_resolver(df1, profile1)
        resolver2 = create_resolver(df2, profile2)
        
        # Each resolver should work with its own dataset
        assert resolver1.resolve("revenue") == "revenue"
        assert resolver2.resolve("sales") == "sales"
        
        # Resolver for df1 should resolve "sales" to "revenue" via alias/semantic matching
        # (COMMON_ALIASES maps "sales" to "revenue")
        assert resolver1.resolve("sales") == "revenue"
        
        # Resolver for df2 should resolve "revenue" to "sales" via alias/semantic matching
        assert resolver2.resolve("revenue") == "sales"
        
        # Test that a completely unknown column fails
        assert resolver1.resolve("nonexistent_column") is None
        assert resolver2.resolve("nonexistent_column") is None
        
        # Test that the resolver correctly uses the correct DataFrame's columns
        # df1 has "revenue" column, df2 has "sales" column
        # The resolver's _normalized_to_original index should be specific to each DataFrame
        assert "revenue" in resolver1._normalized_to_original.values()
        assert "sales" in resolver2._normalized_to_original.values()
        assert "revenue" not in resolver2._normalized_to_original.values()
        assert "sales" not in resolver1._normalized_to_original.values()


class TestMetadataQuestionHandler:
    """Tests for the deterministic metadata question handler."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "order_id": ["ORD-001", "ORD-002", "ORD-003", "ORD-004", "ORD-005"],
            "revenue": [100.0, 200.0, 150.0, 250.0, 300.0],
            "quantity": [1, 2, 3, 4, 5],
            "order_date": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01", "2024-02-15", "2024-03-01"]),
            "category": ["A", "B", "A", "B", "A"],
            "customer_id": ["CUST-1", "CUST-2", "CUST-1", "CUST-3", "CUST-2"],
            "is_churned": [0, 1, 0, 1, 0],
            "profit": [10.0, 20.0, -5.0, 30.0, 8.0],
        })
    
    def test_is_metadata_question_columns(self):
        """Test detection of column name questions."""
        from modules.analysis_planner import is_metadata_question
        
        questions = [
            "list out column names",
            "what are the columns?",
            "show me the column names",
            "column names",
            "list columns",
            "what columns are there?",
            "display the column headers",
        ]
        
        for q in questions:
            result = is_metadata_question(q)
            assert result == "columns", f"Failed for: {q}"
    
    def test_is_metadata_question_column_count(self):
        """Test detection of column count questions."""
        from modules.analysis_planner import is_metadata_question
        
        questions = [
            "how many columns are there?",
            "column count",
            "number of columns",
            "what is the column count?",
        ]
        
        for q in questions:
            result = is_metadata_question(q)
            assert result == "column_count", f"Failed for: {q}"
    
    def test_is_metadata_question_shape(self):
        """Test detection of dataset shape questions."""
        from modules.analysis_planner import is_metadata_question
        
        questions = [
            "show dataset shape",
            "what is the shape",
            "dataset shape",
            "show me the shape of the dataset",
        ]
        
        for q in questions:
            result = is_metadata_question(q)
            assert result == "shape", f"Failed for: {q}"
    
    def test_is_metadata_question_row_count(self):
        """Test detection of row count questions."""
        from modules.analysis_planner import is_metadata_question
        
        questions = [
            "how many rows are there?",
            "row count",
            "number of rows",
            "how many records are there?",
            "record count",
        ]
        
        for q in questions:
            result = is_metadata_question(q)
            assert result == "row_count", f"Failed for: {q}"
    
    def test_is_metadata_question_not_matched(self):
        """Test that analytical questions are not matched as metadata."""
        from modules.analysis_planner import is_metadata_question
        
        questions = [
            "what is the average revenue by category?",
            "show monthly sales trend",
            "average delivery risk by market",
            "top 5 customers by revenue",
            "correlation between revenue and profit",
        ]
        
        for q in questions:
            result = is_metadata_question(q)
            assert result is None, f"Should not match: {q}"
    
    def test_handle_metadata_question_columns(self, sample_df):
        """Test handling of column name questions."""
        from modules.analysis_planner import handle_metadata_question
        
        response = handle_metadata_question("list out column names", sample_df, None)
        
        assert response is not None
        assert "Column names:" in response
        assert "order_id" in response
        assert "revenue" in response
        assert "category" in response
        # Should include all 7 columns
        for col in sample_df.columns:
            assert col in response
    
    def test_handle_metadata_question_column_count(self, sample_df):
        """Test handling of column count questions."""
        from modules.analysis_planner import handle_metadata_question
        
        response = handle_metadata_question("how many columns are there?", sample_df, None)
        
        assert response is not None
        assert "Column count:" in response
        assert "8" in response
    
    def test_handle_metadata_question_shape(self, sample_df):
        """Test handling of dataset shape questions."""
        from modules.analysis_planner import handle_metadata_question
        
        response = handle_metadata_question("show dataset shape", sample_df, None)
        
        assert response is not None
        assert "Dataset shape:" in response
        assert "5" in response  # rows
        assert "8" in response  # columns
    
    def test_handle_metadata_question_row_count(self, sample_df):
        """Test handling of row count questions."""
        from modules.analysis_planner import handle_metadata_question
        
        response = handle_metadata_question("how many rows are there?", sample_df, None)
        
        assert response is not None
        assert "Row count:" in response
        assert "5" in response
    
    def test_is_metadata_question_dtypes(self):
        """Test detection of data type questions."""
        from modules.analysis_planner import is_metadata_question
        
        questions = [
            "show data types",
            "what are the data types",
            "column types",
            "dtypes",
            "show me the data types",
        ]
        
        for q in questions:
            result = is_metadata_question(q)
            assert result == "dtypes", f"Failed for: {q}"
    
    def test_is_metadata_question_missing(self):
        """Test detection of missing value questions."""
        from modules.analysis_planner import is_metadata_question
        
        questions = [
            "show missing values",
            "what are the missing values",
            "missing data",
            "missing count",
            "how many missing",
        ]
        
        for q in questions:
            result = is_metadata_question(q)
            assert result == "missing", f"Failed for: {q}"
    
    def test_is_metadata_question_duplicates(self):
        """Test detection of duplicate row questions."""
        from modules.analysis_planner import is_metadata_question
        
        questions = [
            "show duplicates",
            "what are the duplicates",
            "duplicate rows",
            "duplicates",
            "show me duplicates",
        ]
        
        for q in questions:
            result = is_metadata_question(q)
            assert result == "duplicates", f"Failed for: {q}"


    def test_handle_metadata_question_dtypes(self, sample_df):
        """Test handling of data type questions."""
        from modules.analysis_planner import handle_metadata_question
        
        response = handle_metadata_question("show data types", sample_df, None)
        
        assert response is not None
        assert "Column data types:" in response
        for col in sample_df.columns:
            assert col in response
        # Should include dtype info
        assert "int64" in response or "float64" in response
    
    def test_handle_metadata_question_missing(self, sample_df):
        """Test handling of missing value questions."""
        from modules.analysis_planner import handle_metadata_question
        
        # Test with no missing values
        response = handle_metadata_question("show missing values", sample_df, None)
        
        assert response is not None
        assert "Missing values:" in response
        assert "None" in response or "0" in response
        
        # Test with missing values
        import pandas as pd
        df_with_missing = sample_df.copy()
        df_with_missing.loc[0, "revenue"] = None
        response = handle_metadata_question("show missing values", df_with_missing, None)
        
        assert response is not None
        assert "Missing values:" in response
        assert "revenue" in response
        assert "1" in response  # 1 missing value
    
    def test_handle_metadata_question_duplicates(self, sample_df):
        """Test handling of duplicate row questions."""
        from modules.analysis_planner import handle_metadata_question
        
        # Test with no duplicates
        response = handle_metadata_question("show duplicates", sample_df, None)
        
        assert response is not None
        assert "Duplicate rows:" in response
        assert "None" in response or "0" in response
        
        # Test with duplicates
        import pandas as pd
        df_with_dups = sample_df.copy()
        df_with_dups = pd.concat([df_with_dups, df_with_dups.iloc[[0]]], ignore_index=True)
        response = handle_metadata_question("show duplicates", df_with_dups, None)
        
        assert response is not None
        assert "Duplicate rows:" in response
        assert "1" in response  # 1 duplicate row
    

    def test_handle_metadata_question_empty_df(self):
        """Test handling with empty DataFrame."""
        from modules.analysis_planner import handle_metadata_question
        import pandas as pd
        
        empty_df = pd.DataFrame()
        
        response = handle_metadata_question("list column names", empty_df, None)
        assert response is not None
        assert "no columns" in response.lower() or "no dataset" in response.lower()
     
    def test_handle_metadata_question_none_df(self):
        """Test handling with None DataFrame."""
        from modules.analysis_planner import handle_metadata_question
        
        response = handle_metadata_question("list column names", None, None)
        assert response is not None
        assert "no dataset" in response.lower()
     
    def test_handle_metadata_question_returns_none_for_analytical(self, sample_df):
        """Test that analytical questions return None (not handled)."""
        from modules.analysis_planner import handle_metadata_question
        
        response = handle_metadata_question("what is the average revenue by category?", sample_df, None)
        assert response is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])