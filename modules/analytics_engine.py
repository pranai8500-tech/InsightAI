from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Optional

import pandas as pd

from modules.column_resolver import ColumnResolver, create_resolver
from modules.semantic_model import DatasetProfile


class OperationType(str, Enum):
    ROW_COUNT = "row_count"
    MISSING_SUMMARY = "missing_summary"
    FILTER = "filter"
    GROUP_BY = "group_by"
    AGGREGATE = "aggregate"
    SORT = "sort"
    DATE_ANALYSIS = "date_analysis"
    COLUMN_STATS = "column_stats"
    VALUE_COUNTS = "value_counts"
    CORRELATION = "correlation"
    MOM_COMPARISON = "mom_comparison"
    OUTLIER_DETECTION = "outlier_detection"


class AggregationFunction(str, Enum):
    SUM = "sum"
    MEAN = "mean"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    STD = "std"
    MEDIAN = "median"


class FilterOperator(str, Enum):
    EQ = "=="
    NE = "!="
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="
    CONTAINS = "contains"
    STARTSWITH = "startswith"
    ENDSWITH = "endswith"
    IN = "in"
    NOT_IN = "not_in"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


@dataclass
class FilterCondition:
    column: str
    operator: FilterOperator
    value: Any = None


@dataclass
class GroupBySpec:
    columns: list[str]
    aggregations: dict[str, list[AggregationFunction]]


@dataclass
class AggregateSpec:
    columns: list[str]
    functions: list[AggregationFunction]
    group_by: list[str] | None = None


@dataclass
class SortSpec:
    columns: list[str]
    ascending: list[bool] | bool = True


@dataclass
class DateAnalysisSpec:
    date_column: str
    value_column: str | None = None
    frequency: Literal["D", "W", "M", "Q", "Y"] = "M"
    aggregation: AggregationFunction = AggregationFunction.SUM


@dataclass
class MoMComparisonSpec:
    date_column: str
    value_column: str
    frequency: Literal["M", "Q"] = "M"
    aggregation: AggregationFunction = AggregationFunction.SUM
    periods: int = 12  # Number of periods to include


@dataclass
class OutlierDetectionSpec:
    column: str
    method: Literal["iqr", "zscore"] = "iqr"
    threshold: float = 1.5  # For IQR: multiplier, for z-score: standard deviations


@dataclass
class AnalysisPlan:
    operations: list[OperationType]
    filters: list[FilterCondition] | None = None
    group_by: GroupBySpec | None = None
    aggregate: AggregateSpec | None = None
    sort: SortSpec | None = None
    date_analysis: DateAnalysisSpec | None = None
    mom_comparison: MoMComparisonSpec | None = None
    outlier_detection: OutlierDetectionSpec | None = None
    column_stats: list[str] | None = None
    value_counts: list[str] | None = None
    correlation_columns: list[str] | None = None
    limit: int | None = None


@dataclass
class AnalysisResult:
    success: bool
    data: Any = None
    error: str | None = None
    plan: AnalysisPlan | None = None
    row_count: int | None = None


ALLOWED_COLUMN_TYPES = {
    "numeric": ["int64", "int32", "int16", "int8", "float64", "float32", "float16"],
    "datetime": ["datetime64[ns]", "datetime64[ns, UTC]", "datetime64[us]", "datetime64[ms]", "datetime64[s]"],
    "string": ["object", "string", "str"],
    "categorical": ["category"],
    "boolean": ["bool"],
}

MAX_RESULT_ROWS = 1000
MAX_GROUPBY_CARDINALITY = 1000


class AnalyticsEngine:
    def __init__(self, df: pd.DataFrame, semantic_profile: Optional[DatasetProfile] = None):
        self.df = df.copy()
        self.semantic_profile = semantic_profile
        self.resolver = create_resolver(df, semantic_profile)
        self._validate_dataframe()

    def _validate_dataframe(self) -> None:
        if self.df.empty:
            raise ValueError("Dataset is empty")
        if len(self.df.columns) == 0:
            raise ValueError("Dataset has no columns")

    def _validate_column_exists(self, column: str, df: pd.DataFrame | None = None) -> str:
        """Validate column exists and return resolved column name."""
        target_df = df if df is not None else self.df
        # Create a temporary resolver for the target_df if it's different from self.df
        if target_df is not self.df:
            temp_resolver = create_resolver(target_df, self.semantic_profile)
            resolved = temp_resolver.resolve(column)
        else:
            resolved = self.resolver.resolve(column)
        
        if resolved is None:
            available = list(target_df.columns)
            context = "working DataFrame" if df is not None else "dataset"
            raise ValueError(
                f"Column '{column}' not found in {context}. Available columns: {available}"
            )
        return resolved

    def _validate_columns_exist(self, columns: list[str], df: pd.DataFrame | None = None) -> list[str]:
        """Validate all columns exist and return resolved column names."""
        target_df = df if df is not None else self.df
        if target_df is not self.df:
            temp_resolver = create_resolver(target_df, self.semantic_profile)
            return [temp_resolver.resolve_required(col) for col in columns]
        return [self.resolver.resolve_required(col) for col in columns]

    def _validate_column_type(self, column: str, allowed_types: list[str], df: pd.DataFrame | None = None) -> str:
        resolved = self._validate_column_exists(column, df)
        dtype_str = str(self.df[resolved].dtype) if df is None else str(df[resolved].dtype)
        if dtype_str not in allowed_types:
            raise ValueError(
                f"Column '{column}' (resolved to '{resolved}') has type '{dtype_str}', expected one of {allowed_types}"
            )
        return resolved

    def _validate_numeric_column(self, column: str, df: pd.DataFrame | None = None) -> str:
        return self._validate_column_type(column, ALLOWED_COLUMN_TYPES["numeric"], df)

    def _validate_datetime_column(self, column: str, df: pd.DataFrame | None = None) -> str:
        return self._validate_column_type(column, ALLOWED_COLUMN_TYPES["datetime"], df)

    def _apply_filters(self, df: pd.DataFrame, filters: list[FilterCondition]) -> pd.DataFrame:
        result = df.copy()
        for f in filters:
            resolved_col = self._validate_column_exists(f.column, result)
            col = result[resolved_col]

            if f.operator == FilterOperator.EQ:
                result = result[col == f.value]
            elif f.operator == FilterOperator.NE:
                result = result[col != f.value]
            elif f.operator == FilterOperator.GT:
                self._validate_numeric_column(f.column, result)
                result = result[col > f.value]
            elif f.operator == FilterOperator.GTE:
                self._validate_numeric_column(f.column, result)
                result = result[col >= f.value]
            elif f.operator == FilterOperator.LT:
                self._validate_numeric_column(f.column, result)
                result = result[col < f.value]
            elif f.operator == FilterOperator.LTE:
                self._validate_numeric_column(f.column, result)
                result = result[col <= f.value]
            elif f.operator == FilterOperator.CONTAINS:
                self._validate_column_type(f.column, ALLOWED_COLUMN_TYPES["string"], result)
                result = result[col.astype(str).str.contains(str(f.value), case=False, na=False)]
            elif f.operator == FilterOperator.STARTSWITH:
                self._validate_column_type(f.column, ALLOWED_COLUMN_TYPES["string"], result)
                result = result[col.astype(str).str.startswith(str(f.value), na=False)]
            elif f.operator == FilterOperator.ENDSWITH:
                self._validate_column_type(f.column, ALLOWED_COLUMN_TYPES["string"], result)
                result = result[col.astype(str).str.endswith(str(f.value), na=False)]
            elif f.operator == FilterOperator.IN:
                result = result[col.isin(f.value if isinstance(f.value, list) else [f.value])]
            elif f.operator == FilterOperator.NOT_IN:
                result = result[~col.isin(f.value if isinstance(f.value, list) else [f.value])]
            elif f.operator == FilterOperator.IS_NULL:
                result = result[col.isna()]
            elif f.operator == FilterOperator.IS_NOT_NULL:
                result = result[col.notna()]
            else:
                raise ValueError(f"Unsupported filter operator: {f.operator}")

        return result

    def execute(self, plan: AnalysisPlan) -> AnalysisResult:
        try:
            # Start with original dataframe
            result_df = self.df.copy()

            # Apply filters first
            if plan.filters:
                result_df = self._apply_filters(result_df, plan.filters)

            # Keep a reference to the filtered original dataframe for operations that need original columns
            filtered_df = result_df.copy()

            if result_df.empty:
                return AnalysisResult(
                    success=True,
                    data={"row_count": 0},
                    plan=plan,
                    row_count=0,
                )

            results = {}
            executed_ops = []

            for op in plan.operations:
                if op == OperationType.ROW_COUNT:
                    results["row_count"] = int(len(result_df))
                    executed_ops.append("row_count")

                elif op == OperationType.MISSING_SUMMARY:
                    missing_data = []
                    for col in filtered_df.columns:
                        missing_count = int(filtered_df[col].isna().sum())
                        missing_pct = round(missing_count / len(filtered_df) * 100, 2) if len(filtered_df) > 0 else 0.0
                        missing_data.append({
                            "column": col,
                            "missing_count": missing_count,
                            "missing_percentage": missing_pct,
                            "dtype": str(filtered_df[col].dtype),
                        })
                    results["missing_summary"] = pd.DataFrame(missing_data)
                    executed_ops.append("missing_summary")

                elif op == OperationType.COLUMN_STATS:
                    cols = plan.column_stats or filtered_df.columns.tolist()
                    self._validate_columns_exist(cols, filtered_df)
                    stats_data = []
                    for col in cols:
                        series = filtered_df[col]
                        dtype = str(series.dtype)
                        stat = {
                            "column": col,
                            "dtype": dtype,
                            "count": int(series.count()),
                            "missing": int(series.isna().sum()),
                            "unique": int(series.nunique(dropna=True)),
                        }
                        if dtype in ALLOWED_COLUMN_TYPES["numeric"]:
                            stat.update({
                                "mean": float(series.mean()) if series.count() > 0 else None,
                                "std": float(series.std()) if series.count() > 1 else None,
                                "min": float(series.min()) if series.count() > 0 else None,
                                "max": float(series.max()) if series.count() > 0 else None,
                                "median": float(series.median()) if series.count() > 0 else None,
                            })
                        elif dtype in ALLOWED_COLUMN_TYPES["datetime"]:
                            stat.update({
                                "min": series.min() if series.count() > 0 else None,
                                "max": series.max() if series.count() > 0 else None,
                            })
                        stats_data.append(stat)
                    results["column_stats"] = pd.DataFrame(stats_data)
                    executed_ops.append("column_stats")

                elif op == OperationType.VALUE_COUNTS:
                    cols = plan.value_counts or []
                    self._validate_columns_exist(cols, filtered_df)
                    vc_results = {}
                    for col in cols:
                        vc = filtered_df[col].value_counts(dropna=False).head(50)
                        vc_results[col] = pd.DataFrame({"value": vc.index.astype(str), "count": vc.values})
                    results["value_counts"] = vc_results
                    executed_ops.append("value_counts")

                elif op == OperationType.GROUP_BY:
                    if not plan.group_by:
                        raise ValueError("group_by operation requires GroupBySpec")
                    group_cols = self._validate_columns_exist(plan.group_by.columns)
                    agg_dict = {}
                    for col, funcs in plan.group_by.aggregations.items():
                        resolved_col = self._validate_column_exists(col)
                        # Validate numeric columns for numeric aggregation functions
                        for func in funcs:
                            if func not in AggregationFunction:
                                raise ValueError(f"Unsupported aggregation function: {func}")
                            if func in (AggregationFunction.SUM, AggregationFunction.MEAN, AggregationFunction.STD, AggregationFunction.MEDIAN, AggregationFunction.MIN, AggregationFunction.MAX):
                                self._validate_numeric_column(col)
                        agg_dict[resolved_col] = [f.value for f in funcs]

                    grouped = result_df.groupby(group_cols, dropna=False).agg(agg_dict)
                    grouped.columns = ["_".join(map(str, c)).strip() for c in grouped.columns.values]
                    grouped = grouped.reset_index()
                    if len(grouped) > MAX_GROUPBY_CARDINALITY:
                        raise ValueError(f"GroupBy result exceeds maximum cardinality ({MAX_GROUPBY_CARDINALITY})")
                    results["group_by"] = grouped
                    result_df = grouped
                    executed_ops.append("group_by")

                elif op == OperationType.AGGREGATE:
                    if not plan.aggregate:
                        raise ValueError("aggregate operation requires AggregateSpec")
                    agg_cols = self._validate_columns_exist(plan.aggregate.columns)
                    for func in plan.aggregate.functions:
                        if func not in AggregationFunction:
                            raise ValueError(f"Unsupported aggregation function: {func}")
                        if func in (AggregationFunction.SUM, AggregationFunction.MEAN, AggregationFunction.STD, AggregationFunction.MEDIAN, AggregationFunction.MIN, AggregationFunction.MAX):
                            for col in agg_cols:
                                self._validate_numeric_column(col)

                    agg_dict = {col: [f.value for f in plan.aggregate.functions] for col in agg_cols}
                    if plan.aggregate.group_by:
                        group_cols = self._validate_columns_exist(plan.aggregate.group_by)
                        grouped = result_df.groupby(group_cols, dropna=False).agg(agg_dict)
                        grouped.columns = ["_".join(map(str, c)).strip() for c in grouped.columns.values]
                        grouped = grouped.reset_index()
                        if len(grouped) > MAX_GROUPBY_CARDINALITY:
                            raise ValueError(f"GroupBy result exceeds maximum cardinality ({MAX_GROUPBY_CARDINALITY})")
                        results["aggregate"] = grouped
                        result_df = grouped
                    else:
                        agg_result = result_df[agg_cols].agg([f.value for f in plan.aggregate.functions])
                        results["aggregate"] = agg_result.reset_index()
                    executed_ops.append("aggregate")

                elif op == OperationType.SORT:
                    if not plan.sort:
                        raise ValueError("sort operation requires SortSpec")
                    sort_cols = self._validate_columns_exist(plan.sort.columns, result_df)
                    ascending = plan.sort.ascending
                    if isinstance(ascending, bool):
                        ascending = [ascending] * len(sort_cols)
                    result_df = result_df.sort_values(by=sort_cols, ascending=ascending)
                    # Update stored results if they exist and are DataFrames
                    for key in ("group_by", "aggregate"):
                        if key in results and isinstance(results[key], pd.DataFrame):
                            results[key] = result_df.copy()
                    executed_ops.append("sort")

                elif op == OperationType.DATE_ANALYSIS:
                    if not plan.date_analysis:
                        raise ValueError("date_analysis operation requires DateAnalysisSpec")
                    
                    # Resolve column names against the filtered original DataFrame
                    # This ensures columns exist in the original data we're actually operating on
                    date_col = self._validate_column_exists(plan.date_analysis.date_column, filtered_df)
                    
                    # Resolve value column name if provided
                    value_col = None
                    if plan.date_analysis.value_column:
                        value_col = self._validate_numeric_column(plan.date_analysis.value_column, filtered_df)
                    
                    # Create a working copy to avoid mutating original
                    df_copy = filtered_df.copy()
                    
                    # Verify resolved columns exist in the working DataFrame
                    if date_col not in df_copy.columns:
                        raise ValueError(
                            f"Resolved date column '{date_col}' not found in working DataFrame. "
                            f"Available columns: {list(df_copy.columns)}"
                        )
                    if value_col and value_col not in df_copy.columns:
                        raise ValueError(
                            f"Resolved value column '{value_col}' not found in working DataFrame. "
                            f"Available columns: {list(df_copy.columns)}"
                        )
                    
                    # Parse date column if not already datetime
                    if str(df_copy[date_col].dtype) not in ALLOWED_COLUMN_TYPES["datetime"]:
                        # Try common date formats
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message="Could not infer format")
                            parsed = pd.to_datetime(df_copy[date_col], errors="coerce")
                        invalid_count = int(parsed.isna().sum()) - int(df_copy[date_col].isna().sum())
                        if parsed.notna().sum() == 0:
                            raise ValueError(f"Column '{plan.date_analysis.date_column}' (resolved to '{date_col}') could not be parsed as dates. No valid dates found.")
                        df_copy[date_col] = parsed
                    
                    # Drop rows with invalid/missing dates (NaT) before grouping
                    df_copy = df_copy.dropna(subset=[date_col])
                    
                    if df_copy.empty:
                        raise ValueError(f"Column '{date_col}' has no valid dates after parsing.")
                    
                    df_copy["_period"] = df_copy[date_col].dt.to_period(plan.date_analysis.frequency)
                    
                    if value_col:
                        grouped = df_copy.groupby("_period", dropna=False)[value_col].agg(
                            plan.date_analysis.aggregation.value
                        ).reset_index()
                        grouped.columns = ["period", f"{value_col}_{plan.date_analysis.aggregation.value}"]
                    else:
                        grouped = df_copy.groupby("_period", dropna=False).size().reset_index(name="count")
                    grouped["period"] = grouped["period"].astype(str)
                    results["date_analysis"] = grouped
                    result_df = grouped
                    executed_ops.append("date_analysis")

                elif op == OperationType.MOM_COMPARISON:
                    if not plan.mom_comparison:
                        raise ValueError("mom_comparison operation requires MoMComparisonSpec")
                    
                    # Resolve column names against filtered original DataFrame
                    date_col = self._validate_column_exists(plan.mom_comparison.date_column, filtered_df)
                    value_col = self._validate_numeric_column(plan.mom_comparison.value_column, filtered_df)
                    
                    # Verify resolved columns exist
                    if date_col not in filtered_df.columns:
                        raise ValueError(f"Resolved date column '{date_col}' not found in working DataFrame.")
                    if value_col not in filtered_df.columns:
                        raise ValueError(f"Resolved value column '{value_col}' not found in working DataFrame.")
                    
                    # Parse date column if not already datetime
                    df_copy = filtered_df.copy()
                    if str(df_copy[date_col].dtype) not in ALLOWED_COLUMN_TYPES["datetime"]:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message="Could not infer format")
                            parsed = pd.to_datetime(df_copy[date_col], errors="coerce")
                        if parsed.notna().sum() == 0:
                            raise ValueError(f"Column '{plan.mom_comparison.date_column}' (resolved to '{date_col}') could not be parsed as dates.")
                        df_copy[date_col] = parsed
                    
                    df_copy = df_copy.dropna(subset=[date_col])
                    if df_copy.empty:
                        raise ValueError(f"Column '{date_col}' has no valid dates after parsing.")
                    
                    df_copy["_period"] = df_copy[date_col].dt.to_period(plan.mom_comparison.frequency)
                    
                    # Aggregate by period
                    grouped = df_copy.groupby("_period", dropna=False)[value_col].agg(
                        plan.mom_comparison.aggregation.value
                    ).reset_index()
                    grouped.columns = ["period", f"{value_col}_{plan.mom_comparison.aggregation.value}"]
                    grouped["period"] = grouped["period"].astype(str)
                    
                    # Sort by period chronologically
                    grouped = grouped.sort_values("period").reset_index(drop=True)
                    
                    # Calculate MoM change
                    value_col_name = f"{value_col}_{plan.mom_comparison.aggregation.value}"
                    grouped[f"{value_col_name}_mom_pct_change"] = grouped[value_col_name].pct_change() * 100
                    grouped[f"{value_col_name}_mom_abs_change"] = grouped[value_col_name].diff()
                    
                    # Limit periods if specified
                    if plan.mom_comparison.periods and len(grouped) > plan.mom_comparison.periods:
                        grouped = grouped.tail(plan.mom_comparison.periods)
                    
                    results["mom_comparison"] = grouped
                    result_df = grouped
                    executed_ops.append("mom_comparison")

                elif op == OperationType.OUTLIER_DETECTION:
                    if not plan.outlier_detection:
                        raise ValueError("outlier_detection operation requires OutlierDetectionSpec")
                    
                    # Resolve column against filtered original DataFrame
                    col = self._validate_numeric_column(plan.outlier_detection.column, filtered_df)
                    
                    if col not in filtered_df.columns:
                        raise ValueError(f"Resolved column '{col}' not found in working DataFrame.")
                    
                    series = filtered_df[col].dropna()
                    if len(series) == 0:
                        raise ValueError(f"Column '{col}' has no valid numeric values.")
                    
                    outliers_info = []
                    
                    if plan.outlier_detection.method == "iqr":
                        q1 = series.quantile(0.25)
                        q3 = series.quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - plan.outlier_detection.threshold * iqr
                        upper_bound = q3 + plan.outlier_detection.threshold * iqr
                        
                        outlier_mask = (filtered_df[col] < lower_bound) | (filtered_df[col] > upper_bound)
                        outliers = filtered_df[outlier_mask]
                        
                        outliers_info.append({
                            "method": "iqr",
                            "threshold": plan.outlier_detection.threshold,
                            "lower_bound": float(lower_bound),
                            "upper_bound": float(upper_bound),
                            "outlier_count": int(outlier_mask.sum()),
                            "total_count": int(series.count()),
                            "outlier_percentage": round(outlier_mask.sum() / len(series) * 100, 2),
                        })
                    
                    elif plan.outlier_detection.method == "zscore":
                        mean_val = series.mean()
                        std_val = series.std()
                        if std_val == 0:
                            raise ValueError(f"Column '{col}' has zero standard deviation, cannot compute z-scores.")
                        
                        z_scores = (filtered_df[col] - mean_val) / std_val
                        outlier_mask = z_scores.abs() > plan.outlier_detection.threshold
                        outliers = filtered_df[outlier_mask]
                        
                        outliers_info.append({
                            "method": "zscore",
                            "threshold": plan.outlier_detection.threshold,
                            "mean": float(mean_val),
                            "std": float(std_val),
                            "outlier_count": int(outlier_mask.sum()),
                            "total_count": int(series.count()),
                            "outlier_percentage": round(outlier_mask.sum() / len(series) * 100, 2),
                        })
                    
                    # Prepare outlier details
                    if len(outliers) > 0:
                        outlier_cols = [c for c in outliers.columns if not c.startswith("_")]
                        outlier_details = outliers[outlier_cols].head(100).to_dict(orient="records")
                    else:
                        outlier_details = []
                    
                    results["outlier_detection"] = {
                        "summary": outliers_info,
                        "outliers": outlier_details,
                    }
                    executed_ops.append("outlier_detection")

                elif op == OperationType.CORRELATION:
                    if not plan.correlation_columns:
                        raise ValueError("correlation operation requires correlation_columns")
                    corr_cols = self._validate_columns_exist(plan.correlation_columns, filtered_df)
                    numeric_cols = [c for c in corr_cols if str(filtered_df[c].dtype) in ALLOWED_COLUMN_TYPES["numeric"]]
                    if len(numeric_cols) < 2:
                        raise ValueError("Correlation requires at least 2 numeric columns")
                    corr_matrix = filtered_df[numeric_cols].corr()
                    results["correlation"] = corr_matrix.reset_index()
                    executed_ops.append("correlation")

            # Apply limit if specified
            if plan.limit and "row_count" not in results:
                for key, val in results.items():
                    if isinstance(val, pd.DataFrame) and len(val) > plan.limit:
                        results[key] = val.head(plan.limit)
                    elif isinstance(val, dict):
                        for sub_key, sub_val in val.items():
                            if isinstance(sub_val, pd.DataFrame) and len(sub_val) > plan.limit:
                                val[sub_key] = sub_val.head(plan.limit)

            return AnalysisResult(
                success=True,
                data=results,
                plan=plan,
                row_count=len(result_df),
            )

        except Exception as e:
            return AnalysisResult(
                success=False,
                error=str(e),
                plan=plan,
            )


def create_engine(df: pd.DataFrame, semantic_profile: Optional[DatasetProfile] = None) -> AnalyticsEngine:
    return AnalyticsEngine(df, semantic_profile)