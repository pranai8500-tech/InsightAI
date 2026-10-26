from __future__ import annotations

"""
Dynamic chart selection based on column roles and data characteristics.

This module selects appropriate chart types based on the semantic roles
of columns involved in the analysis, ensuring meaningful visualizations
for any dataset.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Literal

import pandas as pd

from modules.semantic_model import ColumnRole, DatasetProfile
from modules.dashboard_charts import (
    create_bar_chart,
    create_line_chart,
    create_horizontal_bar_chart,
    apply_executive_layout,
    format_currency_axis,
    format_percentage_axis,
)
import plotly.graph_objects as go
import plotly.express as px


class ChartType(str, Enum):
    """Supported chart types."""
    BAR = "bar"
    LINE = "line"
    HORIZONTAL_BAR = "horizontal_bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    PIE = "pie"
    GROUPED_BAR = "grouped_bar"
    HEATMAP = "heatmap"
    BOX = "box"
    VIOLIN = "violin"


@dataclass
class ChartSpec:
    """Specification for a chart to be created."""
    chart_type: ChartType
    x_column: str
    y_column: Optional[str] = None
    color_column: Optional[str] = None
    title: str = ""
    x_format: Optional[str] = None
    y_format: Optional[str] = None
    height: int = 300
    hover_template: Optional[str] = None
    # Additional options
    show_legend: bool = False
    markers: bool = True
    orientation: Literal["v", "h"] = "v"
    # For pie charts
    values_column: Optional[str] = None
    names_column: Optional[str] = None


def infer_chart_type(
    x_role: ColumnRole,
    y_role: Optional[ColumnRole],
    color_role: Optional[ColumnRole] = None,
    x_cardinality: int = 0,
    y_cardinality: int = 0,
) -> ChartType:
    """
    Infer the best chart type based on column roles and cardinality.
    
    Args:
        x_role: Role of the x-axis column
        y_role: Role of the y-axis column (None for counts/histograms)
        color_role: Role of the color/grouping column
        x_cardinality: Number of unique values in x column
        y_cardinality: Number of unique values in y column
    
    Returns:
        Recommended ChartType
    """
    # Time series: date + measure
    if x_role == ColumnRole.DATE and y_role in (ColumnRole.MEASURE, ColumnRole.CURRENCY):
        return ChartType.LINE
    
    # Category comparison: dimension + measure
    if x_role == ColumnRole.DIMENSION and y_role in (ColumnRole.MEASURE, ColumnRole.CURRENCY):
        if x_cardinality <= 20:
            return ChartType.BAR
        else:
            return ChartType.HORIZONTAL_BAR
    
    # Distribution: single measure
    if y_role is None and x_role in (ColumnRole.MEASURE, ColumnRole.CURRENCY):
        return ChartType.HISTOGRAM
    
    # Scatter: two measures
    if x_role in (ColumnRole.MEASURE, ColumnRole.CURRENCY) and y_role in (ColumnRole.MEASURE, ColumnRole.CURRENCY):
        return ChartType.SCATTER
    
    # Category counts: dimension only
    if y_role is None and x_role == ColumnRole.DIMENSION:
        if x_cardinality <= 10:
            return ChartType.PIE
        elif x_cardinality <= 30:
            return ChartType.BAR
        else:
            return ChartType.HORIZONTAL_BAR
    
    # Boolean/risk by dimension
    if x_role == ColumnRole.DIMENSION and y_role == ColumnRole.BOOLEAN:
        return ChartType.BAR
    
    # Measure by dimension with color grouping
    if x_role == ColumnRole.DIMENSION and y_role in (ColumnRole.MEASURE, ColumnRole.CURRENCY) and color_role:
        return ChartType.GROUPED_BAR
    
    # Correlation heatmap
    if x_role in (ColumnRole.MEASURE, ColumnRole.CURRENCY) and y_role in (ColumnRole.MEASURE, ColumnRole.CURRENCY) and color_role is None:
        # Could also be scatter, but heatmap for correlation matrix
        return ChartType.HEATMAP
    
    # Default fallback
    return ChartType.BAR


def get_format_for_role(role: ColumnRole, is_currency: bool = False) -> Optional[str]:
    """Get axis format string for a column role."""
    if role == ColumnRole.CURRENCY or is_currency:
        return "$,.0f"
    elif role == ColumnRole.MEASURE:
        return ",.0f"
    elif role == ColumnRole.BOOLEAN:
        return ".0%"
    elif role == ColumnRole.DATE:
        return None  # Handled by period formatting
    elif role == ColumnRole.DIMENSION:
        return None
    elif role == ColumnRole.IDENTIFIER:
        return None
    return None


def create_chart_from_spec(
    df: pd.DataFrame,
    spec: ChartSpec,
    semantic_profile: Optional[DatasetProfile] = None,
) -> go.Figure:
    """
    Create a chart from a ChartSpec using the appropriate chart function.
    """
    # Apply semantic formatting if profile available
    if semantic_profile and spec.y_column:
        y_col_obj = semantic_profile.semantic_columns.get(spec.y_column)
        if y_col_obj and y_col_obj.role == ColumnRole.CURRENCY:
            spec.y_format = spec.y_format or "$,.0f"
        elif y_col_obj and y_col_obj.role == ColumnRole.BOOLEAN:
            spec.y_format = spec.y_format or ".0%"
    
    if spec.chart_type == ChartType.BAR:
        return create_bar_chart(
            df,
            x=spec.x_column,
            y=spec.y_column,
            title=spec.title,
            color=spec.color_column,
            y_format=spec.y_format,
            height=spec.height,
            hover_template=spec.hover_template,
        )
    
    elif spec.chart_type == ChartType.LINE:
        return create_line_chart(
            df,
            x=spec.x_column,
            y=spec.y_column,
            title=spec.title,
            y_format=spec.y_format,
            height=spec.height,
            markers=spec.markers,
            hover_template=spec.hover_template,
        )
    
    elif spec.chart_type == ChartType.HORIZONTAL_BAR:
        return create_horizontal_bar_chart(
            df,
            x=spec.x_column,
            y=spec.y_column,
            title=spec.title,
            x_format=spec.x_format,
            height=spec.height,
            hover_template=spec.hover_template,
        )
    
    elif spec.chart_type == ChartType.SCATTER:
        fig = px.scatter(
            df, x=spec.x_column, y=spec.y_column, color=spec.color_column,
            title=spec.title, height=spec.height
        )
        apply_executive_layout(fig, title=spec.title, height=spec.height)
        return fig
    
    elif spec.chart_type == ChartType.HISTOGRAM:
        fig = px.histogram(
            df, x=spec.x_column, color=spec.color_column,
            title=spec.title, height=spec.height, nbins=30
        )
        apply_executive_layout(fig, title=spec.title, height=spec.height)
        return fig
    
    elif spec.chart_type == ChartType.PIE:
        fig = px.pie(
            df, values=spec.values_column, names=spec.names_column,
            title=spec.title, height=spec.height
        )
        apply_executive_layout(fig, title=spec.title, height=spec.height)
        return fig
    
    elif spec.chart_type == ChartType.GROUPED_BAR:
        fig = px.bar(
            df, x=spec.x_column, y=spec.y_column, color=spec.color_column,
            title=spec.title, height=spec.height, barmode="group"
        )
        apply_executive_layout(fig, title=spec.title, height=spec.height, show_legend=True)
        return fig
    
    elif spec.chart_type == ChartType.HEATMAP:
        # For correlation matrices
        if spec.x_column == spec.y_column and spec.x_column == "index":
            # Correlation matrix format
            corr_df = df.set_index("index") if "index" in df.columns else df
            fig = px.imshow(
                corr_df, text_auto=".2f", aspect="auto",
                title=spec.title, height=spec.height,
                color_continuous_scale="RdBu", zmin=-1, zmax=1
            )
            apply_executive_layout(fig, title=spec.title, height=spec.height)
            return fig
        # Otherwise regular heatmap
        fig = px.density_heatmap(
            df, x=spec.x_column, y=spec.y_column,
            title=spec.title, height=spec.height
        )
        apply_executive_layout(fig, title=spec.title, height=spec.height)
        return fig
    
    elif spec.chart_type == ChartType.BOX:
        fig = px.box(
            df, x=spec.x_column, y=spec.y_column, color=spec.color_column,
            title=spec.title, height=spec.height
        )
        apply_executive_layout(fig, title=spec.title, height=spec.height)
        return fig
    
    elif spec.chart_type == ChartType.VIOLIN:
        fig = px.violin(
            df, x=spec.x_column, y=spec.y_column, color=spec.color_column,
            title=spec.title, height=spec.height, box=True
        )
        apply_executive_layout(fig, title=spec.title, height=spec.height)
        return fig
    
    # Fallback
    return create_bar_chart(df, x=spec.x_column, y=spec.y_column, title=spec.title, height=spec.height)


def build_chart_spec_from_result(
    result_data: dict,
    operation: str,
    plan_spec: dict,
    semantic_profile: Optional[DatasetProfile] = None,
) -> Optional[ChartSpec]:
    """
    Build a ChartSpec from analytics engine result and plan specification.
    
    Args:
        result_data: The data returned from analytics engine (group_by, date_analysis, etc.)
        operation: The operation type (group_by, date_analysis, etc.)
        plan_spec: The original plan specification (group_by, date_analysis, etc.)
        semantic_profile: Optional semantic profile for formatting hints
    
    Returns:
        ChartSpec if a chart can be created, None otherwise
    """
    if operation == "group_by" and "group_by" in result_data:
        gb_df = result_data["group_by"]
        if gb_df.empty:
            return None
        
        group_cols = plan_spec.get("group_by", {}).get("columns", [])
        aggregations = plan_spec.get("group_by", {}).get("aggregations", {})
        
        if not group_cols:
            return None
        
        x_col = group_cols[0]
        # Find first aggregated value column
        value_cols = [c for c in gb_df.columns if c not in group_cols]
        if not value_cols:
            return None
        
        y_col = value_cols[0]
        
        # Get roles from semantic profile
        x_role = ColumnRole.DIMENSION
        y_role = ColumnRole.MEASURE
        is_currency = False
        
        if semantic_profile:
            x_sem = semantic_profile.semantic_columns.get(x_col)
            y_sem = semantic_profile.semantic_columns.get(y_col.replace("_sum", "").replace("_mean", "").replace("_count", ""))
            if x_sem:
                x_role = x_sem.role
            if y_sem:
                y_role = y_sem.role
                is_currency = y_sem.role == ColumnRole.CURRENCY
        
        x_cardinality = gb_df[x_col].nunique()
        
        chart_type = infer_chart_type(x_role, y_role, x_cardinality=x_cardinality)
        
        # Override for percentage data
        if y_role == ColumnRole.BOOLEAN or (y_col.endswith("_mean") and "risk" in y_col.lower()):
            chart_type = ChartType.BAR
        
        return ChartSpec(
            chart_type=chart_type,
            x_column=x_col,
            y_column=y_col,
            title=f"{y_col.replace('_sum', '').replace('_mean', '').replace('_count', '')} by {x_col}",
            y_format=get_format_for_role(y_role, is_currency),
            hover_template=f"<b>%{{x}}</b><br>{y_col}: %{{y:,.0f}}<extra></extra>",
            height=320,
        )
    
    elif operation == "date_analysis" and "date_analysis" in result_data:
        da_df = result_data["date_analysis"]
        if da_df.empty:
            return None
        
        date_col = plan_spec.get("date_analysis", {}).get("date_column")
        value_col = plan_spec.get("date_analysis", {}).get("value_column")
        
        # Find the actual value column in result
        if "period" in da_df.columns:
            x_col = "period"
        else:
            x_col = da_df.columns[0]
        
        value_cols = [c for c in da_df.columns if c != x_col]
        if not value_cols:
            return None
        
        y_col = value_cols[0]
        
        # Get roles
        y_role = ColumnRole.MEASURE
        is_currency = False
        
        if semantic_profile and value_col:
            y_sem = semantic_profile.semantic_columns.get(value_col)
            if y_sem:
                y_role = y_sem.role
                is_currency = y_sem.role == ColumnRole.CURRENCY
        
        return ChartSpec(
            chart_type=ChartType.LINE,
            x_column=x_col,
            y_column=y_col,
            title=f"{value_col or y_col} Over Time",
            y_format=get_format_for_role(y_role, is_currency),
            hover_template=f"<b>%{{x}}</b><br>{value_col or y_col}: %{{y:,.0f}}<extra></extra>",
            height=320,
            markers=True,
        )
    
    elif operation == "aggregate" and "aggregate" in result_data:
        agg_df = result_data["aggregate"]
        if agg_df.empty:
            return None
        
        # Single row aggregate - show as metrics, not chart
        if len(agg_df) == 1:
            return None
        
        # Grouped aggregate - treat like group_by
        group_cols = plan_spec.get("aggregate", {}).get("group_by", [])
        if group_cols:
            x_col = group_cols[0]
            value_cols = [c for c in agg_df.columns if c not in group_cols]
            if value_cols:
                y_col = value_cols[0]
                
                y_role = ColumnRole.MEASURE
                is_currency = False
                if semantic_profile:
                    y_sem = semantic_profile.semantic_columns.get(y_col.replace("_sum", "").replace("_mean", ""))
                    if y_sem:
                        y_role = y_sem.role
                        is_currency = y_sem.role == ColumnRole.CURRENCY
                
                return ChartSpec(
                    chart_type=ChartType.BAR,
                    x_column=x_col,
                    y_column=y_col,
                    title=f"{y_col} by {x_col}",
                    y_format=get_format_for_role(y_role, is_currency),
                    height=320,
                )
    
    elif operation == "correlation" and "correlation" in result_data:
        corr_df = result_data["correlation"]
        if corr_df.empty:
            return None
        
        return ChartSpec(
            chart_type=ChartType.HEATMAP,
            x_column="index" if "index" in corr_df.columns else corr_df.columns[0],
            y_column="index",
            title="Correlation Matrix",
            height=400,
        )
    
    elif operation == "value_counts" and "value_counts" in result_data:
        vc_data = result_data["value_counts"]
        if not vc_data:
            return None
        
        # Use first value_counts result
        col_name, vc_df = next(iter(vc_data.items()))
        if vc_df.empty:
            return None
        
        # Category counts - pie if few categories, bar if many
        cardinality = len(vc_df)
        chart_type = ChartType.PIE if cardinality <= 10 else ChartType.BAR
        
        return ChartSpec(
            chart_type=chart_type,
            x_column=vc_df.columns[0] if chart_type == ChartType.BAR else None,
            y_column=vc_df.columns[1] if chart_type == ChartType.BAR else None,
            names_column=vc_df.columns[0] if chart_type == ChartType.PIE else None,
            values_column=vc_df.columns[1] if chart_type == ChartType.PIE else None,
            title=f"Distribution of {col_name}",
            height=320,
        )
    
    elif operation == "column_stats" and "column_stats" in result_data:
        stats_df = result_data["column_stats"]
        if stats_df.empty:
            return None
        
        # Could show distribution as histogram
        # For now, return None (show as table)
        return None
    
    elif operation == "missing_summary" and "missing_summary" in result_data:
        missing_df = result_data["missing_summary"]
        if missing_df.empty:
            return None
        
        # Bar chart of missing percentages
        return ChartSpec(
            chart_type=ChartType.BAR,
            x_column="column",
            y_column="missing_percentage",
            title="Missing Values by Column (%)",
            y_format=".1f",
            height=320,
        )
    
    return None


def select_chart_for_question(
    question: str,
    semantic_profile: DatasetProfile,
    df: pd.DataFrame,
) -> Optional[ChartSpec]:
    """
    Select an appropriate chart type based on the user's question and dataset profile.
    
    This is used when the AI doesn't specify a chart but the question implies one.
    """
    q_lower = question.lower()
    
    primary_measure = semantic_profile.get_primary_measure()
    primary_date = semantic_profile.get_primary_date()
    primary_dimension = semantic_profile.dimensions[0] if semantic_profile.dimensions else None
    
    # Trend/time series questions
    if any(kw in q_lower for kw in ["trend", "over time", "time series", "monthly", "quarterly", "yearly", "daily"]):
        if primary_date and primary_measure:
            return ChartSpec(
                chart_type=ChartType.LINE,
                x_column=primary_date,
                y_column=primary_measure,
                title=f"{primary_measure} Trend Over Time",
                y_format=get_format_for_role(
                    semantic_profile.semantic_columns.get(primary_measure, ColumnRole.MEASURE).role if primary_measure in semantic_profile.semantic_columns else ColumnRole.MEASURE,
                    semantic_profile.semantic_columns.get(primary_measure, ColumnRole.MEASURE).role == ColumnRole.CURRENCY if primary_measure in semantic_profile.semantic_columns else False
                ),
                height=350,
            )
    
    # Comparison by category
    if any(kw in q_lower for kw in ["by", "per", "each", "compare", "comparison", "breakdown"]):
        if primary_dimension and primary_measure:
            return ChartSpec(
                chart_type=ChartType.BAR,
                x_column=primary_dimension,
                y_column=primary_measure,
                title=f"{primary_measure} by {primary_dimension}",
                y_format=get_format_for_role(
                    semantic_profile.semantic_columns.get(primary_measure, ColumnRole.MEASURE).role if primary_measure in semantic_profile.semantic_columns else ColumnRole.MEASURE,
                    semantic_profile.semantic_columns.get(primary_measure, ColumnRole.MEASURE).role == ColumnRole.CURRENCY if primary_measure in semantic_profile.semantic_columns else False
                ),
                height=320,
            )
    
    # Distribution
    if any(kw in q_lower for kw in ["distribution", "histogram", "spread", "frequency"]):
        if primary_measure:
            return ChartSpec(
                chart_type=ChartType.HISTOGRAM,
                x_column=primary_measure,
                title=f"Distribution of {primary_measure}",
                height=320,
            )
    
    # Correlation/relationship
    if any(kw in q_lower for kw in ["correlation", "relationship", "vs", "versus", "scatter"]):
        measures = semantic_profile.measures + semantic_profile.currencies
        if len(measures) >= 2:
            return ChartSpec(
                chart_type=ChartType.SCATTER,
                x_column=measures[0],
                y_column=measures[1],
                title=f"{measures[1]} vs {measures[0]}",
                height=350,
            )
    
    return None