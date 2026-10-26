from __future__ import annotations

import plotly.graph_objects as go
import plotly.express as px
from typing import Optional


# Executive dashboard color palette
DASHBOARD_COLORS = {
    "primary": "#00d4ff",
    "secondary": "#7c3aed",
    "success": "#22c55e",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "risk": "#ff6b6b",
    "text_primary": "#1e293b",
    "text_secondary": "#64748b",
    "grid": "#e2e8f0",
    "background": "#ffffff",
    "paper_bg": "#f8fafc",
}

# Dark theme variant
DASHBOARD_COLORS_DARK = {
    "primary": "#00d4ff",
    "secondary": "#a78bfa",
    "success": "#4ade80",
    "warning": "#fbbf24",
    "danger": "#f87171",
    "risk": "#fb7185",
    "text_primary": "#f1f5f9",
    "text_secondary": "#94a3b8",
    "grid": "#334155",
    "background": "#0f172a",
    "paper_bg": "#1e293b",
}


def apply_executive_layout(
    fig: go.Figure,
    title: str = "",
    height: int = 300,
    show_legend: bool = False,
    y_format: Optional[str] = None,
    x_format: Optional[str] = None,
    dark: bool = False,
) -> go.Figure:
    """Apply executive dashboard styling to a Plotly figure."""
    colors = DASHBOARD_COLORS_DARK if dark else DASHBOARD_COLORS
    
    layout_kwargs = {
        "height": height,
        "margin": dict(l=50, r=30, t=60, b=50),
        "plot_bgcolor": colors["background"],
        "paper_bgcolor": colors["paper_bg"],
        "font": dict(
            family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color=colors["text_primary"],
            size=12,
        ),
        "title": dict(
            text=title,
            x=0.02,
            xanchor="left",
            font=dict(size=14, color=colors["text_primary"], family="Inter, sans-serif"),
        ),
        "showlegend": show_legend,
        "legend": dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11, color=colors["text_secondary"]),
        ),
        "xaxis": dict(
            showgrid=False,
            showline=True,
            linecolor=colors["grid"],
            linewidth=1,
            tickfont=dict(size=11, color=colors["text_secondary"]),
            title_font=dict(size=12, color=colors["text_primary"]),
            zeroline=False,
        ),
        "yaxis": dict(
            showgrid=True,
            gridcolor=colors["grid"],
            gridwidth=1,
            showline=False,
            tickfont=dict(size=11, color=colors["text_secondary"]),
            title_font=dict(size=12, color=colors["text_primary"]),
            zeroline=False,
        ),
        "hovermode": "x unified",
        "hoverlabel": dict(
            bgcolor=colors["paper_bg"],
            font_size=12,
            font_family="Inter, sans-serif",
            font_color=colors["text_primary"],
            bordercolor=colors["grid"],
        ),
    }
    
    if y_format:
        layout_kwargs["yaxis"]["tickformat"] = y_format
    if x_format:
        layout_kwargs["xaxis"]["tickformat"] = x_format
    
    fig.update_layout(**layout_kwargs)
    return fig


def create_bar_chart(
    df,
    x: str,
    y: str,
    title: str = "",
    color: Optional[str] = None,
    color_discrete_sequence: Optional[list] = None,
    y_format: Optional[str] = None,
    height: int = 300,
    dark: bool = False,
    hover_template: Optional[str] = None,
) -> go.Figure:
    """Create a professionally styled bar chart."""
    colors = DASHBOARD_COLORS_DARK if dark else DASHBOARD_COLORS
    
    if color_discrete_sequence is None:
        color_discrete_sequence = [colors["primary"]]
    
    fig = px.bar(df, x=x, y=y, color=color, color_discrete_sequence=color_discrete_sequence)
    
    # Update bar styling
    fig.update_traces(
        marker_line_width=0,
        marker_line_color=colors["background"],
        opacity=0.95,
        hovertemplate=hover_template or f"<b>%{{x}}</b><br>%{{y:,.0f}}<extra></extra>",
    )
    
    apply_executive_layout(fig, title=title, height=height, y_format=y_format, dark=dark)
    return fig


def create_line_chart(
    df,
    x: str,
    y: str,
    title: str = "",
    color_discrete_sequence: Optional[list] = None,
    y_format: Optional[str] = None,
    height: int = 300,
    dark: bool = False,
    markers: bool = True,
    hover_template: Optional[str] = None,
) -> go.Figure:
    """Create a professionally styled line chart."""
    colors = DASHBOARD_COLORS_DARK if dark else DASHBOARD_COLORS
    
    if color_discrete_sequence is None:
        color_discrete_sequence = [colors["primary"]]
    
    fig = px.line(df, x=x, y=y, color_discrete_sequence=color_discrete_sequence, markers=markers)
    
    # Update line styling
    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=6, line=dict(width=1, color=colors["background"])),
        hovertemplate=hover_template or f"<b>%{{x}}</b><br>%{{y:,.0f}}<extra></extra>",
    )
    
    # Add area fill for visual weight
    if not dark:
        fig.update_traces(
            fill="tozeroy",
            fillcolor="rgba(0, 212, 255, 0.1)",
        )
    
    apply_executive_layout(fig, title=title, height=height, y_format=y_format, dark=dark)
    return fig


def create_horizontal_bar_chart(
    df,
    x: str,
    y: str,
    title: str = "",
    color_discrete_sequence: Optional[list] = None,
    x_format: Optional[str] = None,
    height: int = 300,
    dark: bool = False,
    hover_template: Optional[str] = None,
) -> go.Figure:
    """Create a professionally styled horizontal bar chart."""
    colors = DASHBOARD_COLORS_DARK if dark else DASHBOARD_COLORS
    
    if color_discrete_sequence is None:
        color_discrete_sequence = [colors["primary"]]
    
    fig = px.bar(df, x=x, y=y, orientation="h", color_discrete_sequence=color_discrete_sequence)
    
    fig.update_traces(
        marker_line_width=0,
        opacity=0.95,
        hovertemplate=hover_template or f"<b>%{{y}}</b><br>%{{x:,.0f}}<extra></extra>",
    )
    
    layout_kwargs = {
        "height": height,
        "margin": dict(l=100, r=30, t=60, b=50),
        "plot_bgcolor": colors["background"],
        "paper_bgcolor": colors["paper_bg"],
        "font": dict(
            family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            color=colors["text_primary"],
            size=12,
        ),
        "title": dict(
            text=title,
            x=0.02,
            xanchor="left",
            font=dict(size=14, color=colors["text_primary"], family="Inter, sans-serif"),
        ),
        "showlegend": False,
        "xaxis": dict(
            showgrid=True,
            gridcolor=colors["grid"],
            gridwidth=1,
            showline=False,
            tickfont=dict(size=11, color=colors["text_secondary"]),
            title_font=dict(size=12, color=colors["text_primary"]),
            zeroline=False,
        ),
        "yaxis": dict(
            showgrid=False,
            showline=True,
            linecolor=colors["grid"],
            linewidth=1,
            tickfont=dict(size=11, color=colors["text_secondary"]),
            title_font=dict(size=12, color=colors["text_primary"]),
            zeroline=False,
            categoryorder="total ascending",
        ),
        "hovermode": "y unified",
        "hoverlabel": dict(
            bgcolor=colors["paper_bg"],
            font_size=12,
            font_family="Inter, sans-serif",
            font_color=colors["text_primary"],
            bordercolor=colors["grid"],
        ),
    }
    
    if x_format:
        layout_kwargs["xaxis"]["tickformat"] = x_format
    
    fig.update_layout(**layout_kwargs)
    return fig


def format_currency_axis(value: float) -> str:
    """Format axis value as currency."""
    if value >= 1_000_000_000:
        return f"${value/1_000_000_000:.1f}B"
    elif value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value/1_000:.0f}K"
    else:
        return f"${value:,.0f}"


def format_percentage_axis(value: float) -> str:
    """Format axis value as percentage."""
    return f"{value:.0%}"


def format_compact_number(value: float) -> str:
    """Format axis value with compact notation."""
    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:.1f}B"
    elif value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"{value/1_000:.0f}K"
    else:
        return f"{value:,.0f}"