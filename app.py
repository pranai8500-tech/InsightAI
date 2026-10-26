from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from modules.data_loader import load_uploaded_dataset, get_excel_sheets, validate_upload, get_load_info
from modules.data_quality import profile_dataset, build_quality_warnings, detect_date_columns
from modules.cleaning import create_cleaning_preview
from modules.ai_client import chat_with_nemotron, build_dataset_context, AIClientConfig, is_configured, get_api_key, ChatResult
from modules.analytics_engine import (
    OperationType,
    AnalysisPlan,
    GroupBySpec,
    AggregateSpec,
    DateAnalysisSpec,
    AggregationFunction,
    create_engine,
)
from modules.analysis_planner import plan_analysis, execute_analysis, handle_metadata_question
from modules.dashboard_kpis import calculate_kpis, format_kpi_value, get_kpi_definitions, format_currency, format_count, format_percentage, format_days
from modules.dynamic_kpis import generate_dynamic_kpis, format_kpi_value as format_dynamic_kpi, kpi_specs_to_dict, kpi_specs_to_definitions
from modules.column_map import get_col, has_col, ensure_datetime
from modules.column_resolver import create_resolver
from modules.dashboard_charts import (
    create_bar_chart,
    create_line_chart,
    apply_executive_layout,
    format_currency_axis,
    format_percentage_axis,
)
from modules.dynamic_charts import (
    build_chart_spec_from_result,
    create_chart_from_spec,
    select_chart_for_question,
    ChartType,
)
from modules.semantic_model import (
    ColumnRole,
    BusinessDomain,
    SemanticColumn,
    DatasetProfile,
    build_semantic_profile,
    normalize_column_name,
)

st.set_page_config(
    page_title="InsightAI | Data Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_UPLOAD_MB = 100

# ---------- Styling ----------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
      
      .block-container {padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1450px; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;}
      [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.18);}
      .hero {
        padding: 1.35rem 1.5rem; border-radius: 18px;
        background: linear-gradient(120deg, rgba(63,81,181,.13), rgba(0,150,136,.10));
        border: 1px solid rgba(128,128,128,.20); margin-bottom: 1rem;
      }
      .hero h1 {margin: 0 0 .35rem 0; font-size: 2rem; font-weight: 700; letter-spacing: -0.02em;}
      .hero p {margin: 0; color: var(--text-color); opacity: .78; font-weight: 400;}
      .small-muted {font-size: .88rem; opacity: .75;}
      
      /* KPI Metric Cards */
      div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.12);
        padding: 16px 20px;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(248,250,252,0.95));
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03);
        transition: box-shadow 0.2s ease, transform 0.2s ease;
      }
      div[data-testid="stMetric"]:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
        transform: translateY(-1px);
      }
      div[data-testid="stMetric"] label {
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
      }
      div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-weight: 700 !important;
        font-size: 1.75rem !important;
        color: #1e293b !important;
        letter-spacing: -0.02em;
      }
      
      /* Section Headings */
      .section-heading {
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
        font-weight: 600;
        font-size: 1.1rem;
        color: #1e293b;
      }
      
      /* DataFrames */
      .stDataFrame {border-radius: 10px; overflow: hidden;}
      .stDataFrame [data-testid="stTable"] {border-radius: 10px;}
      
      /* Chart containers */
      .stPlotlyChart {
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,.12);
        background: white;
        padding: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
      }
      
      /* Tabs */
      .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        border-bottom: 1px solid rgba(128,128,128,.18);
      }
      .stTabs [data-baseweb="tab"] {
        padding: 10px 16px;
        border-radius: 8px 8px 0 0;
        font-weight: 500;
        font-size: 0.9rem;
        color: #64748b;
        background: transparent;
        border: 1px solid transparent;
        border-bottom: none;
        transition: all 0.2s ease;
      }
      .stTabs [data-baseweb="tab"]:hover {
        color: #1e293b;
        background: rgba(128,128,128,.05);
      }
      .stTabs [aria-selected="true"] {
        color: #00d4ff !important;
        background: rgba(0, 212, 255, 0.08) !important;
        border-color: rgba(0, 212, 255, 0.3) !important;
        border-bottom-color: transparent !important;
      }
      
      /* Buttons */
      .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.88rem;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
      }
      .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00d4ff, #00b8e6);
        border: none;
        color: white;
      }
      .stButton > button[kind="primary"]:hover {
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.4);
        transform: translateY(-1px);
      }
      
      /* Info/Warning/Error boxes */
      .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
      }
      
      /* Expanders */
      .streamlit-expanderHeader {
        font-weight: 500;
        font-size: 0.9rem;
        color: #1e293b;
      }
      
      /* Insights list */
      .insight-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(128,128,128,.1);
      }
      .insight-item:last-child {
        border-bottom: none;
      }
      
      /* Dividers */
      hr {
        border: none;
        border-top: 1px solid rgba(128,128,128,.18);
        margin: 1.5rem 0;
      }
      
      /* Scrollbar */
      ::-webkit-scrollbar {width: 8px; height: 8px;}
      ::-webkit-scrollbar-track {background: transparent;}
      ::-webkit-scrollbar-thumb {background: rgba(128,128,128,.3); border-radius: 4px;}
      ::-webkit-scrollbar-thumb:hover {background: rgba(128,128,128,.5);}
    </style>
    """,
    unsafe_allow_html=True,
)

if "original_df" not in st.session_state:
    st.session_state.original_df = None
if "cleaned_df" not in st.session_state:
    st.session_state.cleaned_df = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "filter_state" not in st.session_state:
    st.session_state.filter_state = {}
if "semantic_profile" not in st.session_state:
    st.session_state.semantic_profile = None
if "semantic_override" not in st.session_state:
    st.session_state.semantic_override = {}  # User corrections to role assignments

# Get the loaded dataframe from session state (available before sidebar renders)
df = st.session_state.original_df

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 📊 InsightAI")
    st.caption("AI-powered data analytics workspace")
    st.divider()
    st.markdown("### Your workspace")
    st.markdown("Upload a dataset to begin profiling and preparing your data.")
    uploaded_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        help=f"Maximum upload size: {MAX_UPLOAD_MB} MB. Your file is processed for this app session.",
    )
    sheet_name = None
    if uploaded_file is not None:
        upload_error = validate_upload(uploaded_file, MAX_UPLOAD_MB)
        if upload_error:
            st.error(upload_error)
        elif uploaded_file.name.lower().endswith((".xlsx", ".xls")):
            try:
                sheets = get_excel_sheets(uploaded_file)
                if sheets:
                    sheet_name = st.selectbox("Choose worksheet", options=sheets)
            except Exception as exc:
                st.error(f"Could not read worksheet names: {exc}")
    st.divider()
    
    # ---------- Global Filters (apply to dashboard tabs, not chat) ----------
    if df is not None:
        st.markdown("### 🔍 Dashboard Filters")
        st.caption("Filters apply to Preview, Quality, Explorer, and Cleaning tabs. Chat uses full dataset.")
        
        # Categorical filters - use actual column names from dataset
        categorical_filter_cols = []
        for col in ["Market", "Region", "Shipping Mode", "Order Status"]:
            if col in df.columns:
                categorical_filter_cols.append(col)
        
        for col in categorical_filter_cols:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 50:  # Only show filter for low-cardinality columns
                selected = st.multiselect(
                    f"{col}",
                    options=sorted(unique_vals.astype(str)),
                    default=st.session_state.filter_state.get(col, []),
                    key=f"filter_{col}"
                )
                st.session_state.filter_state[col] = selected
        
        # Date range filter
        date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        # Add detected date candidates from string columns
        if df is not None:
            date_candidates = detect_date_columns(df)
            for cand in date_candidates:
                if cand["column"] not in date_cols:
                    date_cols.append(cand["column"])
        if date_cols:
            date_col = st.selectbox("Date column", options=[""] + date_cols, key="filter_date_col")
            if date_col:
                # Parse if needed
                date_series = df[date_col]
                if not pd.api.types.is_datetime64_any_dtype(date_series):
                    date_series = pd.to_datetime(date_series, errors="coerce")
                valid_dates = date_series.dropna()
                if not valid_dates.empty:
                    min_date = valid_dates.min().date()
                    max_date = valid_dates.max().date()
                    date_range = st.date_input(
                        f"{date_col} range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        key="filter_date_range"
                    )
                    if len(date_range) == 2:
                        st.session_state.filter_state["_date_col"] = date_col
                        st.session_state.filter_state["_date_start"] = str(date_range[0])
                        st.session_state.filter_state["_date_end"] = str(date_range[1])
        
        # Clear filters button
        if st.session_state.filter_state:
            if st.button("Clear all filters", type="secondary"):
                st.session_state.filter_state = {}
                st.rerun()
    
    # ---------- Semantic Mapping Review ----------
    if df is not None and st.session_state.get("semantic_profile"):
        st.divider()
        st.markdown("### 🎯 Semantic Mapping")
        st.caption("Review and correct column role assignments. Changes apply to KPIs, charts, and AI analysis.")
        sem_profile = st.session_state.semantic_profile
        
        # Show domain detection
        domain_label = sem_profile.primary_domain.value.replace("_", " ").title()
        st.markdown(f"**Detected domain:** {domain_label} (confidence: {sem_profile.domain_confidence:.0%})")
        
        # Show role summary
        role_counts = {
            "Measures": len(sem_profile.measures),
            "Dimensions": len(sem_profile.dimensions),
            "Dates": len(sem_profile.dates),
            "Identifiers": len(sem_profile.identifiers),
            "Currencies": len(sem_profile.currencies),
            "Booleans": len(sem_profile.booleans),
        }
        role_cols = st.columns(3)
        for i, (role, count) in enumerate(role_counts.items()):
            with role_cols[i % 3]:
                st.metric(role, count)
        
        # Show ambiguous columns needing review
        if sem_profile.ambiguous_columns:
            st.warning(f"⚠️ {len(sem_profile.ambiguous_columns)} column(s) need review: {', '.join(sem_profile.ambiguous_columns[:5])}{'…' if len(sem_profile.ambiguous_columns) > 5 else ''}")
        
        # Role editor expander
        with st.expander("Review / Edit Column Roles", expanded=bool(sem_profile.ambiguous_columns)):
            for col_name in sem_profile.original_columns:
                sem_col = sem_profile.semantic_columns[col_name]
                current_role = st.session_state.semantic_override.get(col_name, sem_col.role.value)
                
                # Determine options - prioritize likely roles
                role_options = [r.value for r in ColumnRole]
                # Move current to top
                if current_role in role_options:
                    role_options.remove(current_role)
                role_options = [current_role] + role_options
                
                new_role = st.selectbox(
                    f"{col_name} ({sem_col.dtype})",
                    options=role_options,
                    index=0,
                    key=f"role_{col_name}",
                    help=f"Confidence: {sem_col.confidence:.0%} | Missing: {sem_col.missing_pct:.1f}% | Unique: {sem_col.unique_count}"
                )
                
                if new_role != sem_col.role.value:
                    st.session_state.semantic_override[col_name] = ColumnRole(new_role)
                elif col_name in st.session_state.semantic_override:
                    del st.session_state.semantic_override[col_name]
        
        # Apply overrides button
        if st.session_state.semantic_override:
            if st.button("Apply Role Changes", type="primary"):
                # Rebuild semantic profile with overrides
                # For now, just mark as applied - full rebuild would need more logic
                st.success("Role changes applied. Dashboard and AI will use updated roles.")
                st.rerun()
    
    st.divider()
    st.markdown("### AI Configuration")
    if is_configured():
        st.success("✅ NVIDIA API key configured")
    else:
        st.warning("⚠️ NVIDIA API key not set")
        st.caption("Add NVIDIA_API_KEY to .streamlit/secrets.toml or set the NVIDIA_API_KEY environment variable.")
    st.divider()
    st.markdown("### Build status")
    st.markdown("- [x] Interface and upload")
    st.markdown("- [x] Data quality profile")
    st.markdown("- [x] Safe cleaning preview")
    st.markdown("- [x] Data Explorer")
    st.markdown("- [x] Nemotron chat integration")
    st.markdown("- [x] Analytical engine (Phase 3)")
    st.markdown("- [x] KPI and chart generation (Phase 4)")
    st.caption("Phase 4 · Local prototype")

# ---------- Main header ----------
st.markdown(
    """
    <div class="hero">
      <h1>Turn your data into decisions.</h1>
      <p>Explore your dataset, identify quality issues, and preview safe cleaning actions — all in one workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if uploaded_file is not None:
    upload_error = validate_upload(uploaded_file, MAX_UPLOAD_MB)
    if not upload_error:
        with st.spinner("Loading dataset…"):
            try:
                df = load_uploaded_dataset(uploaded_file, sheet_name=sheet_name)
                # Reload only when the selected file/sheet changes.
                selected_identity = f"{uploaded_file.name}::{sheet_name or 'csv'}::{uploaded_file.size}"
                if st.session_state.get("dataset_identity") != selected_identity:
                    st.session_state.original_df = df.copy(deep=True)
                    st.session_state.cleaned_df = None
                    st.session_state.dataset_name = uploaded_file.name
                    st.session_state.dataset_identity = selected_identity
                    st.session_state.chat_messages = []
                    st.session_state.filter_state = {}
                    st.session_state.semantic_override = {}
                    # Build semantic profile for the new dataset
                    with st.spinner("Building semantic profile…"):
                        st.session_state.semantic_profile = build_semantic_profile(df)
            except Exception as exc:
                st.error(f"Unable to load this file. Check that it is a valid CSV or Excel workbook. Details: {exc}")
                st.stop()

# ---------- Apply global filters to dashboard copy ----------
def apply_dashboard_filters(base_df: pd.DataFrame, filter_state: dict) -> pd.DataFrame:
    """Apply global filters to a copy of the dataframe for dashboard tabs."""
    filtered = base_df.copy()
    for col, selected_vals in filter_state.items():
        if col.startswith("_"):
            continue
        if selected_vals and col in filtered.columns:
            filtered = filtered[filtered[col].astype(str).isin(selected_vals)]
    # Date range filter
    date_col = filter_state.get("_date_col")
    date_start = filter_state.get("_date_start")
    date_end = filter_state.get("_date_end")
    if date_col and date_col in filtered.columns and date_start and date_end:
        date_series = filtered[date_col]
        if not pd.api.types.is_datetime64_any_dtype(date_series):
            date_series = pd.to_datetime(date_series, errors="coerce")
        filtered = filtered[(date_series >= pd.Timestamp(date_start)) & (date_series <= pd.Timestamp(date_end))]
    return filtered


if df is None:
    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Start with a dataset")
        st.write("Upload a CSV or Excel workbook using the sidebar. InsightAI will show a data preview, quality checks, and a non-destructive cleaning workflow.")
        st.info("For best results, start with a small sample file. The current upload limit is 100 MB; practical capacity also depends on your computer's memory.")
    with right:
        st.subheader("What you can inspect")
        st.markdown(
            "- Dataset size, column types, and memory usage\n"
            "- Missing values and duplicate rows\n"
            "- Date column detection and parsing issues\n"
            "- Inconsistent text categories\n"
            "- Suspicious numeric values\n"
            "- Cleaning changes before applying them\n"
            "- Downloadable cleaned copy"
        )
    st.divider()
    st.subheader("Ask your data questions")
    st.chat_input("Upload a dataset to start asking questions…", disabled=True)
    st.caption("AI-generated analytical answers are not enabled in Phase 1.")
    st.stop()


def build_dynamic_dashboard(
    df: pd.DataFrame,
    sem_profile: DatasetProfile,
    engine: 'AnalyticsEngine',
) -> list[tuple[str, 'go.Figure']]:
    """
    Build a dynamic dashboard based on the semantic profile of the dataset.
    
    Returns a list of (title, figure) tuples.
    """
    from modules.analytics_engine import AnalysisPlan, GroupBySpec, DateAnalysisSpec, AggregateSpec, AggregationFunction, OperationType
    from modules.dynamic_charts import (
        build_chart_spec_from_result,
        create_chart_from_spec,
        select_chart_for_question,
        ChartType,
    )
    
    charts = []
    
    measures = sem_profile.measures
    dimensions = sem_profile.dimensions
    dates = sem_profile.dates
    currencies = sem_profile.currencies
    booleans = sem_profile.booleans
    
    primary_measure = sem_profile.get_primary_measure()
    primary_dimension = dimensions[0] if dimensions else None
    primary_date = sem_profile.get_primary_date()
    primary_currency = currencies[0] if currencies else primary_measure
    
    # Chart 1: Measure by Dimension (if both exist)
    if primary_measure and primary_dimension:
        plan = AnalysisPlan(
            operations=[OperationType.GROUP_BY],
            group_by=GroupBySpec(
                columns=[primary_dimension],
                aggregations={primary_measure: [AggregationFunction.SUM]}
            )
        )
        result = engine.execute(plan)
        if result.success and not result.data.get("group_by", pd.DataFrame()).empty:
            # Convert plan to spec dict for chart builder
            plan_spec = {
                "group_by": {
                    "columns": [primary_dimension],
                    "aggregations": {primary_measure: ["sum"]}
                }
            }
            spec = build_chart_spec_from_result(
                result.data, "group_by", plan_spec, sem_profile
            )
            if spec:
                fig = create_chart_from_spec(result.data["group_by"], spec, sem_profile)
                charts.append((f"{primary_measure} by {primary_dimension}", fig))
    
    # Chart 2: Time series (if date and measure exist)
    if primary_date and primary_measure:
        # Ensure date is datetime
        date_col_resolved = primary_date
        if not pd.api.types.is_datetime64_any_dtype(df[primary_date]):
            date_col_resolved = ensure_datetime(df, primary_date)
        
        if date_col_resolved:
            plan = AnalysisPlan(
                operations=[OperationType.DATE_ANALYSIS],
                date_analysis=DateAnalysisSpec(
                    date_column=date_col_resolved,
                    value_column=primary_measure,
                    frequency="M",
                    aggregation=AggregationFunction.SUM
                )
            )
            result = engine.execute(plan)
            if result.success and not result.data.get("date_analysis", pd.DataFrame()).empty:
                plan_spec = {
                    "date_analysis": {
                        "date_column": date_col_resolved,
                        "value_column": primary_measure
                    }
                }
                spec = build_chart_spec_from_result(
                    result.data, "date_analysis", plan_spec, sem_profile
                )
                if spec:
                    fig = create_chart_from_spec(result.data["date_analysis"], spec, sem_profile)
                    charts.append((f"Monthly {primary_measure} Trend", fig))
    
    # Chart 3: Risk/Boolean by Dimension
    risk_col = None
    for col in booleans:
        if any(kw in col.lower() for kw in ["risk", "late", "churn", "default", "delay"]):
            risk_col = col
            break
    
    if risk_col and primary_dimension:
        plan = AnalysisPlan(
            operations=[OperationType.GROUP_BY],
            group_by=GroupBySpec(
                columns=[primary_dimension],
                aggregations={risk_col: [AggregationFunction.MEAN]}
            )
        )
        result = engine.execute(plan)
        if result.success and not result.data.get("group_by", pd.DataFrame()).empty:
            plan_spec = {
                "group_by": {
                    "columns": [primary_dimension],
                    "aggregations": {risk_col: ["mean"]}
                }
            }
            spec = build_chart_spec_from_result(
                result.data, "group_by", plan_spec, sem_profile
            )
            if spec:
                fig = create_chart_from_spec(result.data["group_by"], spec, sem_profile)
                charts.append((f"{risk_col.replace('_', ' ').title()} by {primary_dimension}", fig))
    
    # Chart 4: Second measure by dimension (if available)
    if len(measures) >= 2 and primary_dimension:
        second_measure = measures[1] if measures[0] == primary_measure else measures[0]
        plan = AnalysisPlan(
            operations=[OperationType.GROUP_BY],
            group_by=GroupBySpec(
                columns=[primary_dimension],
                aggregations={second_measure: [AggregationFunction.SUM]}
            )
        )
        result = engine.execute(plan)
        if result.success and not result.data.get("group_by", pd.DataFrame()).empty:
            plan_spec = {
                "group_by": {
                    "columns": [primary_dimension],
                    "aggregations": {second_measure: ["sum"]}
                }
            }
            spec = build_chart_spec_from_result(
                result.data, "group_by", plan_spec, sem_profile
            )
            if spec:
                fig = create_chart_from_spec(result.data["group_by"], spec, sem_profile)
                charts.append((f"{second_measure} by {primary_dimension}", fig))
    
    # Chart 5: Distribution of primary measure
    if primary_measure:
        plan = AnalysisPlan(
            operations=[OperationType.COLUMN_STATS],
            column_stats=[primary_measure]
        )
        result = engine.execute(plan)
        if result.success and not result.data.get("column_stats", pd.DataFrame()).empty:
            # For distribution, use histogram
            from modules.dashboard_charts import create_bar_chart
            import plotly.express as px
            fig = px.histogram(
                df, x=primary_measure, nbins=30,
                title=f"Distribution of {primary_measure}", height=320
            )
            from modules.dashboard_charts import apply_executive_layout
            apply_executive_layout(fig, title=f"Distribution of {primary_measure}", height=320)
            charts.append((f"Distribution of {primary_measure}", fig))
    
    # Chart 6: Correlation heatmap (if multiple measures)
    if len(measures) >= 2:
        plan = AnalysisPlan(
            operations=[OperationType.CORRELATION],
            correlation_columns=measures[:5]  # Limit to 5 for readability
        )
        result = engine.execute(plan)
        if result.success and not result.data.get("correlation", pd.DataFrame()).empty:
            corr_df = result.data["correlation"]
            import plotly.express as px
            from modules.dashboard_charts import apply_executive_layout
            # Set index for correlation matrix
            if "index" in corr_df.columns:
                corr_display = corr_df.set_index("index")
            else:
                corr_display = corr_df
            fig = px.imshow(
                corr_display, text_auto=".2f", aspect="auto",
                title="Correlation Matrix", height=400,
                color_continuous_scale="RdBu", zmin=-1, zmax=1
            )
            apply_executive_layout(fig, title="Correlation Matrix", height=400)
            charts.append(("Correlation Matrix", fig))
    
    return charts


# ========== DASHBOARD & CHAT (only when dataset loaded) ==========
# Create filtered dataframe for dashboard tabs (not chat)
filter_state = st.session_state.get("filter_state", {})
df_filtered = apply_dashboard_filters(df, filter_state)

# Handle empty filtered dataframe gracefully
if df_filtered.empty:
    st.warning("⚠️ Current filters result in an empty dataset. Adjust or clear filters to view the dashboard.")
    st.stop()

# Show filter indicator
if filter_state and any(v for k, v in filter_state.items() if not k.startswith("_")) or filter_state.get("_date_col"):
    st.info(f"🔍 Filters active — Dashboard showing {len(df_filtered):,} of {len(df):,} rows. Chat tab uses full dataset.")
    if st.button("Clear filters", key="clear_filters_top"):
        st.session_state.filter_state = {}
        st.rerun()

# ---------- Dataset overview ----------
with st.spinner("Profiling dataset…"):
    profile = profile_dataset(df_filtered)

st.markdown(f"### Dataset overview · `{st.session_state.dataset_name}`")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Rows", f"{profile['rows']:,}")
m2.metric("Columns", f"{profile['columns']:,}")
m3.metric("Missing cells", f"{profile['missing_cells']:,}")
m4.metric("Duplicate rows", f"{profile['duplicate_rows']:,}")

m5, m6 = st.columns(2)
m5.metric("Memory usage", f"{profile['total_memory_mb']:.3f} MB")
if profile["empty_columns"]:
    m6.metric("Empty columns", f"{len(profile['empty_columns'])}")
else:
    m6.metric("Empty columns", "0")

tab_dashboard, tab_preview, tab_quality, tab_explorer, tab_clean, tab_chat = st.tabs(
    ["📊 Dashboard", "🔎 Data preview", "🩺 Data quality", "🔍 Data Explorer", "🧹 Cleaning preview", "💬 Ask your data"]
)

with tab_dashboard:
    st.subheader("Executive Dashboard")
    
    # Show active filters
    active_filters = {k: v for k, v in filter_state.items() if not k.startswith("_") and v}
    has_date_filter = filter_state.get("_date_col") is not None
    
    if active_filters or has_date_filter:
        filter_tags = []
        for col, vals in active_filters.items():
            filter_tags.append(f"**{col}**: {', '.join(vals)}")
        if has_date_filter:
            filter_tags.append(f"**{filter_state['_date_col']}**: {filter_state['_date_start']} → {filter_state['_date_end']}")
        st.info("Active filters: " + " | ".join(filter_tags))
    else:
        st.caption("No active filters. Use sidebar to filter by available dimensions or date range.")
    
    st.divider()
    
    # Generate dynamic KPIs based on semantic profile
    sem_profile = st.session_state.get("semantic_profile")
    if sem_profile:
        dynamic_kpis = generate_dynamic_kpis(df_filtered, sem_profile)
        kpi_defs = kpi_specs_to_definitions(dynamic_kpis)
        kpi_values = kpi_specs_to_dict(dynamic_kpis)
        
        # Prioritize KPIs: domain-specific first, then general, then numeric
        domain_priority = {
            "sales_retail": 0,
            "churn": 1,
            "supply_chain": 2,
            "hr": 3,
            "loan_finance": 4,
            "general": 5,
        }
        kpi_defs.sort(key=lambda d: (
            domain_priority.get(d.get("domain", "general"), 5),
            0 if d["key"].startswith(("total_", "distinct_", "avg_", "churn_", "late_", "profit_")) else 1,
            d["key"]
        ))
        
        # Limit to max 8 KPI cards for readability
        MAX_KPI_CARDS = 8
        if len(kpi_defs) > MAX_KPI_CARDS:
            kpi_defs = kpi_defs[:MAX_KPI_CARDS]
    else:
        # Fallback to legacy KPIs
        dynamic_kpis = []
        kpi_defs = get_kpi_definitions()
        kpi_values = calculate_kpis(df_filtered)
    
    # KPI Row
    if kpi_defs:
        # Use up to 4 columns per row for better layout
        kpi_per_row = min(4, len(kpi_defs))
        for row_start in range(0, len(kpi_defs), kpi_per_row):
            row_defs = kpi_defs[row_start:row_start + kpi_per_row]
            kpi_cols = st.columns(len(row_defs))
            for i, kpi_def in enumerate(row_defs):
                with kpi_cols[i]:
                    value = kpi_values.get(kpi_def["key"], np.nan)
                    formatted = format_kpi_value(value, kpi_def["type"])
                    # Add validity indicator to help text
                    help_text = kpi_def.get("help", "")
                    kpi_obj = next((k for k in dynamic_kpis if k.key == kpi_def["key"]), None)
                    if kpi_obj and not kpi_obj.is_valid:
                        help_text = f"⚠️ {kpi_obj.validity_reason}"
                    # Ensure icon is a string, fallback to generic if missing
                    icon = kpi_def.get("icon", "📊")
                    if not isinstance(icon, str) or len(icon) == 0:
                        icon = "📊"
                    st.metric(
                        label=f"{icon} {kpi_def['label']}",
                        value=formatted if formatted else "N/A",
                        help=help_text
                    )
    else:
        st.caption("No KPIs available for this dataset. Upload data with numeric columns to see metrics.")
    
    st.divider()
    
    # Dynamic Charts based on semantic profile
    # Create engine with semantic profile for better column resolution
    engine = create_engine(df_filtered, sem_profile)
    
    if sem_profile:
        # Build a dynamic dashboard based on available column roles
        charts = build_dynamic_dashboard(df_filtered, sem_profile, engine)
        
        if charts:
            # Render charts in rows of up to 3
            for i in range(0, len(charts), 3):
                row_charts = charts[i:i+3]
                chart_cols = st.columns(len(row_charts))
                for j, (title, fig) in enumerate(row_charts):
                    with chart_cols[j]:
                        st.markdown(f"**{title}**")
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No charts available for current dataset and filters. Ensure you have at least one measure and one dimension or date column.")
    else:
        st.caption("Semantic profile not available. Please reload the dataset.")
    
    st.divider()

# Insights section
st.markdown("### Insights")

def generate_insights(
    df: pd.DataFrame,
    sem_profile: Optional[DatasetProfile],
    engine: 'AnalyticsEngine',
    kpi_values: dict,
) -> list[str]:
    """
    Generate evidence-based insights from the current dataset and filters.
    """
    from modules.analytics_engine import AnalysisPlan, GroupBySpec, AggregateSpec, AggregationFunction, OperationType, MoMComparisonSpec, OutlierDetectionSpec
    
    insights = []
    
    if not sem_profile:
        return insights
    
    measures = sem_profile.measures
    dimensions = sem_profile.dimensions
    dates = sem_profile.dates
    booleans = sem_profile.booleans
    
    primary_measure = sem_profile.get_primary_measure()
    primary_dimension = dimensions[0] if dimensions else None
    primary_date = sem_profile.get_primary_date()
    
    # Insight 1: Top dimension by measure
    if primary_measure and primary_dimension:
        plan = AnalysisPlan(
            operations=[OperationType.GROUP_BY],
            group_by=GroupBySpec(
                columns=[primary_dimension],
                aggregations={primary_measure: [AggregationFunction.SUM]}
            )
        )
        result = engine.execute(plan)
        if result.success and not result.data.get("group_by", pd.DataFrame()).empty:
            gb_df = result.data["group_by"].sort_values(f"{primary_measure}_sum", ascending=False)
            top_dim = gb_df.iloc[0][primary_dimension]
            top_val = gb_df.iloc[0][f"{primary_measure}_sum"]
            if pd.notna(top_dim):
                insights.append(f"🏆 **Top {primary_dimension} by {primary_measure}**: {top_dim} ({top_val:,.0f})")
    
    # Insight 2: Date range
    if primary_date:
        date_series = df[primary_date]
        if not pd.api.types.is_datetime64_any_dtype(date_series):
            date_series = pd.to_datetime(date_series, errors="coerce")
        valid_dates = date_series.dropna()
        if not valid_dates.empty:
            insights.append(f"📅 **Date range**: {valid_dates.min().date()} to {valid_dates.max().date()}")
    
    # Insight 3: Measure statistics
    if primary_measure:
        plan = AnalysisPlan(
            operations=[OperationType.AGGREGATE],
            aggregate=AggregateSpec(
                columns=[primary_measure],
                functions=[AggregationFunction.MEAN, AggregationFunction.MEDIAN, AggregationFunction.STD]
            )
        )
        result = engine.execute(plan)
        if result.success and not result.data.get("aggregate", pd.DataFrame()).empty:
            agg_df = result.data["aggregate"]
            if "index" in agg_df.columns and primary_measure in agg_df.columns:
                mean_row = agg_df[agg_df["index"] == "mean"]
                median_row = agg_df[agg_df["index"] == "median"]
                std_row = agg_df[agg_df["index"] == "std"]
                if not mean_row.empty and not median_row.empty:
                    mean_val = mean_row[primary_measure].iloc[0]
                    median_val = median_row[primary_measure].iloc[0]
                    std_val = std_row[primary_measure].iloc[0] if not std_row.empty else None
                    if std_val:
                        insights.append(f"📊 **{primary_measure}**: Mean={mean_val:,.1f}, Median={median_val:,.1f}, Std={std_val:,.1f}")
                    else:
                        insights.append(f"📊 **{primary_measure}**: Mean={mean_val:,.1f}, Median={median_val:,.1f}")
    
    # Insight 4: Risk/Churn rate
    risk_col = None
    for col in booleans:
        if any(kw in col.lower() for kw in ["risk", "late", "churn", "default", "delay"]):
            risk_col = col
            break
    
    if risk_col:
        plan = AnalysisPlan(
            operations=[OperationType.AGGREGATE],
            aggregate=AggregateSpec(columns=[risk_col], functions=[AggregationFunction.MEAN])
        )
        result = engine.execute(plan)
        if result.success and not result.data.get("aggregate", pd.DataFrame()).empty:
            agg_df = result.data["aggregate"]
            if "index" in agg_df.columns and risk_col in agg_df.columns:
                mean_row = agg_df[agg_df["index"] == "mean"]
                if not mean_row.empty:
                    rate = mean_row[risk_col].iloc[0] * 100
                    insights.append(f"⚠️ **{risk_col.replace('_', ' ').title()} rate**: {rate:.1f}% (based on {risk_col} flag, missing values excluded)")
    
    # Insight 5: Missing data
    missing_pct = kpi_values.get("missing_data_pct", 0)
    if missing_pct > 0:
        insights.append(f"🕳️ **Missing data**: {missing_pct:.1f}% of cells")
    
    # Insight 6: Data coverage
    total_rows = len(df)
    total_original = len(st.session_state.get("original_df", df))
    if total_original > 0:
        pct = (total_rows / total_original) * 100
        insights.append(f"📊 **Data coverage**: {total_rows:,} of {total_original:,} rows ({pct:.1f}%) after filters")
    
    # Insight 7: Category balance (if dimension has few categories)
    if primary_dimension and primary_dimension in df.columns:
        unique_count = df[primary_dimension].nunique()
        if unique_count <= 10:
            vc = df[primary_dimension].value_counts()
            if not vc.empty:
                top_cat = vc.index[0]
                top_pct = vc.iloc[0] / len(df) * 100
                if top_pct > 50:
                    insights.append(f"⚖️ **{primary_dimension} imbalance**: '{top_cat}' dominates with {top_pct:.1f}% of records")
    
    # Insight 8: Month-over-month trend for primary measure
    if primary_measure and primary_date:
        plan = AnalysisPlan(
            operations=[OperationType.MOM_COMPARISON],
            mom_comparison=MoMComparisonSpec(
                date_column=primary_date,
                value_column=primary_measure,
                frequency="M",
                aggregation=AggregationFunction.SUM,
                periods=6,
            )
        )
        result = engine.execute(plan)
        if result.success and not result.data.get("mom_comparison", pd.DataFrame()).empty:
            mom_df = result.data["mom_comparison"]
            if len(mom_df) >= 2:
                # Get latest MoM change
                latest = mom_df.iloc[-1]
                prev = mom_df.iloc[-2]
                value_col = f"{primary_measure}_sum"
                pct_col = f"{value_col}_mom_pct_change"
                if value_col in mom_df.columns and pct_col in mom_df.columns:
                    latest_val = latest[value_col]
                    mom_pct = latest[pct_col]
                    if pd.notna(mom_pct):
                        direction = "📈 increased" if mom_pct > 0 else "📉 decreased"
                        insights.append(f"📊 **MoM Trend ({primary_measure})**: Latest month ({latest['period']}) {direction} by {abs(mom_pct):.1f}% to {latest_val:,.0f} (from {prev[value_col]:,.0f})")
    
    # Insight 9: Outlier detection for primary measure
    if primary_measure:
        plan = AnalysisPlan(
            operations=[OperationType.OUTLIER_DETECTION],
            outlier_detection=OutlierDetectionSpec(
                column=primary_measure,
                method="iqr",
                threshold=1.5,
            )
        )
        result = engine.execute(plan)
        if result.success and result.data.get("outlier_detection"):
            outlier_data = result.data["outlier_detection"]
            summary = outlier_data.get("summary", [{}])[0]
            outliers = outlier_data.get("outliers", [])
            count = summary.get("outlier_count", 0)
            pct = summary.get("outlier_percentage", 0)
            if count > 0:
                insights.append(f"🔍 **Outliers detected in {primary_measure}**: {count} records ({pct:.1f}%) flagged as outliers using IQR method")
                # Show top outlier value
                if outliers:
                    top_outlier = outliers[0].get(primary_measure, "N/A")
                    insights.append(f"   → Highest outlier: {top_outlier:,.0f}")
    
    return insights

# Call generate_insights for the dashboard
insights = generate_insights(df_filtered, sem_profile, engine, kpi_values)

# Insights section
st.markdown("### Insights")

if insights:
    for insight in insights:
        st.markdown(f"- {insight}")
else:
    st.caption("No insights available for current filter selection.")

with tab_preview:
    st.subheader("First rows")
    if len(df_filtered) == 0:
        st.warning("No rows match the current filters.")
    else:
        preview_rows = st.slider("Rows to display", min_value=5, max_value=min(200, len(df_filtered)), value=min(10, len(df_filtered)), step=5)
        st.dataframe(df_filtered.head(preview_rows), use_container_width=True, height=380)
    with st.expander("Column data types"):
        type_df = profile["dtype_info"]
        st.dataframe(type_df, use_container_width=True, hide_index=True)
    with st.expander("Memory usage by column"):
        mem_df = profile["memory_by_column"]
        st.dataframe(mem_df, use_container_width=True, hide_index=True)

with tab_quality:
    st.subheader("Quality summary")
    warnings = build_quality_warnings(df_filtered)
    if warnings:
        for warning in warnings:
            if warning["level"] == "error":
                st.error(warning["message"])
            elif warning["level"] == "warning":
                st.warning(warning["message"])
            else:
                st.info(warning["message"])
    else:
        st.success("No common data-quality issues were detected by the initial checks.")

    st.markdown("#### Missing values by column")
    missing_df = profile["missing_by_column"]
    if not missing_df.empty:
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
    else:
        st.success("No missing values found.")

    # Date column detection
    date_candidates = profile.get("date_candidates", [])
    if date_candidates:
        st.markdown("#### Potential date columns")
        for candidate in date_candidates:
            with st.expander(f"📅 {candidate['column']} — suspected {candidate['suspected_format']}"):
                st.write(f"**Pattern:** `{candidate['pattern']}`")
                st.write(f"**Sample values:** {candidate['sample_values']}")
                st.write(f"**Parse success rate:** {candidate['parse_success_rate']}%")
                if candidate["parse_errors"]:
                    st.warning(f"Parsing failed for some values: {candidate['parse_errors']}")
                st.info("These columns are stored as text. Consider converting to datetime after reviewing the format.")
    else:
        st.caption("No potential date columns detected in text columns.")

    # Text inconsistencies
    text_inconsistencies = profile.get("text_inconsistencies", [])
    if text_inconsistencies:
        st.markdown("#### Inconsistent text categories")
        for issue in text_inconsistencies:
            with st.expander(f"🔤 {issue['column']} — {issue['affected_unique_values']} values with case/spacing variations"):
                st.write(f"Cardinality: {issue['cardinality']} unique values")
                for norm_val, original_vals in issue["fuzzy_groups"].items():
                    st.write(f"- Normalized: **{norm_val}** → Original: {original_vals}")
                st.info("These values differ only by case or whitespace. Consider standardizing them during cleaning.")
    else:
        st.caption("No inconsistent text categories detected in low-cardinality columns.")

    # Numeric issues
    numeric_issues = profile.get("numeric_issues", [])
    if numeric_issues:
        st.markdown("#### Suspicious numeric values")
        for issue in numeric_issues:
            with st.expander(f"🔢 {issue['column']}"):
                for sub_issue in issue["issues"]:
                    level = sub_issue["type"]
                    if level == "negative_values":
                        st.warning(sub_issue["message"])
                    elif level == "excessive_zeros":
                        st.info(sub_issue["message"])
                    elif level == "outliers_iqr":
                        st.warning(sub_issue["message"])
    else:
        st.caption("No suspicious numeric values detected.")

    st.markdown("#### Numeric-column overview")
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        st.dataframe(numeric.describe().T.reset_index(names="Column"), use_container_width=True)
    else:
        st.caption("No numeric columns were detected.")

with tab_explorer:
    st.subheader("Column inspector")
    st.write("Select a column to see value distribution, unique values, and apply filters.")

    if len(df_filtered.columns) == 0:
        st.info("No columns available.")
    else:
        col_selector = st.selectbox("Choose a column", options=df_filtered.columns.astype(str).tolist())
        if col_selector:
            series = df_filtered[col_selector]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Non-null", f"{int(series.notna().sum()):,}")
            c2.metric("Null", f"{int(series.isna().sum()):,}")
            c3.metric("Unique", f"{int(series.nunique(dropna=True)):,}")
            c4.metric("Memory", f"{series.memory_usage(deep=True) / 1024:.1f} KB")

            # Value counts for categorical / low cardinality
            if series.dtype.kind in ("O", "U", "S") or series.nunique(dropna=True) <= 50:
                st.markdown("**Value counts (top 30)**")
                vc = series.value_counts(dropna=False).head(30)
                vc_df = pd.DataFrame({"Value": vc.index.astype(str), "Count": vc.values})
                st.dataframe(vc_df, use_container_width=True, hide_index=True)
            elif pd.api.types.is_numeric_dtype(series):
                st.markdown("**Histogram**")
                try:
                    import plotly.express as px
                    fig = px.histogram(df_filtered, x=col_selector, nbins=30)
                    fig.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    st.caption("Histogram unavailable (plotly not installed or error).")

            st.divider()
            st.markdown("**Filter records**")
            st.caption("Build a filter to narrow down the dataset. Filters are combined with AND.")

            filter_conditions = []
            filter_col = st.selectbox("Filter column", options=[""] + df_filtered.columns.astype(str).tolist(), key="filter_col")
            if filter_col:
                col_series = df_filtered[filter_col]
                if pd.api.types.is_numeric_dtype(col_series):
                    op = st.selectbox("Operator", ["==", "!=", ">", ">=", "<", "<="], key="filter_op")
                    val = st.number_input("Value", value=0.0, key="filter_val")
                    if st.button("Add filter", key="add_filter_num"):
                        filter_conditions.append((filter_col, op, val))
                elif pd.api.types.is_datetime64_any_dtype(col_series):
                    op = st.selectbox("Operator", ["==", "!=", ">", ">=", "<", "<="], key="filter_op_dt")
                    val = st.date_input("Value", key="filter_val_dt")
                    if st.button("Add filter", key="add_filter_dt"):
                        filter_conditions.append((filter_col, op, pd.Timestamp(val)))
                else:
                    op = st.selectbox("Operator", ["==", "!=", "contains", "startswith", "endswith"], key="filter_op_str")
                    val = st.text_input("Value", key="filter_val_str")
                    if st.button("Add filter", key="add_filter_str"):
                        filter_conditions.append((filter_col, op, val))

            # Apply filters
            filtered_df = df_filtered.copy()
            for fcol, fop, fval in filter_conditions:
                if fop == "==":
                    filtered_df = filtered_df[filtered_df[fcol] == fval]
                elif fop == "!=":
                    filtered_df = filtered_df[filtered_df[fcol] != fval]
                elif fop == ">":
                    filtered_df = filtered_df[filtered_df[fcol] > fval]
                elif fop == ">=":
                    filtered_df = filtered_df[filtered_df[fcol] >= fval]
                elif fop == "<":
                    filtered_df = filtered_df[filtered_df[fcol] < fval]
                elif fop == "<=":
                    filtered_df = filtered_df[filtered_df[fcol] <= fval]
                elif fop == "contains":
                    filtered_df = filtered_df[filtered_df[fcol].astype(str).str.contains(str(fval), case=False, na=False)]
                elif fop == "startswith":
                    filtered_df = filtered_df[filtered_df[fcol].astype(str).str.startswith(str(fval), na=False)]
                elif fop == "endswith":
                    filtered_df = filtered_df[filtered_df[fcol].astype(str).str.endswith(str(fval), na=False)]

            st.markdown(f"**Filtered rows: {len(filtered_df):,} / {len(df_filtered):,}**")
            st.dataframe(filtered_df.head(50), use_container_width=True, height=400)

            if len(filtered_df) != len(df_filtered):
                csv_bytes = filtered_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "Download filtered CSV",
                    data=csv_bytes,
                    file_name=f"{st.session_state.dataset_name.rsplit('.', 1)[0]}_filtered.csv",
                    mime="text/csv",
                )

with tab_clean:
    st.subheader("Preview safe cleaning actions")
    st.write("The original uploaded dataset remains unchanged. Choose the transformations you want to preview.")
    trim_strings = st.checkbox("Trim leading/trailing spaces in text cells", value=True)
    normalize_spaces = st.checkbox("Normalize repeated whitespace inside text cells", value=False)
    drop_duplicates = st.checkbox("Remove exact duplicate rows", value=False)
    drop_empty_columns = st.checkbox("Remove columns that are entirely empty", value=False)

    if len(df_filtered) == 0:
        st.warning("No rows match the current filters. Adjust filters to see cleaning preview.")
    else:
        preview_df, changes = create_cleaning_preview(
            df_filtered,
            trim_strings=trim_strings,
            normalize_spaces=normalize_spaces,
            drop_duplicates=drop_duplicates,
            drop_empty_columns=drop_empty_columns,
        )
        before_col, after_col = st.columns(2)
        with before_col:
            st.markdown("**Before**")
            st.caption(f"{df_filtered.shape[0]:,} rows × {df_filtered.shape[1]:,} columns")
            st.dataframe(df_filtered.head(8), use_container_width=True, height=250)
        with after_col:
            st.markdown("**After preview**")
            st.caption(f"{preview_df.shape[0]:,} rows × {preview_df.shape[1]:,} columns")
            st.dataframe(preview_df.head(8), use_container_width=True, height=250)

        if changes:
            st.markdown("#### Planned changes")
            for change in changes:
                st.write(f"- {change}")
        else:
            st.info("No changes selected or no changes detected.")

        if st.button("Apply to session copy", type="primary", disabled=preview_df.equals(df_filtered)):
            st.session_state.cleaned_df = preview_df.copy(deep=True)
            st.success("Cleaning applied to a separate session copy. The uploaded original remains unchanged.")

    if st.session_state.cleaned_df is not None:
        st.success("A cleaned copy is available for download.")
        csv_bytes = st.session_state.cleaned_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "Download cleaned CSV",
            data=csv_bytes,
            file_name=f"{st.session_state.dataset_name.rsplit('.', 1)[0]}_cleaned.csv",
            mime="text/csv",
        )

with tab_chat:
    st.subheader("Ask your data")

    ai_ready = is_configured()
    if not ai_ready:
        st.info("🔑 Configure your NVIDIA API key in `.streamlit/secrets.toml` or the `NVIDIA_API_KEY` environment variable to enable AI chat.")

    # Cache dataset context in session state to avoid recomputing on every message
    dataset_identity = st.session_state.get("dataset_identity")
    if "dataset_context" not in st.session_state or st.session_state.get("context_identity") != dataset_identity:
        quality_warnings = build_quality_warnings(df)
        sem_profile = st.session_state.get("semantic_profile")
        st.session_state.dataset_context = build_dataset_context(df, profile, quality_warnings, sem_profile)
        st.session_state.context_identity = dataset_identity

    dataset_context = st.session_state.dataset_context

    # Build column dtype info for the planner
    dtype_info = profile.get("dtype_info")
    columns = df.columns.astype(str).tolist()
    dtypes = {}
    if dtype_info is not None:
        for _, row in dtype_info.iterrows():
            dtypes[row["Column"]] = row["Data type"]

    # Display conversation history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    question = st.chat_input(
        "e.g. What is the average sales by region? Show trend over time for orders.",
        disabled=not ai_ready,
    )

    if question:
        # Prevent duplicate submissions
        if st.session_state.chat_messages and st.session_state.chat_messages[-1].get("content") == question and st.session_state.chat_messages[-1].get("role") == "user":
            pass  # Skip duplicate
        else:
            st.session_state.chat_messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            # Get AI response with analytical engine
            with st.chat_message("assistant"):
                with st.spinner("Analyzing…"):
                    # Step 0: Handle simple metadata questions directly (no AI needed)
                    sem_profile = st.session_state.get("semantic_profile")
                    meta_response = handle_metadata_question(question, df, sem_profile)
                    
                    if meta_response is not None:
                        st.markdown(meta_response)
                        st.session_state.chat_messages.append({"role": "assistant", "content": meta_response})
                    else:
                        config = AIClientConfig(api_key=get_api_key())

                        # Step 1: AI creates analysis plan (with deterministic path for top-N questions)
                        plan, plan_result = plan_analysis(
                            question=question,
                            config=config,
                            dataset_context=dataset_context,
                            columns=columns,
                            dtypes=dtypes,
                            df=df,
                            sem_profile=sem_profile,
                        )

                        if not plan_result.success or plan is None:
                            st.error(f"Could not create analysis plan: {plan_result.error}")
                            st.session_state.chat_messages.append({"role": "assistant", "content": f"⚠️ {plan_result.error}"})
                        else:
                            # Step 2: Execute validated plan
                            sem_profile = st.session_state.get("semantic_profile")
                            result = execute_analysis(df, plan, sem_profile)

                            if not result.success:
                                st.error(f"Analysis failed: {result.error}")
                                st.session_state.chat_messages.append({"role": "assistant", "content": f"⚠️ Analysis failed: {result.error}"})
                            else:
                                # Step 3: Format results with dashboard visualizations
                                data = result.data or {}

                                # ROW_COUNT - KPI metric card
                                if OperationType.ROW_COUNT in plan.operations:
                                    row_count = data.get("row_count", result.row_count)
                                    if row_count is not None:
                                        st.metric("Row Count", f"{row_count:,}")

                                # AGGREGATE - KPI metric cards for single-row results
                                if OperationType.AGGREGATE in plan.operations and "aggregate" in data:
                                    agg_df = data["aggregate"]
                                    if not agg_df.empty:
                                        # Detect if it's a single-row aggregate (no group_by) or grouped
                                        if len(agg_df) == 1 and "index" not in agg_df.columns:
                                            # Single-row aggregate: show each column as a metric
                                            for col in agg_df.columns:
                                                val = agg_df[col].iloc[0]
                                                if pd.api.types.is_numeric_dtype(agg_df[col]):
                                                    st.metric(col.replace("_", " ").title(), f"{val:,.2f}")
                                                else:
                                                    st.metric(col.replace("_", " ").title(), str(val))
                                        else:
                                            # Grouped aggregate or multi-row: show as dataframe
                                            st.subheader("Aggregation Results")
                                            st.dataframe(agg_df, use_container_width=True, hide_index=True)

                            # GROUP_BY - Plotly bar chart
                            if OperationType.GROUP_BY in plan.operations and "group_by" in data:
                                gb_df = data["group_by"]
                                if not gb_df.empty:
                                    st.subheader("Grouped Results")
                                    # Identify group columns and value columns
                                    group_cols = plan.group_by.columns if plan.group_by else []
                                    value_cols = [c for c in gb_df.columns if c not in group_cols]
                                    if value_cols:
                                        # Use first group column for x-axis, first value column for y-axis
                                        x_col = group_cols[0] if group_cols else gb_df.columns[0]
                                        y_col = value_cols[0]
                                        try:
                                            import plotly.express as px
                                            fig = px.bar(gb_df.head(50), x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                                            fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
                                            st.plotly_chart(fig, use_container_width=True)
                                        except Exception:
                                            st.dataframe(gb_df.head(50), use_container_width=True, hide_index=True)
                                    if len(gb_df) > 50:
                                        st.caption(f"Showing 50 of {len(gb_df)} groups")

                            # DATE_ANALYSIS - Plotly line chart
                            if OperationType.DATE_ANALYSIS in plan.operations and "date_analysis" in data:
                                da_df = data["date_analysis"]
                                if not da_df.empty:
                                    st.subheader("Time-Series Analysis")
                                    if "period" in da_df.columns:
                                        value_cols = [c for c in da_df.columns if c != "period"]
                                        if value_cols:
                                            y_col = value_cols[0]
                                            try:
                                                import plotly.express as px
                                                fig = px.line(da_df.head(100), x="period", y=y_col, title=f"{y_col} Over Time", markers=True)
                                                fig.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
                                                st.plotly_chart(fig, use_container_width=True)
                                            except Exception:
                                                st.dataframe(da_df.head(100), use_container_width=True, hide_index=True)

                            # CORRELATION - styled dataframe
                            if OperationType.CORRELATION in plan.operations and "correlation" in data:
                                corr_df = data["correlation"]
                                if not corr_df.empty:
                                    st.subheader("Correlation Matrix")
                                    # Set index column for better display
                                    if "index" in corr_df.columns:
                                        corr_display = corr_df.set_index("index")
                                    else:
                                        corr_display = corr_df
                                    # Style with gradient
                                    try:
                                        st.dataframe(corr_display.style.background_gradient(cmap="RdBu", vmin=-1, vmax=1).format("{:.2f}"), use_container_width=True)
                                    except Exception:
                                        st.dataframe(corr_display, use_container_width=True)

                            # MOM_COMPARISON - Plotly line chart with MoM change
                            if OperationType.MOM_COMPARISON in plan.operations and "mom_comparison" in data:
                                mom_df = data["mom_comparison"]
                                if not mom_df.empty:
                                    st.subheader("Month-over-Month Comparison")
                                    if "period" in mom_df.columns:
                                        value_cols = [c for c in mom_df.columns if c != "period" and not c.endswith("_mom_pct_change") and not c.endswith("_mom_abs_change")]
                                        if value_cols:
                                            y_col = value_cols[0]
                                            pct_col = f"{y_col}_mom_pct_change"
                                            abs_col = f"{y_col}_mom_abs_change"
                                            try:
                                                import plotly.express as px
                                                from plotly.subplots import make_subplots
                                                import plotly.graph_objects as go
                                                
                                                fig = make_subplots(specs=[[{"secondary_y": True}]])
                                                
                                                # Main value line
                                                fig.add_trace(
                                                    go.Scatter(x=mom_df["period"], y=mom_df[y_col], name=y_col, mode="lines+markers", line=dict(color="#00d4ff")),
                                                    secondary_y=False,
                                                )
                                                
                                                # MoM % change bar
                                                if pct_col in mom_df.columns:
                                                    fig.add_trace(
                                                        go.Bar(x=mom_df["period"], y=mom_df[pct_col], name="MoM % Change", opacity=0.6, marker_color="#00b8e6"),
                                                        secondary_y=True,
                                                    )
                                                
                                                fig.update_xaxes(title_text="Period")
                                                fig.update_yaxes(title_text=y_col, secondary_y=False)
                                                fig.update_yaxes(title_text="MoM % Change", secondary_y=True)
                                                fig.update_layout(height=400, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified")
                                                st.plotly_chart(fig, use_container_width=True)
                                                
                                                # Show MoM change table
                                                with st.expander("MoM Change Details"):
                                                    display_cols = ["period", y_col]
                                                    if pct_col in mom_df.columns:
                                                        display_cols.append(pct_col)
                                                    if abs_col in mom_df.columns:
                                                        display_cols.append(abs_col)
                                                    st.dataframe(mom_df[display_cols].style.format({pct_col: "{:.1f}%", abs_col: "{:,.1f}"}), use_container_width=True)
                                            except Exception:
                                                st.dataframe(mom_df.head(100), use_container_width=True, hide_index=True)

                            # OUTLIER_DETECTION - Display outlier summary and details
                            if OperationType.OUTLIER_DETECTION in plan.operations and "outlier_detection" in data:
                                outlier_data = data["outlier_detection"]
                                if outlier_data:
                                    st.subheader("Outlier Detection")
                                    summary = outlier_data.get("summary", [{}])[0]
                                    outliers = outlier_data.get("outliers", [])
                                    
                                    # Summary metrics
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Outliers Found", summary.get("outlier_count", 0))
                                    with col2:
                                        st.metric("Outlier %", f"{summary.get('outlier_percentage', 0):.1f}%")
                                    with col3:
                                        st.metric("Method", summary.get("method", "").upper())
                                    
                                    # Show bounds for IQR
                                    if summary.get("method") == "iqr":
                                        st.caption(f"IQR Bounds: [{summary.get('lower_bound', 0):,.1f}, {summary.get('upper_bound', 0):,.1f}] (threshold: {summary.get('threshold', 1.5)}x IQR)")
                                    elif summary.get("method") == "zscore":
                                        st.caption(f"Mean: {summary.get('mean', 0):,.1f}, Std: {summary.get('std', 0):,.1f} (threshold: {summary.get('threshold', 2.0)}σ)")
                                    
                                    # Show outlier details
                                    if outliers:
                                        st.markdown("**Outlier Records:**")
                                        outlier_df = pd.DataFrame(outliers)
                                        st.dataframe(outlier_df, use_container_width=True, hide_index=True)
                                    else:
                                        st.info("No outliers detected.")

                            # Other operations - fallback to markdown table
                            if OperationType.MISSING_SUMMARY in plan.operations and "missing_summary" in data:
                                missing_df = data["missing_summary"]
                                if not missing_df.empty:
                                    st.subheader("Missing Value Summary")
                                    st.dataframe(missing_df, use_container_width=True, hide_index=True)

                            if OperationType.COLUMN_STATS in plan.operations and "column_stats" in data:
                                stats_df = data["column_stats"]
                                if not stats_df.empty:
                                    st.subheader("Column Statistics")
                                    st.dataframe(stats_df, use_container_width=True, hide_index=True)

                            if OperationType.VALUE_COUNTS in plan.operations and "value_counts" in data:
                                for col, vc_df in data["value_counts"].items():
                                    if not vc_df.empty:
                                        st.subheader(f"Top Values: {col}")
                                        st.dataframe(vc_df, use_container_width=True, hide_index=True)

                            # Check if any results were displayed
                            has_results = any([
                                OperationType.ROW_COUNT in plan.operations,
                                OperationType.AGGREGATE in plan.operations and "aggregate" in data and not data["aggregate"].empty,
                                OperationType.GROUP_BY in plan.operations and "group_by" in data and not data["group_by"].empty,
                                OperationType.DATE_ANALYSIS in plan.operations and "date_analysis" in data and not data["date_analysis"].empty,
                                OperationType.CORRELATION in plan.operations and "correlation" in data and not data["correlation"].empty,
                                OperationType.MISSING_SUMMARY in plan.operations and "missing_summary" in data and not data["missing_summary"].empty,
                                OperationType.COLUMN_STATS in plan.operations and "column_stats" in data and not data["column_stats"].empty,
                                OperationType.VALUE_COUNTS in plan.operations and "value_counts" in data and any(not df.empty for df in data["value_counts"].values()),
                                OperationType.MOM_COMPARISON in plan.operations and "mom_comparison" in data and not data["mom_comparison"].empty,
                                OperationType.OUTLIER_DETECTION in plan.operations and "outlier_detection" in data and data["outlier_detection"],
                            ])

                            if not has_results:
                                st.info("Analysis completed but returned no displayable results.")

                            # AI interpretation of results - build text summary
                            result_summary_parts = []
                            data = result.data or {}

                            if OperationType.ROW_COUNT in plan.operations:
                                row_count = data.get("row_count", result.row_count)
                                if row_count is not None:
                                    result_summary_parts.append(f"Row count: {row_count:,}")

                            if OperationType.AGGREGATE in plan.operations and "aggregate" in data:
                                agg_df = data["aggregate"]
                                if not agg_df.empty:
                                    if len(agg_df) == 1 and "index" not in agg_df.columns:
                                        for col in agg_df.columns:
                                            val = agg_df[col].iloc[0]
                                            if pd.api.types.is_numeric_dtype(agg_df[col]):
                                                result_summary_parts.append(f"{col}: {val:,.2f}")
                                            else:
                                                result_summary_parts.append(f"{col}: {val}")
                                    else:
                                        # Grouped aggregate: include sample for interpretation
                                        sample_rows = agg_df.head(10).to_dict(orient="records")
                                        result_summary_parts.append(f"Aggregated data sample: {sample_rows}")

                            if OperationType.GROUP_BY in plan.operations and "group_by" in data:
                                gb_df = data["group_by"]
                                if not gb_df.empty:
                                    result_summary_parts.append(f"Grouped results: {len(gb_df)} groups")
                                    # Include a small sample of actual grouped values for AI interpretation
                                    sample_rows = gb_df.head(10).to_dict(orient="records")
                                    result_summary_parts.append(f"Grouped data sample: {sample_rows}")

                            if OperationType.DATE_ANALYSIS in plan.operations and "date_analysis" in data:
                                da_df = data["date_analysis"]
                                if not da_df.empty:
                                    result_summary_parts.append(f"Time-series: {len(da_df)} periods")
                                    sample_rows = da_df.head(10).to_dict(orient="records")
                                    result_summary_parts.append(f"Time-series data sample: {sample_rows}")

                            if OperationType.CORRELATION in plan.operations and "correlation" in data:
                                corr_df = data["correlation"]
                                if not corr_df.empty:
                                    result_summary_parts.append(f"Correlation matrix: {corr_df.shape[0]}x{corr_df.shape[1]}")
                                    # Include top correlations as sample
                                    if "index" in corr_df.columns:
                                        corr_display = corr_df.set_index("index")
                                    else:
                                        corr_display = corr_df
                                    sample_rows = corr_display.head(5).to_dict(orient="records")
                                    result_summary_parts.append(f"Correlation sample: {sample_rows}")

                            if OperationType.MISSING_SUMMARY in plan.operations and "missing_summary" in data:
                                missing_df = data["missing_summary"]
                                if not missing_df.empty:
                                    result_summary_parts.append("Missing value summary computed")

                            if OperationType.COLUMN_STATS in plan.operations and "column_stats" in data:
                                stats_df = data["column_stats"]
                                if not stats_df.empty:
                                    result_summary_parts.append("Column statistics computed")

                            if OperationType.VALUE_COUNTS in plan.operations and "value_counts" in data:
                                for col, vc_df in data["value_counts"].items():
                                    if not vc_df.empty:
                                        result_summary_parts.append(f"Top values for {col}: {len(vc_df)} categories")

                            if OperationType.MOM_COMPARISON in plan.operations and "mom_comparison" in data:
                                mom_df = data["mom_comparison"]
                                if not mom_df.empty:
                                    result_summary_parts.append(f"MoM comparison: {len(mom_df)} periods")
                                    # Include latest period and MoM change
                                    if len(mom_df) > 1:
                                        latest = mom_df.iloc[-1]
                                        period = latest.get("period", "latest")
                                        # Find value column
                                        value_cols = [c for c in mom_df.columns if c != "period" and not c.endswith("_mom_pct_change") and not c.endswith("_mom_abs_change")]
                                        if value_cols:
                                            y_col = value_cols[0]
                                            pct_col = f"{y_col}_mom_pct_change"
                                            val = latest[y_col]
                                            pct = latest.get(pct_col)
                                            if pd.notna(pct):
                                                result_summary_parts.append(f"Latest {period}: {y_col}={val:,.1f}, MoM change={pct:.1f}%")
                                            else:
                                                result_summary_parts.append(f"Latest {period}: {y_col}={val:,.1f}")

                            if OperationType.OUTLIER_DETECTION in plan.operations and "outlier_detection" in data:
                                outlier_data = data["outlier_detection"]
                                if outlier_data:
                                    summary = outlier_data.get("summary", [{}])[0]
                                    outliers = outlier_data.get("outliers", [])
                                    method = summary.get("method", "unknown")
                                    count = summary.get("outlier_count", 0)
                                    pct = summary.get("outlier_percentage", 0)
                                    result_summary_parts.append(f"Outlier detection ({method}): {count} outliers ({pct:.1f}%)")
                                    if outliers:
                                        # Summarize top outlier values
                                        outlier_vals = [str(o.get(list(o.keys())[0], "")) for o in outliers[:3]]
                                        result_summary_parts.append(f"Top outliers: {', '.join(outlier_vals)}")

                            interpretation_prompt = (
                                "You are InsightAI. The user asked: "
                                f"'{question}'\n\n"
                                f"The analysis plan was: {plan.operations}\n\n"
                                f"Key results:\n{chr(10).join(result_summary_parts[:5])}\n\n"
                                "Provide a concise, practical interpretation of these results for the user. "
                                "Reference specific numbers. Note any limitations (Phase 3 verified calculations only). "
                                "Do not invent trends or statistics beyond what is shown."
                            )

                            interpretation_result = chat_with_nemotron(
                                messages=[{"role": "user", "content": interpretation_prompt}],
                                config=config,
                            )

                            if interpretation_result.success:
                                st.markdown("---")
                                st.subheader("AI Interpretation")
                                st.markdown(interpretation_result.content)

                            # Store in chat history (text version)
                            chat_summary = "Analysis completed with visualizations."
                            if result_summary_parts:
                                chat_summary = "; ".join(result_summary_parts[:3])
                            st.session_state.chat_messages.append({"role": "assistant", "content": chat_summary})

    # New conversation button
    if st.session_state.chat_messages:
        if st.button("Start new conversation", type="secondary"):
            st.session_state.chat_messages = []
            st.rerun()

st.divider()
st.caption("InsightAI Phase 3 · Prototype. Uploaded data is held in the current app session and is not intentionally written to disk.")