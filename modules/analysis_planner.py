from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any, Literal, Optional

import pandas as pd

from modules.ai_client import chat_with_nemotron, AIClientConfig, ChatResult
from modules.analytics_engine import (
    AnalysisPlan,
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
    AnalysisResult,
)

METADATA_PATTERNS = [
    # Column count (check before columns to avoid "how many columns" matching "columns")
    (r"(how\s*many|count|number)\s*(of\s*)?columns\b", "column_count"),
    (r"column\s*(count|number)\b", "column_count"),
    # Column names
    (r"(list|show|what|display|get|show\s+me)\s*(out\s*)?(the\s*)?col(?:umn)?s?\s*(names?|headers?)\b", "columns"),
    (r"what\s+are\s+the\s+columns\b", "columns"),
    (r"list\s+columns\b", "columns"),
    (r"col(?:umn)?s?\s*(names?|headers?)\b", "columns"),
    (r"\bcolumns\b\s*(are|list)\b", "columns"),
    (r"(show|display|get|list)\s+(?:me\s+)?(?:the\s+)?columns\b", "columns"),
    (r"what\s+columns\b", "columns"),
    # Dataset shape
    (r"(show|what|display|get)\s*(the\s*)?(dataset\s*)?shape\b", "shape"),
    (r"dataset\s*shape\b", "shape"),
    (r"\bshape\b\s*(of\s*)?(the\s*)?dataset\b", "shape"),
    (r"what\s+is\s+the\s+shape\b", "shape"),
    # Row count
    (r"(how\s*many|count|number)\s*(of\s*)?(rows|records)\b", "row_count"),
    (r"(row|record)\s*(count|number)\b", "row_count"),
    (r"how\s*many\s+records\b", "row_count"),
    # Data types / dtypes
    (r"(show|display|get|list)\s+(?:me\s+)?(?:the\s+)?(?:data\s+)?types?\b", "dtypes"),
    (r"what\s+(?:are\s+)?(?:the\s+)?(?:data\s+)?types?\b", "dtypes"),
    (r"(?:column|col)\s+(?:data\s+)?types?\b", "dtypes"),
    (r"dtypes?\b", "dtypes"),
    # Missing values
    (r"(show|display|get|list)\s+(?:me\s+)?(?:the\s+)?missing\s+(?:values?|data)\b", "missing"),
    (r"what\s+(?:are\s+)?(?:the\s+)?missing\s+(?:values?|data)\b", "missing"),
    (r"missing\s+(?:values?|data|count)\b", "missing"),
    (r"(?:how\s+many\s+)?missing\b", "missing"),
    # Duplicates
    (r"(show|display|get|list)\s+(?:me\s+)?(?:the\s+)?duplicates?\b", "duplicates"),
    (r"what\s+(?:are\s+)?(?:the\s+)?duplicates?\b", "duplicates"),
    (r"duplicate\s+(?:rows?|count)\b", "duplicates"),
    (r"duplicates?\b", "duplicates"),
]


def is_metadata_question(question: str) -> Optional[str]:
    """Check if question is a simple metadata query.
    
    Returns the metadata type if matched: 'columns', 'column_count', 'shape', 'row_count'
    Returns None if not a metadata question.
    """
    question_lower = question.lower().strip()
    
    for pattern, meta_type in METADATA_PATTERNS:
        if re.search(pattern, question_lower):
            return meta_type
    
    return None


def handle_metadata_question(
    question: str,
    df,
    sem_profile=None,
) -> Optional[str]:
    """Handle simple metadata questions directly without AI.
    
    Returns formatted response string if question is handled, None otherwise.
    """
    meta_type = is_metadata_question(question)
    if meta_type is None:
        return None
    
    if df is None or (hasattr(df, 'empty') and df.empty):
        return "No dataset is currently loaded. Please upload a dataset first."
    
    if meta_type == "columns":
        cols = df.columns.astype(str).tolist()
        if not cols:
            return "Dataset has no columns."
        lines = ["**Column names:**"]
        for i, col in enumerate(cols, 1):
            dtype = str(df[col].dtype) if col in df.columns else "unknown"
            lines.append(f"  {i}. `{col}` ({dtype})")
        return "\n".join(lines)
    
    elif meta_type == "column_count":
        count = len(df.columns)
        return f"**Column count:** {count:,}"
    
    elif meta_type == "shape":
        rows, cols = df.shape
        return f"**Dataset shape:** {rows:,} rows × {cols:,} columns"
    
    elif meta_type == "row_count":
        count = len(df)
        return f"**Row count:** {count:,}"
    
    elif meta_type == "dtypes":
        lines = ["**Column data types:**"]
        for i, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype)
            non_null = int(df[col].notna().sum())
            unique = int(df[col].nunique(dropna=True))
            lines.append(f"  {i}. `{col}` — {dtype} ({non_null:,} non-null, {unique:,} unique)")
        return "\n".join(lines)
    
    elif meta_type == "missing":
        missing_df = df.isna().sum()
        total_missing = int(missing_df.sum())
        if total_missing == 0:
            return "**Missing values:** None — all columns are complete."
        lines = [f"**Missing values:** {total_missing:,} total"]
        for col, count in missing_df[missing_df > 0].items():
            pct = count / len(df) * 100
            lines.append(f"  - `{col}`: {int(count):,} ({pct:.1f}%)")
        return "\n".join(lines)
    
    elif meta_type == "duplicates":
        dup_count = int(df.duplicated().sum())
        if dup_count == 0:
            return "**Duplicate rows:** None found."
        return f"**Duplicate rows:** {dup_count:,}"
    
    return None


PII_PATTERNS = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b\d{16}\b"),
]


def sanitize_for_ai_context(text: str) -> str:
    """Remove potential PII from text before sending to AI."""
    result = text
    for pattern in PII_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result


def _fix_common_json_issues(json_str: str) -> str:
    """Fix common JSON issues from AI responses."""
    # Replace Python None/True/False with JSON null/true/false
    json_str = re.sub(r'\bNone\b', 'null', json_str)
    json_str = re.sub(r'\bTrue\b', 'true', json_str)
    json_str = re.sub(r'\bFalse\b', 'false', json_str)
    
    # Remove trailing commas before } or ]
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Fix missing commas between array elements (e.g., "a" "b" -> "a", "b")
    # Pattern: quoted string followed by whitespace and another quoted string or {
    json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)
    json_str = re.sub(r'"\s+"', '", "', json_str)
    json_str = re.sub(r'"\s+{', '", {', json_str)
    json_str = re.sub(r'}\s+"', '}, "', json_str)
    json_str = re.sub(r'}\s*{', '}, {', json_str)
    json_str = re.sub(r']\s*"', '], "', json_str)
    json_str = re.sub(r'"\s+\[', '", [', json_str)
    json_str = re.sub(r']\s+{', '], {', json_str)
    json_str = re.sub(r'}\s+\[', '}, [', json_str)
    
    # Fix missing commas between object properties (key-value pairs)
    # Pattern: value followed by newline and quote (property name)
    # Handle: value \n "key"  -> value, \n "key"
    json_str = re.sub(r'(["\d\]\}])\s*\n\s*"', r'\1,\n"', json_str)
    # Handle: value "key" -> value, "key"
    json_str = re.sub(r'(["\d\]\}])\s+"', r'\1, "', json_str)
    # Handle: } "key" -> }, "key" (closing brace followed by property)
    json_str = re.sub(r'}\s*\n\s*"', '},\n"', json_str)
    json_str = re.sub(r'}\s+"', '}, "', json_str)
    # Handle: ] "key" -> ], "key" (closing bracket followed by property)
    json_str = re.sub(r']\s*\n\s*"', '],\n"', json_str)
    json_str = re.sub(r']\s+"', '], "', json_str)
    # Handle: null "key" -> null, "key"
    json_str = re.sub(r'null\s*\n\s*"', 'null,\n"', json_str)
    json_str = re.sub(r'null\s+"', 'null, "', json_str)
    # Handle: true/false "key" -> true/false, "key"
    json_str = re.sub(r'\btrue\s*\n\s*"', 'true,\n"', json_str)
    json_str = re.sub(r'\btrue\s+"', 'true, "', json_str)
    json_str = re.sub(r'\bfalse\s*\n\s*"', 'false,\n"', json_str)
    json_str = re.sub(r'\bfalse\s+"', 'false, "', json_str)
    
    # Remove Python-style comments
    json_str = re.sub(r'//.*', '', json_str)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    
    # Replace single quotes with double quotes for string delimiters
    # This is a heuristic: replace '...' with "..." when it looks like a string value
    # We need to be careful not to break already-valid JSON
    # Pattern: ' followed by non-quote chars, followed by ' (not preceded by \)
    json_str = re.sub(r"'([^'\\]*(?:\\.[^'\\]*)*)'", r'"\1"', json_str)
    
    return json_str


def _balance_braces(json_str: str) -> str:
    """Attempt to fix unclosed braces/brackets in JSON string.
    
    Counts opening and closing braces/brackets and adds missing closing ones.
    """
    # Count braces and brackets
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')
    
    # Add missing closing braces
    if open_braces > close_braces:
        json_str += '}' * (open_braces - close_braces)
    
    # Add missing closing brackets
    if open_brackets > close_brackets:
        json_str += ']' * (open_brackets - close_brackets)
    
    return json_str


def _normalize_aggregation_synonyms(json_str: str) -> str:
    """Normalize common AI synonyms for aggregation functions to valid enum values.
    
    Handles cases where AI uses natural language terms instead of exact enum values.
    Only normalizes within aggregation function arrays to avoid false positives.
    """
    # Mapping of common synonyms to valid AggregationFunction enum values
    synonyms = {
        r'\btotal\b': 'sum',
        r'\btotals\b': 'sum',
        r'\baverage\b': 'mean',
        r'\bavg\b': 'mean',
        r'\bcount\b': 'count',
        r'\bminimum\b': 'min',
        r'\bmin\b': 'min',
        r'\bmaximum\b': 'max',
        r'\bmax\b': 'max',
        r'\bstd\b': 'std',
        r'\bstddev\b': 'std',
        r'\bstandard_deviation\b': 'std',
        r'\bmedian\b': 'median',
    }
    
    # Apply replacements within function arrays (inside [...])
    def replace_in_arrays(match):
        array_content = match.group(1)
        for pattern, replacement in synonyms.items():
            # Only replace whole words to avoid partial matches
            array_content = re.sub(pattern, replacement, array_content, flags=re.IGNORECASE)
        return f'[{array_content}]'
    
    # Find arrays of strings and normalize within them
    json_str = re.sub(r'\[\s*([^\]]*?)\s*\]', replace_in_arrays, json_str)
    
    return json_str


def build_analysis_system_prompt(columns: list[str], dtypes: dict[str, str]) -> str:
    """Build the system prompt for analysis planning."""
    col_info = []
    for col in columns:
        col_info.append(f"  - {col}: {dtypes.get(col, 'unknown')}")

    return f"""You are InsightAI's Analysis Planner. Convert the user's question into a structured JSON analysis plan.

Dataset columns and types:
{chr(10).join(col_info)}

Available operations:
- row_count: Count rows (after filters)
- missing_summary: Missing value counts and percentages per column
- column_stats: Statistics for specific columns (count, mean, std, min, max, median for numeric; min/max for dates)
- value_counts: Top value frequencies for categorical columns (max 50)
- filter: Filter rows by conditions
- group_by: Group by columns with aggregations (supports multiple measure columns)
- aggregate: Aggregate columns (optionally with group_by)
- sort: Sort by columns
- date_analysis: Time-series analysis by period (D/W/M/Q/Y)
- mom_comparison: Month-over-month or quarter-over-quarter comparison with % change
- correlation: Correlation matrix for numeric columns
- outlier_detection: Detect outliers using IQR or z-score method

Allowed filter operators: ==, !=, >, >=, <, <=, contains, startswith, endswith, in, not_in, is_null, is_not_null
Allowed aggregations: sum, mean, min, max, count, std, median

Output ONLY valid JSON with this exact structure:
{{
  "operations": ["operation1", "operation2", ...],
  "filters": [{{"column": "col", "operator": "==", "value": "val"}}, ...] or null,
  "group_by": {{"columns": ["col1"], "aggregations": {{"col2": ["sum", "mean"], "col3": ["sum"]}}}} or null,
  "aggregate": {{"columns": ["col1", "col2"], "functions": ["sum", "mean"], "group_by": ["col3"]}} or null,
  "sort": {{"columns": ["col1"], "ascending": [true]}} or null,
  "date_analysis": {{"date_column": "date_col", "value_column": "val_col", "frequency": "M", "aggregation": "sum"}} or null,
  "mom_comparison": {{"date_column": "date_col", "value_column": "val_col", "frequency": "M", "aggregation": "sum", "periods": 12}} or null,
  "column_stats": ["col1", "col2"] or null,
  "value_counts": ["col1"] or null,
  "correlation_columns": ["col1", "col2"] or null,
  "outlier_detection": {{"column": "val_col", "method": "iqr", "threshold": 1.5}} or null,
  "limit": 100 or null
}}

Rules:
1. Always include at least one operation
2. Only reference columns that exist in the dataset
3. Match operations to column types (e.g., date_analysis only on datetime columns)
4. For "how many" questions, use row_count
5. For "missing" questions, use missing_summary
6. For "average/sum/count by X", use group_by with aggregate (can include multiple measure columns)
7. For "trend over time", use date_analysis
8. For "month over month" or "MoM" or "compare months", use mom_comparison
9. For "top N", use sort + limit
10. For "outliers" or "anomalies", use outlier_detection
11. For comparing revenue and profit, use group_by with both columns in aggregations
12. Never invent columns or operations
13. If question is ambiguous, pick the most likely interpretation and note in analysis
14. Set limit to 100 for exploratory queries unless user specifies otherwise
15. Return ONLY the JSON, no extra text"""


def is_top_n_products_question(question: str) -> Optional[tuple[str, int]]:
    """Check if question is asking for top N products by revenue/sales.
    
    Returns tuple of (product_column_type, limit) if matched, None otherwise.
    product_column_type can be 'product', 'item', 'sku', or generic 'dimension'.
    """
    question_lower = question.lower().strip()
    
    # Patterns for "top N products/items by revenue/sales"
    patterns = [
        r"top\s+(\d+)\s+(products?|items?|skus?)\s+by\s+(revenue|sales|amount|total)",
        r"show\s+(?:me\s+)?(?:the\s+)?top\s+(\d+)\s+(products?|items?|skus?)\s+by\s+(revenue|sales|amount|total)",
        r"what\s+(?:are\s+)?(?:the\s+)?top\s+(\d+)\s+(products?|items?|skus?)\s+by\s+(revenue|sales|amount|total)",
        r"best\s+(\d+)\s+(products?|items?|skus?)\s+by\s+(revenue|sales|amount|total)",
        r"highest\s+(\d+)\s+(products?|items?|skus?)\s+by\s+(revenue|sales|amount|total)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, question_lower)
        if match:
            limit = int(match.group(1))
            # Cap at reasonable limit
            limit = min(max(limit, 1), 100)
            return ("product", limit)
    
    # More generic pattern: "top N by revenue/sales" without explicit product keyword
    generic_patterns = [
        r"top\s+(\d+)\s+by\s+(revenue|sales|amount|total)",
        r"show\s+(?:me\s+)?(?:the\s+)?top\s+(\d+)\s+by\s+(revenue|sales|amount|total)",
    ]
    
    for pattern in generic_patterns:
        match = re.search(pattern, question_lower)
        if match:
            limit = int(match.group(1))
            limit = min(max(limit, 1), 100)
            return ("dimension", limit)
    
    return None


def build_top_n_products_plan(
    question: str,
    df: pd.DataFrame,
    sem_profile=None,
    limit: int = 10,
) -> tuple[AnalysisPlan | None, pd.DataFrame | None, str | None]:
    """Build a deterministic analysis plan for top N products by revenue.
    
    This avoids AI and uses the dataset's actual column names.
    
    Returns:
        Tuple of (AnalysisPlan or None, modified DataFrame or None, error_message or None)
        The modified DataFrame includes any computed columns (e.g., revenue from qty * price).
        Returns original df if no computed columns were added.
    """
    from modules.semantic_model import ColumnRole, BusinessDomain
    from modules.column_resolver import create_resolver
    
    if df is None or (hasattr(df, 'empty') and df.empty):
        return None, "No dataset is currently loaded. Please upload a dataset first."
    
    # Build or use semantic profile
    if sem_profile is None:
        from modules.semantic_model import build_semantic_profile
        sem_profile = build_semantic_profile(df)
    
    # Find product dimension column (prefer product/item/sku dimensions)
    product_col = None
    
    # Priority 1: Explicit product/item/sku dimensions
    for col in sem_profile.dimensions:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ["product", "item", "sku", "description", "name"]):
            product_col = col
            break
    
    # Priority 2: Any dimension if no product-specific one found
    if product_col is None and sem_profile.dimensions:
        product_col = sem_profile.dimensions[0]
    
    if product_col is None:
        return None, None, f"No suitable product dimension column found. Available dimensions: {sem_profile.dimensions or 'none'}"
    
    # Find revenue measure column
    revenue_col = None
    
    # Priority 1: Explicit revenue/sales/amount/total columns (actual revenue, not unit price)
    explicit_revenue_cols = []
    for col in sem_profile.currencies + sem_profile.measures:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ["revenue", "sales", "amount", "total"]) and \
           not any(kw in col_lower for kw in ["unit_price", "unit price", "unit_cost", "price_per"]):
            explicit_revenue_cols.append(col)
    
    if explicit_revenue_cols:
        revenue_col = explicit_revenue_cols[0]
    
    # Priority 2: Check if we can compute revenue from quantity * unit_price
    # This is preferred over using a generic price column
    if revenue_col is None:
        qty_col = None
        price_col = None
        for col in sem_profile.measures + sem_profile.currencies:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ["quantity", "qty", "units", "volume"]):
                qty_col = col
            elif any(kw in col_lower for kw in ["unit_price", "unit price", "price", "cost"]):
                price_col = col
        
        if qty_col and price_col and qty_col in df.columns and price_col in df.columns:
            # We'll compute revenue as a new column
            revenue_col = "_computed_revenue"
    
    # Priority 3: Currency columns with revenue/sales keywords (but not unit_price)
    if revenue_col is None:
        currency_cols = [c for c in sem_profile.currencies if c in df.columns]
        for col in currency_cols:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ["revenue", "sales", "amount", "total"]) and \
               not any(kw in col_lower for kw in ["unit_price", "unit price", "unit_cost", "price_per"]):
                revenue_col = col
                break
    
    # Priority 4: Any currency column as last resort
    if revenue_col is None:
        currency_cols = [c for c in sem_profile.currencies if c in df.columns]
        if currency_cols:
            revenue_col = currency_cols[0]
    
    # Priority 5: Measure columns with revenue/sales keywords
    if revenue_col is None:
        for col in sem_profile.measures:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ["revenue", "sales", "amount", "total", "price"]):
                revenue_col = col
                break
    
    if revenue_col is None:
        available_measures = sem_profile.measures + sem_profile.currencies
        return None, None, (
            f"No suitable revenue column found. "
            f"Available measures: {available_measures or 'none'}. "
            f"Need a revenue/sales/amount column or quantity + unit_price columns."
        )
    
    # Handle computed revenue BEFORE creating resolver
    if revenue_col == "_computed_revenue":
        # Create computed revenue column on the dataframe
        df = df.copy()
        df["_computed_revenue"] = pd.to_numeric(df[qty_col], errors="coerce") * pd.to_numeric(df[price_col], errors="coerce")
        revenue_col = "_computed_revenue"
    
    # Validate columns exist in dataframe (after any computed columns added)
    resolver = create_resolver(df, sem_profile)
    
    try:
        resolved_product = resolver.resolve_required(product_col)
    except ValueError:
        return None, None, f"Product column '{product_col}' not found in dataset."
    
    try:
        resolved_revenue = resolver.resolve_required(revenue_col)
    except ValueError:
        return None, None, f"Revenue column '{revenue_col}' not found in dataset."
    
    # Build the analysis plan: group by product, sum revenue, sort desc, limit
    plan = AnalysisPlan(
        operations=[OperationType.GROUP_BY, OperationType.SORT],
        group_by=GroupBySpec(
            columns=[resolved_product],
            aggregations={resolved_revenue: [AggregationFunction.SUM]}
        ),
        sort=SortSpec(
            columns=[f"{resolved_revenue}_sum"],
            ascending=[False]
        ),
        limit=limit,
    )
    
    return plan, df, None


def parse_analysis_plan(json_str: str) -> tuple[AnalysisPlan | None, str | None]:
    """Parse and validate JSON analysis plan.
    
    Returns:
        Tuple of (AnalysisPlan or None, error_message or None)
    """
    # Handle empty or whitespace-only input
    if not json_str or not json_str.strip():
        return None, "Received empty response from AI model"
    
    # Extract JSON from response (handle markdown code blocks and extra text)
    json_str = json_str.strip()
    
    # Try to find JSON object in the response
    # Look for ```json ... ``` or ``` ... ``` blocks first
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', json_str, re.DOTALL)
    if code_block_match:
        json_str = code_block_match.group(1)
    else:
        # Try to find the first { and last } to extract JSON object
        first_brace = json_str.find('{')
        last_brace = json_str.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_str = json_str[first_brace:last_brace+1]
        else:
            # No valid JSON object found - the response appears to be natural language
            return None, "AI response does not contain valid JSON. Expected a JSON object with analysis plan. Response starts with: " + json_str[:200]
    
    # Fix common JSON issues from AI responses
    # Replace single quotes with double quotes (for string delimiters only)
    # This is a simple heuristic: replace ' with " when it appears to be a string delimiter
    json_str = _fix_common_json_issues(json_str)
    
    # Attempt to fix unclosed braces by balancing them
    json_str = _balance_braces(json_str)
    
    # Normalize common AI synonyms for aggregation functions
    json_str = _normalize_aggregation_synonyms(json_str)
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None, f"JSON parsing failed: {e.msg} at line {e.lineno}, column {e.colno} (char {e.pos}). Extracted text: {json_str[:300]}"
    
    # Ensure parsed data is a JSON object (dict), not an array or primitive
    if not isinstance(data, dict):
        return None, "Parsed JSON is not an object"
    
    try:
        operations = [OperationType(op) for op in data.get("operations", [])]
        if not operations:
            return None, "Missing or empty 'operations' field"

        filters = None
        if data.get("filters"):
            filters = []
            for f in data["filters"]:
                filters.append(FilterCondition(
                    column=f["column"],
                    operator=FilterOperator(f["operator"]),
                    value=f.get("value"),
                ))

        group_by = None
        if data.get("group_by"):
            gb = data["group_by"]
            aggregations = {}
            for col, funcs in gb.get("aggregations", {}).items():
                aggregations[col] = [AggregationFunction(f) for f in funcs]
            group_by = GroupBySpec(columns=gb["columns"], aggregations=aggregations)

        aggregate = None
        if data.get("aggregate"):
            agg = data["aggregate"]
            aggregate = AggregateSpec(
                columns=agg["columns"],
                functions=[AggregationFunction(f) for f in agg["functions"]],
                group_by=agg.get("group_by"),
            )

        sort = None
        if data.get("sort"):
            s = data["sort"]
            sort = SortSpec(columns=s["columns"], ascending=s.get("ascending", True))

        date_analysis = None
        if data.get("date_analysis"):
            da = data["date_analysis"]
            date_analysis = DateAnalysisSpec(
                date_column=da["date_column"],
                value_column=da.get("value_column"),
                frequency=da.get("frequency", "M"),
                aggregation=AggregationFunction(da.get("aggregation", "sum")),
            )

        mom_comparison = None
        if data.get("mom_comparison"):
            mom = data["mom_comparison"]
            mom_comparison = MoMComparisonSpec(
                date_column=mom["date_column"],
                value_column=mom["value_column"],
                frequency=mom.get("frequency", "M"),
                aggregation=AggregationFunction(mom.get("aggregation", "sum")),
                periods=mom.get("periods", 12),
            )

        outlier_detection = None
        if data.get("outlier_detection"):
            od = data["outlier_detection"]
            outlier_detection = OutlierDetectionSpec(
                column=od["column"],
                method=od.get("method", "iqr"),
                threshold=od.get("threshold", 1.5),
            )

        return AnalysisPlan(
            operations=operations,
            filters=filters,
            group_by=group_by,
            aggregate=aggregate,
            sort=sort,
            date_analysis=date_analysis,
            mom_comparison=mom_comparison,
            outlier_detection=outlier_detection,
            column_stats=data.get("column_stats"),
            value_counts=data.get("value_counts"),
            correlation_columns=data.get("correlation_columns"),
            limit=data.get("limit"),
        ), None
    except (KeyError, ValueError, TypeError) as e:
        return None, f"Schema validation failed: {e}"


def plan_analysis(
    question: str,
    config: AIClientConfig,
    dataset_context: str,
    columns: list[str],
    dtypes: dict[str, str],
    max_retries: int = 2,
    df: pd.DataFrame | None = None,
    sem_profile=None,
) -> tuple[AnalysisPlan | None, ChatResult]:
    """Use AI to generate an analysis plan from a natural language question.
    
    Args:
        question: User question (natural language)
        config: AI client configuration
        dataset_context: Dataset metadata context
        columns: List of actual column names
        dtypes: Dictionary of column names to data types
        max_retries: Maximum number of retry attempts for failed plans (default 2)
        df: Optional DataFrame for deterministic path
        sem_profile: Optional semantic profile for deterministic path
    
    Returns:
        Tuple of (AnalysisPlan or None, ChatResult)
    """
    # Try deterministic path for top N products by revenue questions
    if df is not None:
        top_n_match = is_top_n_products_question(question)
        if top_n_match:
            _, limit = top_n_match
            plan, exec_df, plan_error = build_top_n_products_plan(question, df, sem_profile, limit=limit)
            if plan is not None:
                return plan, ChatResult(success=True, content="Deterministic plan generated")
            # If deterministic path fails, fall through to AI planner
    
    system_prompt = build_analysis_system_prompt(columns, dtypes)
    
    # Base messages without error feedback
    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": dataset_context},
    ]
    
    last_error = None
    
    for attempt in range(max_retries + 1):
        if attempt == 0:
            # First attempt: just the question
            user_content = f"Question: {question}"
        else:
            # Retry: include error feedback
            user_content = (
                f"Question: {question}\n\n"
                f"Previous attempt failed with error: {last_error}\n"
                f"Available columns: {columns}\n"
                f"IMPORTANT: You must output ONLY valid JSON with the exact structure shown in the system prompt. "
                f"No explanations, no markdown, no extra text. Just the JSON object.\n"
                f"Required fields: operations (array), filters (array or null), group_by (object or null), "
                f"aggregate (object or null), sort (object or null), date_analysis (object or null), "
                f"mom_comparison (object or null), column_stats (array or null), value_counts (array or null), "
                f"correlation_columns (array or null), outlier_detection (object or null), limit (number or null)."
            )
        
        full_messages = base_messages + [{"role": "user", "content": user_content}]
        
        try:
            client_result = chat_with_nemotron(
                messages=full_messages,
                config=config,
            )
        except Exception as e:
            last_error = f"AI planning failed: {e}"
            continue
        
        if not client_result.success:
            last_error = client_result.error
            continue
        
        plan, parse_error = parse_analysis_plan(client_result.content)
        if plan is None:
            last_error = parse_error or "Failed to parse AI analysis plan (invalid JSON structure)."
            continue
        
        # Validate plan against actual columns before returning
        validation_error = _validate_plan_against_columns(plan, columns, dtypes)
        if validation_error:
            last_error = validation_error
            continue
        
        return plan, client_result
    
    return None, ChatResult(success=False, error=f"Failed to generate valid analysis plan after {max_retries + 1} attempts. Last error: {last_error}")


def _validate_plan_against_columns(
    plan: AnalysisPlan,
    columns: list[str],
    dtypes: dict[str, str],
) -> str | None:
    """Validate that all columns referenced in the plan exist in the dataset.
    
    Returns error message if validation fails, None if valid.
    """
    from modules.column_resolver import create_resolver
    from modules.semantic_model import DatasetProfile
    
    # Create a temporary resolver for validation
    df_dummy = type('obj', (object,), {'columns': columns})()
    resolver = create_resolver(pd.DataFrame(columns=columns))
    
    all_referenced = []
    
    # Check filters
    if plan.filters:
        for f in plan.filters:
            all_referenced.append(f.column)
    
    # Check group_by
    if plan.group_by:
        all_referenced.extend(plan.group_by.columns)
        all_referenced.extend(plan.group_by.aggregations.keys())
    
    # Check aggregate
    if plan.aggregate:
        all_referenced.extend(plan.aggregate.columns)
        if plan.aggregate.group_by:
            all_referenced.extend(plan.aggregate.group_by)
    
    # Check sort
    if plan.sort:
        all_referenced.extend(plan.sort.columns)
    
    # Check date_analysis
    if plan.date_analysis:
        all_referenced.append(plan.date_analysis.date_column)
        if plan.date_analysis.value_column:
            all_referenced.append(plan.date_analysis.value_column)
    
    # Check column_stats
    if plan.column_stats:
        all_referenced.extend(plan.column_stats)
    
    # Check value_counts
    if plan.value_counts:
        all_referenced.extend(plan.value_counts)
    
    # Check correlation_columns
    if plan.correlation_columns:
        all_referenced.extend(plan.correlation_columns)
    
    # Check mom_comparison
    if plan.mom_comparison:
        all_referenced.append(plan.mom_comparison.date_column)
        all_referenced.append(plan.mom_comparison.value_column)
    
    # Check outlier_detection
    if plan.outlier_detection:
        all_referenced.append(plan.outlier_detection.column)
    
    # Validate each referenced column
    missing = []
    for col in all_referenced:
        try:
            resolver.resolve_required(col)
        except ValueError:
            missing.append(col)
    
    if missing:
        return f"Plan references unknown columns: {missing}. Available columns: {columns}"
    
    return None


def execute_analysis(
    df,
    plan: AnalysisPlan,
    semantic_profile=None,
) -> AnalysisResult:
    """Execute a validated analysis plan."""
    from modules.analytics_engine import create_engine
    engine = create_engine(df, semantic_profile)
    return engine.execute(plan)