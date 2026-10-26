from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd


class ColumnRole(str, Enum):
    """Semantic role of a column in business analytics."""
    MEASURE = "measure"           # Numeric values for aggregation (revenue, cost, quantity)
    DIMENSION = "dimension"       # Categorical for grouping (category, region, customer segment)
    DATE = "date"                 # Temporal for time-series analysis
    IDENTIFIER = "identifier"     # Unique keys (customer_id, order_id, transaction_id)
    CURRENCY = "currency"         # Monetary values (subset of measure with $ formatting)
    BOOLEAN = "boolean"           # True/False flags (is_active, churned, delivered)
    TEXT = "text"                 # Free-text (descriptions, notes, comments)
    UNKNOWN = "unknown"           # Could not determine


class BusinessDomain(str, Enum):
    """Business domain hints for domain-specific KPIs."""
    GENERAL = "general"
    SALES_RETAIL = "sales_retail"
    CHURN = "churn"
    SUPPLY_CHAIN = "supply_chain"
    HR = "hr"
    LOAN_FINANCE = "loan_finance"
    AMBIGUOUS = "ambiguous"


@dataclass
class SemanticColumn:
    """Rich metadata for a single column."""
    original_name: str
    normalized_name: str
    role: ColumnRole
    dtype: str
    sample_values: list[Any] = field(default_factory=list)
    missing_pct: float = 0.0
    unique_count: int = 0
    total_count: int = 0
    # Inference details
    confidence: float = 0.0           # 0-1 confidence in role assignment
    domain_hints: list[BusinessDomain] = field(default_factory=list)
    alternative_roles: list[ColumnRole] = field(default_factory=list)
    # Validation
    valid_conversion: bool = True     # Can be converted to target type (e.g., string->date)
    conversion_errors: list[str] = field(default_factory=list)
    # Business meaning
    business_meaning: str = ""        # Human-readable description
    is_sensitive: bool = False        # Potential PII/sensitive data


@dataclass
class DatasetProfile:
    """Complete semantic profile of a dataset."""
    original_columns: list[str]
    semantic_columns: dict[str, SemanticColumn]  # keyed by original_name
    row_count: int
    column_count: int
    # Role-based groupings
    measures: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    currencies: list[str] = field(default_factory=list)
    booleans: list[str] = field(default_factory=list)
    # Domain detection
    primary_domain: BusinessDomain = BusinessDomain.GENERAL
    domain_confidence: float = 0.0
    domain_evidence: dict[str, Any] = field(default_factory=dict)
    # Quality
    duplicate_rows: int = 0
    missing_cells: int = 0
    empty_columns: list[str] = field(default_factory=list)
    # Ambiguities needing user input
    ambiguous_columns: list[str] = field(default_factory=list)
    
    def get_columns_by_role(self, role: ColumnRole) -> list[str]:
        """Get column names with a specific role."""
        return [name for name, col in self.semantic_columns.items() if col.role == role]
    
    def get_primary_measure(self) -> Optional[str]:
        """Get the most likely primary measure (highest confidence currency/measure)."""
        candidates = self.measures + self.currencies
        if not candidates:
            return None
        # Sort by confidence, prefer currency
        def sort_key(name):
            col = self.semantic_columns[name]
            is_currency = col.role == ColumnRole.CURRENCY
            return (is_currency, col.confidence)
        return max(candidates, key=sort_key)
    
    def get_primary_date(self) -> Optional[str]:
        """Get the most likely primary date column."""
        if not self.dates:
            return None
        return max(self.dates, key=lambda n: self.semantic_columns[n].confidence)
    
    def get_primary_identifier(self) -> Optional[str]:
        """Get the most likely primary identifier."""
        if not self.identifiers:
            return None
        return max(self.identifiers, key=lambda n: self.semantic_columns[n].confidence)


def normalize_column_name(name: str) -> str:
    """Normalize column name for matching: lowercase, strip, replace spaces/special chars."""
    # Note: We do NOT split camelCase/PascalCase to maintain compatibility with existing tests
    # and column resolver logic. The column resolver uses fuzzy matching on normalized names.
    return (
        name.lower()
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace(".", "_")
        .replace("__", "_")
    )


# Keyword patterns for role inference
ROLE_KEYWORDS = {
    ColumnRole.MEASURE: [
        "amount", "value", "total", "sum", "count", "quantity", "qty", "volume",
        "score", "rate", "ratio", "percent", "percentage", "avg", "average",
        "mean", "median", "min", "max", "sum", "count", "number", "num",
        "sales", "revenue", "profit", "margin", "income", "earning", "yield",
        "units", "volume", "weight", "size", "length", "duration", "frequency",
    ],
    ColumnRole.DIMENSION: [
        "category", "type", "group", "segment", "class", "status", "state",
        "region", "area", "zone", "territory", "market", "channel", "brand",
        "product", "item", "sku", "name", "description", "label", "code",
        "department", "division", "team", "location", "store", "warehouse",
        "country", "city", "state", "province", "address", "geography",
    ],
    ColumnRole.DATE: [
        "date", "time", "timestamp", "created", "updated", "modified",
        "order_date", "ship_date", "delivery_date", "due_date", "start", "end",
        "year", "month", "day", "quarter", "period", "week",
    ],
    ColumnRole.IDENTIFIER: [
        "id", "key", "uuid", "guid", "number", "num", "code", "ref", "reference",
        "customer_id", "order_id", "transaction_id", "user_id", "employee_id",
        "product_id", "account_id", "invoice_id", "loan_id",
        "invoice_no", "invoice_number", "order_no", "order_number", "cust_id",
        "cust_no", "cust_code", "prod_id", "prod_code", "sku", "sku_code",
    ],
    ColumnRole.CURRENCY: [
        "price", "cost", "revenue", "sales", "profit", "margin", "amount",
        "payment", "paid", "balance", "salary", "wage", "income", "expense",
        "budget", "forecast", "invoice", "total", "subtotal", "tax", "fee",
        "charge", "premium", "claim", "payout", "commission", "bonus",
    ],
    ColumnRole.BOOLEAN: [
        "is_", "has_", "can_", "should_", "will_", "did_", "was_", "were_",
        "active", "enabled", "disabled", "true", "false", "yes", "no",
        "churned", "retained", "converted", "completed", "cancelled",
        "delivered", "shipped", "paid", "overdue", "default", "approved",
    ],
    ColumnRole.TEXT: [
        "description", "desc", "note", "notes", "comment", "comments",
        "text", "details", "info", "remarks", "memo", "summary",
    ],
}


# Business domain detection keywords
DOMAIN_KEYWORDS = {
    BusinessDomain.SALES_RETAIL: [
        "order", "sale", "revenue", "customer", "product", "sku", "quantity",
        "price", "discount", "promotion", "cart", "transaction", "purchase",
        "invoice", "receipt", "payment", "refund", "return",
        "sales", "order_date", "order_id", "customer_id",
    ],
    BusinessDomain.CHURN: [
        "churn", "retention", "tenure", "contract", "subscription", "renewal",
        "cancel", "active", "inactive", "lifetime", "cohort", "cohort",
        "monthly_recurring", "arr", "mrr", "ltv", "cac",
    ],
    BusinessDomain.SUPPLY_CHAIN: [
        "shipment", "delivery", "shipping", "logistics", "warehouse", "inventory",
        "stock", "supplier", "vendor", "procurement", "purchase_order", "po",
        "lead_time", "transit", "carrier", "freight", "customs", "dock",
    ],
    BusinessDomain.HR: [
        "employee", "staff", "personnel", "hire", "termination", "salary", "wage",
        "department", "position", "title", "manager", "performance", "review",
        "benefits", "leave", "attendance", "overtime", "bonus", "commission",
    ],
    BusinessDomain.LOAN_FINANCE: [
        "loan", "credit", "borrower", "lender", "principal", "interest", "apr",
        "default", "delinquency", "repayment", "installment", "collateral",
        "underwriting", "origination", "servicing", "portfolio", "risk_score",
        "fico", "credit_score", "dti", "ltv",
    ],
}


def infer_column_role(
    name: str,
    dtype: str,
    sample_values: list[Any],
    missing_pct: float,
    unique_count: int,
    total_count: int,
) -> tuple[ColumnRole, float, list[ColumnRole], list[BusinessDomain]]:
    """
    Infer the semantic role of a column.
    Returns: (primary_role, confidence, alternative_roles, domain_hints)
    """
    # Check for camelCase identifier patterns BEFORE normalization
    # This preserves the original casing for pattern matching
    import re
    camel_name = re.sub(r'(?<=[a-z0-9])([A-Z])', r'_\1', name)
    camel_name = camel_name.lower()
    
    norm_name = normalize_column_name(name)
    role_scores = {role: 0.0 for role in ColumnRole}
    domain_hints = []
    
    # 1. Name-based scoring - use word-boundary matching to avoid false positives
    # Split normalized name into words (by underscore)
    name_words = set(norm_name.split("_"))
    
    for role, keywords in ROLE_KEYWORDS.items():
        for kw in keywords:
            # Exact word match (whole word or underscore-separated)
            if kw in name_words:
                role_scores[role] += 3.0  # Higher weight for exact word match
            # Also check prefix/suffix with underscore boundary
            elif norm_name.startswith(kw + "_") or norm_name.endswith("_" + kw):
                role_scores[role] += 2.0
            # Substring match (lower weight, for compound words)
            elif kw in norm_name:
                role_scores[role] += 1.0
    
    # Check for strong identifier patterns BEFORE currency boost
    # Check both normalized and camelCase versions
    is_strong_id = False
    import re
    id_patterns = [
        r".*_id$", r".*_no$", r".*_num$", r".*_code$", r".*_key$", r".*_ref$",
        r"^id$", r"^key$", r"^code$", r"^ref$", r"^uuid$", r"^guid$",
        r"customer_id", r"order_id", r"invoice_id", r"transaction_id",
        r"product_id", r"account_id", r"employee_id", r"user_id",
        r"invoice_no", r"order_no", r"cust_id", r"prod_id",
    ]
    for pattern in id_patterns:
        if re.search(pattern, norm_name):
            is_strong_id = True
            break
    # Also check camelCase version (e.g., "InvoiceNo" -> "invoice_no")
    if not is_strong_id:
        camel_name = re.sub(r'(?<=[a-z0-9])([A-Z])', r'_\1', name).lower()
        for pattern in id_patterns:
            if re.search(pattern, camel_name):
                is_strong_id = True
                break
    # Also check camelCase version (e.g., "InvoiceNo" -> "invoice_no")
    if not is_strong_id:
        camel_name = re.sub(r'(?<=[a-z0-9])([A-Z])', r'_\1', norm_name)
        for pattern in id_patterns:
            if re.search(pattern, camel_name):
                is_strong_id = True
                break
    
    # Extra boost for strong currency indicators in column name
    # Skip if column looks like an identifier
    currency_strong_keywords = ["price", "cost", "revenue", "sales", "profit", "margin", "amount", "payment", "paid", "balance", "salary", "wage", "income", "expense", "invoice", "total", "subtotal", "tax", "fee", "charge", "premium", "claim", "payout", "commission", "bonus"]
    if not is_strong_id:
        for kw in currency_strong_keywords:
            if kw in name_words:
                role_scores[ColumnRole.CURRENCY] += 4.0
            elif norm_name.startswith(kw + "_") or norm_name.endswith("_" + kw):
                role_scores[ColumnRole.CURRENCY] += 3.0
            elif kw in norm_name:
                role_scores[ColumnRole.CURRENCY] += 1.0
    
    # Penalize currency/measure for strong ID patterns
    if is_strong_id:
        role_scores[ColumnRole.CURRENCY] = max(0.0, role_scores[ColumnRole.CURRENCY] - 4.0)
        role_scores[ColumnRole.MEASURE] = max(0.0, role_scores[ColumnRole.MEASURE] - 2.0)
        # Boost identifier
        role_scores[ColumnRole.IDENTIFIER] += 3.0
    
    # 2. Dtype-based scoring
    is_numeric = dtype in ("int64", "int32", "float64", "float32", "int16", "int8", "float16")
    is_datetime = "datetime" in dtype
    is_string = dtype in ("object", "string", "str")
    is_bool = dtype == "bool"
    is_categorical = dtype == "category"
    
    if is_datetime:
        role_scores[ColumnRole.DATE] += 5.0
    if is_bool:
        role_scores[ColumnRole.BOOLEAN] += 5.0
    if is_numeric:
        role_scores[ColumnRole.MEASURE] += 3.0
        # Currency gets boosted by name keywords, not just dtype
    if is_string or is_categorical:
        role_scores[ColumnRole.DIMENSION] += 2.0
        role_scores[ColumnRole.IDENTIFIER] += 1.0
    
    # 3. Sample values analysis
    if sample_values:
        # Check for currency-like values (2 decimal places, common ranges)
        numeric_samples = [v for v in sample_values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        if numeric_samples:
            # Heuristic: currency often has 2 decimal places
            decimal_places = sum(
                len(str(v).split(".")[-1]) if "." in str(v) else 0 
                for v in numeric_samples[:10]
            ) / min(len(numeric_samples), 10)
            if 1.5 <= decimal_places <= 2.5:
                role_scores[ColumnRole.CURRENCY] += 2.0
            # Also check for integer-like currency (e.g., 100, 200, 150 could be dollars)
            # or values with .0 decimal (100.0, 200.0)
            elif decimal_places <= 1.0:
                max_val = max(numeric_samples) if numeric_samples else 0
                if max_val > 100:
                    role_scores[ColumnRole.CURRENCY] += 1.0
                if max_val > 1000:
                    role_scores[ColumnRole.CURRENCY] += 1.0
            
            # Check for large numbers typical of currency
            max_val = max(numeric_samples) if numeric_samples else 0
            if max_val > 1000:
                role_scores[ColumnRole.CURRENCY] += 1.5
            if max_val > 100:
                role_scores[ColumnRole.CURRENCY] += 0.5
        
        # Check for boolean-like values
        str_samples = [str(v).lower().strip() for v in sample_values if v is not None]
        bool_indicators = {"true", "false", "yes", "no", "y", "n", "1", "0", "t", "f"}
        if str_samples and all(s in bool_indicators for s in str_samples[:20]):
            role_scores[ColumnRole.BOOLEAN] += 4.0
        
        # For numeric columns, check if they look like measures (not identifiers)
        if is_numeric and numeric_samples:
            # If values are sequential or have small range, likely not a measure
            min_val = min(numeric_samples)
            max_val = max(numeric_samples)
            val_range = max_val - min_val
            if val_range > 10 and len(numeric_samples) > 3:
                role_scores[ColumnRole.MEASURE] += 1.0
            # If values are like 1,2,3... likely identifier
            if min_val >= 1 and max_val <= len(numeric_samples) + 5 and val_range <= len(numeric_samples):
                role_scores[ColumnRole.IDENTIFIER] += 1.0
    
    # 4. Cardinality analysis
    unique_ratio = unique_count / total_count if total_count > 0 else 0
    if unique_ratio > 0.95 and total_count > 10:
        role_scores[ColumnRole.IDENTIFIER] += 4.0
    elif unique_ratio < 0.05 and total_count > 20:
        role_scores[ColumnRole.DIMENSION] += 2.0
    elif unique_ratio < 0.5 and total_count > 10:
        # Moderate cardinality - likely dimension
        role_scores[ColumnRole.DIMENSION] += 1.0
    
    # 5. Domain hints from column names (per-column, for column-level hints)
    for domain, keywords in DOMAIN_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in norm_name)
        if matches >= 1:
            domain_hints.append(domain)
    
    # Determine primary role and alternatives
    sorted_roles = sorted(role_scores.items(), key=lambda x: x[1], reverse=True)
    primary_role = sorted_roles[0][0] if sorted_roles[0][1] > 0 else ColumnRole.UNKNOWN
    primary_score = sorted_roles[0][1]
    
    # Confidence: normalize score (max theoretical ~10-12 for strong matches)
    confidence = min(primary_score / 10.0, 1.0)
    
    # Alternatives with significant scores
    alternatives = [r for r, s in sorted_roles[1:] if s >= 0.4 * primary_score and s > 1.0]
    
    return primary_role, confidence, alternatives, domain_hints


def compute_domain_scores(df: pd.DataFrame) -> dict[BusinessDomain, int]:
    """Compute domain scores based on keyword matches across all columns.
    
    Each column contributes to a domain if it matches at least 1 keyword.
    Weight by number of matches, capped at 2 per column to avoid single column dominating.
    """
    domain_scores = {domain: 0 for domain in DOMAIN_KEYWORDS.keys()}
    for col_name in df.columns:
        norm_name = normalize_column_name(col_name)
        for domain, keywords in DOMAIN_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in norm_name)
            if matches >= 1:
                # Weight by number of matches, but cap at 2 to avoid single column dominating
                domain_scores[domain] += min(matches, 2)
    return domain_scores


def detect_sensitive_columns(df: pd.DataFrame) -> list[str]:
    """Detect columns that may contain sensitive/PII data."""
    import re
    sensitive = []
    pii_patterns = [
        (r"email", "email"),
        (r"phone", "phone"),
        (r"ssn|social_security", "ssn"),
        (r"credit_card|card_number", "credit_card"),
        (r"address", "address"),
        (r"name", "name"),
        (r"dob|date_of_birth", "date_of_birth"),
        (r"passport", "passport"),
        (r"license", "license"),
    ]
    for col in df.columns:
        col_lower = col.lower()
        for pattern, label in pii_patterns:
            if re.search(pattern, col_lower):
                sensitive.append(col)
                break
    return sensitive


def build_semantic_profile(df: pd.DataFrame) -> DatasetProfile:
    """Build a complete semantic profile from a DataFrame."""
    from modules.data_quality import profile_dataset
    
    # Get base profile
    base_profile = profile_dataset(df)
    
    # Detect sensitive columns
    sensitive_cols = set(detect_sensitive_columns(df))
    
    semantic_columns = {}
    all_domain_hints = []
    
    for col_name in df.columns:
        # Handle duplicate column names - get first occurrence
        series = df[col_name]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        dtype = str(series.dtype)
        missing_pct = float(series.isna().mean() * 100) if len(df) else 0.0
        unique_count = int(series.nunique(dropna=True))
        total_count = int(series.notna().sum())
        sample_values = series.dropna().head(10).tolist()
        
        # Infer role
        role, confidence, alternatives, domain_hints = infer_column_role(
            col_name, dtype, sample_values, missing_pct, unique_count, len(df)
        )
        all_domain_hints.extend(domain_hints)
        
        # Check valid conversion for dates
        valid_conversion = True
        conversion_errors = []
        if role == ColumnRole.DATE and not is_datetime(dtype):
            # Will be validated later
            pass
        
        semantic_columns[col_name] = SemanticColumn(
            original_name=col_name,
            normalized_name=normalize_column_name(col_name),
            role=role,
            dtype=dtype,
            sample_values=sample_values,
            missing_pct=missing_pct,
            unique_count=unique_count,
            total_count=total_count,
            confidence=confidence,
            domain_hints=domain_hints,
            alternative_roles=alternatives,
            valid_conversion=valid_conversion,
            conversion_errors=conversion_errors,
            is_sensitive=col_name in sensitive_cols,
        )
    
    # Group by role
    measures = [n for n, c in semantic_columns.items() if c.role == ColumnRole.MEASURE]
    dimensions = [n for n, c in semantic_columns.items() if c.role == ColumnRole.DIMENSION]
    dates = [n for n, c in semantic_columns.items() if c.role == ColumnRole.DATE]
    identifiers = [n for n, c in semantic_columns.items() if c.role == ColumnRole.IDENTIFIER]
    currencies = [n for n, c in semantic_columns.items() if c.role == ColumnRole.CURRENCY]
    booleans = [n for n, c in semantic_columns.items() if c.role == ColumnRole.BOOLEAN]
    
    # Detect primary domain using aggregated domain scores
    domain_scores = compute_domain_scores(df)
    
    # Filter domains with sufficient evidence (score >= 2)
    valid_domains = {d: s for d, s in domain_scores.items() if s >= 2}
    
    if valid_domains:
        primary_domain = max(valid_domains, key=valid_domains.get)
        # Confidence based on relative strength
        total_score = sum(valid_domains.values())
        max_score = valid_domains[primary_domain]
        domain_confidence = max_score / total_score if total_score > 0 else 0.0
        # Cap confidence at 0.9 to avoid overconfidence
        domain_confidence = min(domain_confidence, 0.9)
    else:
        primary_domain = BusinessDomain.GENERAL
        domain_confidence = 0.0
    
    # Find ambiguous columns (low confidence or multiple strong alternatives)
    ambiguous = [
        name for name, col in semantic_columns.items()
        if col.confidence < 0.4 or (len(col.alternative_roles) > 0 and col.confidence < 0.6)
    ]
    
    return DatasetProfile(
        original_columns=list(df.columns),
        semantic_columns=semantic_columns,
        row_count=int(df.shape[0]),
        column_count=int(df.shape[1]),
        measures=measures,
        dimensions=dimensions,
        dates=dates,
        identifiers=identifiers,
        currencies=currencies,
        booleans=booleans,
        primary_domain=primary_domain,
        domain_confidence=domain_confidence,
        domain_evidence={"hint_counts": valid_domains},
        duplicate_rows=base_profile.get("duplicate_rows", 0),
        missing_cells=base_profile.get("missing_cells", 0),
        empty_columns=base_profile.get("empty_columns", []),
        ambiguous_columns=ambiguous,
    )


def is_datetime(dtype: str) -> bool:
    return "datetime" in dtype.lower()


# Export for use in other modules
__all__ = [
    "ColumnRole",
    "BusinessDomain",
    "SemanticColumn",
    "DatasetProfile",
    "normalize_column_name",
    "infer_column_role",
    "detect_sensitive_columns",
    "build_semantic_profile",
]