from __future__ import annotations

"""
Unified column resolution using semantic model.

This module bridges the semantic model (which provides rich column metadata)
with the analytics engine (which needs to resolve logical column names to
actual DataFrame columns).
"""

from typing import Optional

import pandas as pd

from modules.semantic_model import (
    ColumnRole,
    DatasetProfile,
    SemanticColumn,
    normalize_column_name,
)


# Common column aliases for cross-dataset compatibility
COMMON_ALIASES = {
    "order_date": ["order date", "orderdate", "order_date", "dateorders", "date_orders", "order date (dateorders)"],
    "order date (dateorders)": ["order date", "orderdate", "order_date", "dateorders", "date_orders", "order date (dateorders)"],
    "sales": ["revenue", "amount", "total_sales", "sale_amount"],
    "profit": ["margin", "gain", "profit_amount"],
    "quantity": ["qty", "units", "volume", "quantity_ordered"],
    "customer_id": ["customerid", "cust_id", "client_id", "customer_id"],
    "order_id": ["orderid", "ord_id", "transaction_id", "invoice_id"],
    "date": ["timestamp", "created_at", "updated_at", "datetime"],
    "category": ["type", "group", "segment", "class", "product_category"],
    "region": ["area", "zone", "territory", "market", "location"],
    "status": ["state", "order_status", "delivery_status"],
    "churned": ["churn", "cancelled", "inactive", "lost"],
    "late_delivery_risk": ["late_risk", "delivery_risk", "delay_risk", "on_time"],
    "shipping_days": ["ship_days", "delivery_days", "transit_days", "lead_time"],
}


class ColumnResolver:
    """
    Resolves logical column names to actual DataFrame columns using semantic profile.
    
    When a semantic profile is available, it uses the inferred roles and
    normalized names for intelligent matching. Falls back to exact/normalized
    matching when no profile is available.
    """
    
    def __init__(self, df: pd.DataFrame, semantic_profile: Optional[DatasetProfile] = None):
        self.df = df
        self.semantic_profile = semantic_profile
        self._normalized_to_original = {}
        self._build_normalized_index()
    
    def _build_normalized_index(self):
        """Build index from normalized names to original column names."""
        for col in self.df.columns:
            norm = normalize_column_name(col)
            self._normalized_to_original[norm] = col
    
    def _check_aliases(self, norm_logical: str) -> Optional[str]:
        """Check if the logical name matches any common aliases."""
        for canonical, aliases in COMMON_ALIASES.items():
            canonical_norm = normalize_column_name(canonical)
            alias_norms = [normalize_column_name(a) for a in aliases]
            if norm_logical == canonical_norm or norm_logical in alias_norms:
                for col in self.df.columns:
                    col_norm = normalize_column_name(col)
                    if col_norm == canonical_norm or col_norm in alias_norms:
                        return col
        return None
    
    def resolve(self, logical_name: str) -> Optional[str]:
        """
        Resolve a logical column name to an actual column in the DataFrame.
        
        Priority:
        1. Exact match in DataFrame columns
        2. Semantic profile: exact normalized name match
        3. Common aliases (e.g., order_date -> order date (DateOrders))
        4. Semantic profile: role-based match (primary measure, date, etc.)
        5. Case-insensitive match
        6. Partial normalized match
        
        Returns the actual column name or None if not found.
        Raises ValueError if multiple columns match (ambiguous).
        """
        # 1. Exact match
        if logical_name in self.df.columns:
            return logical_name
        
        norm_logical = normalize_column_name(logical_name)
        
        # 2. Normalized exact match (handles spaces, underscores, parentheses)
        if norm_logical in self._normalized_to_original:
            return self._normalized_to_original[norm_logical]
        
        # 3. Common aliases
        alias_match = self._check_aliases(norm_logical)
        if alias_match:
            return alias_match
        
        # 4. Semantic profile based resolution
        if self.semantic_profile:
            resolved = self._resolve_with_semantic_profile(norm_logical)
            if resolved:
                return resolved
        
        # 5. Case-insensitive match
        for col in self.df.columns:
            if col.lower() == logical_name.lower():
                return col
        
        # 6. Partial match on normalized names - check for ambiguity
        matches = []
        for norm, orig in self._normalized_to_original.items():
            if norm_logical in norm or norm in norm_logical:
                # Require substantial overlap
                longer = max(len(norm_logical), len(norm))
                shorter = min(len(norm_logical), len(norm))
                if shorter / longer >= 0.5:
                    matches.append(orig)
        
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            raise ValueError(
                f"Ambiguous column name '{logical_name}' matches multiple columns: {matches}. "
                f"Please use a more specific name."
            )
        
        return None
    
    def _resolve_with_semantic_profile(self, norm_logical: str) -> Optional[str]:
        """Use semantic profile for intelligent column resolution."""
        if not self.semantic_profile:
            return None
        
        # Try exact normalized name match against semantic columns
        for col_name, sem_col in self.semantic_profile.semantic_columns.items():
            if sem_col.normalized_name == norm_logical:
                return col_name
        
        # Try role-based resolution for common logical names
        role_hints = {
            "revenue": ColumnRole.CURRENCY,
            "sales": ColumnRole.CURRENCY,
            "amount": ColumnRole.CURRENCY,
            "profit": ColumnRole.CURRENCY,
            "cost": ColumnRole.CURRENCY,
            "quantity": ColumnRole.MEASURE,
            "qty": ColumnRole.MEASURE,
            "units": ColumnRole.MEASURE,
            "order_date": ColumnRole.DATE,
            "date": ColumnRole.DATE,
            "timestamp": ColumnRole.DATE,
            "customer_id": ColumnRole.IDENTIFIER,
            "order_id": ColumnRole.IDENTIFIER,
            "id": ColumnRole.IDENTIFIER,
            "category": ColumnRole.DIMENSION,
            "region": ColumnRole.DIMENSION,
            "market": ColumnRole.DIMENSION,
            "status": ColumnRole.DIMENSION,
            "churned": ColumnRole.BOOLEAN,
            "late": ColumnRole.BOOLEAN,
            "risk": ColumnRole.BOOLEAN,
        }
        
        # Check if logical name maps to a specific role
        for hint, role in role_hints.items():
            if hint in norm_logical:
                # Find all columns of that role
                role_columns = self.get_columns_by_role(role)
                if len(role_columns) == 1:
                    return role_columns[0]
                elif len(role_columns) > 1:
                    # Ambiguous - multiple columns of the same role
                    raise ValueError(
                        f"Ambiguous column name '{norm_logical}' matches multiple {role.value} columns: {role_columns}. "
                        f"Please use a more specific column name."
                    )
        
        # Try to find by semantic column business meaning
        for col_name, sem_col in self.semantic_profile.semantic_columns.items():
            if sem_col.business_meaning and norm_logical in normalize_column_name(sem_col.business_meaning):
                return col_name
        
        return None
    
    def resolve_required(self, logical_name: str) -> str:
        """Resolve column name, raise ValueError with helpful message if not found."""
        resolved = self.resolve(logical_name)
        if resolved is None:
            available = list(self.df.columns)
            context = []
            if self.semantic_profile:
                context.append(f"Measures: {self.semantic_profile.measures[:5]}")
                context.append(f"Dimensions: {self.semantic_profile.dimensions[:5]}")
                context.append(f"Dates: {self.semantic_profile.dates[:5]}")
                context.append(f"Identifiers: {self.semantic_profile.identifiers[:5]}")
                context.append(f"Currencies: {self.semantic_profile.currencies[:5]}")
            raise ValueError(
                f"Column '{logical_name}' not found in dataset. "
                f"Available columns: {available}. "
                f"Semantic context: {'; '.join(context) if context else 'No semantic profile available'}"
            )
        return resolved
    
    def get_columns_by_role(self, role: ColumnRole) -> list[str]:
        """Get all columns with a specific semantic role."""
        if self.semantic_profile:
            return self.semantic_profile.get_columns_by_role(role)
        # Fallback: infer from dtype
        if role == ColumnRole.MEASURE or role == ColumnRole.CURRENCY:
            return [c for c in self.df.columns if pd.api.types.is_numeric_dtype(self.df[c])]
        elif role == ColumnRole.DATE:
            return [c for c in self.df.columns if pd.api.types.is_datetime64_any_dtype(self.df[c])]
        elif role == ColumnRole.DIMENSION:
            return [c for c in self.df.columns if self.df[c].dtype == "object" and self.df[c].nunique() < 0.5 * len(self.df)]
        elif role == ColumnRole.IDENTIFIER:
            return [c for c in self.df.columns if self.df[c].nunique() == len(self.df)]
        elif role == ColumnRole.BOOLEAN:
            return [c for c in self.df.columns if self.df[c].dtype == "bool" or set(self.df[c].dropna().unique()).issubset({0, 1, True, False})]
        return []
    
    def get_primary_measure(self) -> Optional[str]:
        """Get the most likely primary measure column."""
        if self.semantic_profile:
            return self.semantic_profile.get_primary_measure()
        # Fallback: first numeric column
        numeric_cols = [c for c in self.df.columns if pd.api.types.is_numeric_dtype(self.df[c])]
        return numeric_cols[0] if numeric_cols else None
    
    def get_primary_date(self) -> Optional[str]:
        """Get the most likely primary date column."""
        if self.semantic_profile:
            return self.semantic_profile.get_primary_date()
        # Fallback: first datetime column
        date_cols = [c for c in self.df.columns if pd.api.types.is_datetime64_any_dtype(self.df[c])]
        return date_cols[0] if date_cols else None
    
    def get_primary_identifier(self) -> Optional[str]:
        """Get the most likely primary identifier column."""
        if self.semantic_profile:
            return self.semantic_profile.get_primary_identifier()
        # Fallback: first unique column
        for c in self.df.columns:
            if self.df[c].nunique() == len(self.df):
                return c
        return None
    
    def get_currency_columns(self) -> list[str]:
        """Get currency columns."""
        if self.semantic_profile:
            return self.semantic_profile.currencies
        return []
    
    def get_dimension_columns(self) -> list[str]:
        """Get dimension columns."""
        if self.semantic_profile:
            return self.semantic_profile.dimensions
        return []
    
    def validate_columns_exist(self, logical_names: list[str]) -> list[str]:
        """Resolve and validate multiple column names."""
        resolved = []
        for name in logical_names:
            resolved.append(self.resolve_required(name))
        return resolved
    
    def suggest_columns(self, logical_name: str, max_suggestions: int = 3) -> list[str]:
        """Suggest similar column names when resolution fails."""
        norm_logical = normalize_column_name(logical_name)
        suggestions = []
        
        for norm, orig in self._normalized_to_original.items():
            if norm_logical in norm or norm in norm_logical:
                suggestions.append(orig)
            elif len(norm_logical) > 2 and len(norm) > 2:
                # Check for similar prefixes
                if norm[:3] == norm_logical[:3]:
                    suggestions.append(orig)
        
        # Add role-based suggestions
        if self.semantic_profile:
            for role in ColumnRole:
                cols = self.get_columns_by_role(role)
                for col in cols:
                    if col not in suggestions:
                        suggestions.append(col)
        
        return suggestions[:max_suggestions]


def create_resolver(df: pd.DataFrame, semantic_profile: Optional[DatasetProfile] = None) -> ColumnResolver:
    """Factory function to create a ColumnResolver."""
    return ColumnResolver(df, semantic_profile)


# For backward compatibility with column_map.py interface
def get_col(df: pd.DataFrame, logical_name: str, semantic_profile: Optional[DatasetProfile] = None) -> Optional[str]:
    """Backward compatible get_col function."""
    resolver = create_resolver(df, semantic_profile)
    return resolver.resolve(logical_name)


def get_col_required(df: pd.DataFrame, logical_name: str, semantic_profile: Optional[DatasetProfile] = None) -> str:
    """Backward compatible get_col_required function."""
    resolver = create_resolver(df, semantic_profile)
    return resolver.resolve_required(logical_name)


def has_col(df: pd.DataFrame, logical_name: str, semantic_profile: Optional[DatasetProfile] = None) -> bool:
    """Backward compatible has_col function."""
    resolver = create_resolver(df, semantic_profile)
    return resolver.resolve(logical_name) is not None


def ensure_datetime(df: pd.DataFrame, logical_name: str, semantic_profile: Optional[DatasetProfile] = None) -> Optional[str]:
    """Backward compatible ensure_datetime function."""
    resolver = create_resolver(df, semantic_profile)
    col = resolver.resolve(logical_name)
    if col is None:
        return None
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        except Exception:
            return None
    return col