from __future__ import annotations

import re
import pandas as pd


def create_cleaning_preview(
    df: pd.DataFrame,
    *,
    trim_strings: bool = True,
    normalize_spaces: bool = False,
    drop_duplicates: bool = False,
    drop_empty_columns: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Create a cleaned copy and explain changes; never mutate the input DataFrame."""
    cleaned = df.copy(deep=True)
    changes: list[str] = []

    text_columns = cleaned.select_dtypes(include=["object", "string"]).columns
    if trim_strings:
        cells_changed = 0
        for col in text_columns:
            before = cleaned[col].copy()
            cleaned[col] = cleaned[col].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
            cells_changed += int((before != cleaned[col]).fillna(False).sum())
        if cells_changed:
            changes.append(f"Trimmed leading/trailing spaces in approximately {cells_changed:,} text cell(s).")
        else:
            changes.append("Trim spaces: no changes detected.")

    if normalize_spaces:
        cells_changed = 0
        for col in text_columns:
            before = cleaned[col].copy()
            cleaned[col] = cleaned[col].map(
                lambda value: re.sub(r"\s+", " ", value).strip() if isinstance(value, str) else value
            )
            cells_changed += int((before != cleaned[col]).fillna(False).sum())
        if cells_changed:
            changes.append(f"Normalized repeated whitespace in approximately {cells_changed:,} text cell(s).")
        else:
            changes.append("Normalize whitespace: no changes detected.")

    if drop_duplicates:
        before_rows = len(cleaned)
        cleaned = cleaned.drop_duplicates().copy()
        removed = before_rows - len(cleaned)
        changes.append(f"Removed {removed:,} exact duplicate row(s) from the preview.")

    if drop_empty_columns:
        empty_cols = [col for col in cleaned.columns if cleaned[col].isna().all()]
        if empty_cols:
            cleaned = cleaned.drop(columns=empty_cols)
            changes.append(f"Removed {len(empty_cols):,} entirely empty column(s): {', '.join(map(str, empty_cols[:8]))}.")
        else:
            changes.append("Remove empty columns: no entirely empty columns found.")

    return cleaned, changes
