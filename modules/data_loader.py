from __future__ import annotations

import io
from typing import Optional

import pandas as pd


SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xls")

# Common CSV delimiters to try
CSV_DELIMITERS = [",", ";", "\t", "|", ":"]


def validate_upload(uploaded_file, max_size_mb: int = 100) -> str | None:
    """Return a user-friendly validation error, or None when the upload looks valid."""
    if uploaded_file is None:
        return "Choose a file to upload."
    name = str(getattr(uploaded_file, "name", "")).lower()
    if not name.endswith(SUPPORTED_EXTENSIONS):
        return "Unsupported file type. Upload a CSV, XLSX, or XLS file."
    size = int(getattr(uploaded_file, "size", 0))
    if size > max_size_mb * 1024 * 1024:
        return f"File is too large. The current limit is {max_size_mb} MB."
    if size == 0:
        return "The uploaded file is empty."
    return None


def get_excel_sheets(uploaded_file) -> list[str]:
    """List worksheet names from an Excel upload."""
    uploaded_file.seek(0)
    with pd.ExcelFile(uploaded_file) as workbook:
        sheets = list(workbook.sheet_names)
    uploaded_file.seek(0)
    return sheets


def _detect_delimiter(sample: bytes) -> str:
    """Detect the most likely delimiter from a file sample."""
    # Decode sample for analysis
    try:
        text = sample.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = sample.decode("latin-1", errors="ignore")
    
    # Count delimiter occurrences in first few lines
    lines = text.split("\n")[:5]
    delimiter_counts = {}
    for delim in CSV_DELIMITERS:
        counts = [line.count(delim) for line in lines if line.strip()]
        if counts and all(c == counts[0] and c > 0 for c in counts):
            delimiter_counts[delim] = counts[0]
    
    if delimiter_counts:
        # Return delimiter with most consistent column count
        return max(delimiter_counts, key=delimiter_counts.get)
    return ","  # Default fallback


def _try_read_csv(uploaded_file, delimiter: str, encoding: str) -> pd.DataFrame | None:
    """Try to read CSV with given delimiter and encoding."""
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(
            uploaded_file,
            delimiter=delimiter,
            encoding=encoding,
            low_memory=False,
            on_bad_lines="skip",  # Skip malformed rows
        )
        # Validate: should have at least 1 column and reasonable rows
        if df.shape[1] >= 1 and df.shape[0] >= 0:
            return df
    except Exception:
        pass
    return None


def load_uploaded_dataset(uploaded_file, sheet_name: str | None = None) -> pd.DataFrame:
    """Load a CSV or Excel sheet with robust parsing and clear errors."""
    uploaded_file.seek(0)
    name = str(uploaded_file.name).lower()
    
    if name.endswith(".csv"):
        # Read sample for delimiter detection
        sample = uploaded_file.read(8192)
        uploaded_file.seek(0)
        
        detected_delimiter = _detect_delimiter(sample)
        
        # Try encodings with detected delimiter first, then fallback delimiters
        encodings = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
        delimiters_to_try = [detected_delimiter] + [d for d in CSV_DELIMITERS if d != detected_delimiter]
        
        last_error = None
        for encoding in encodings:
            for delim in delimiters_to_try:
                df = _try_read_csv(uploaded_file, delim, encoding)
                if df is not None:
                    # Success - process columns and return
                    return _finalize_dataframe(df)
                uploaded_file.seek(0)
        
        # If all failed, try default comma with error details
        raise ValueError(f"Could not parse CSV. Detected delimiter: '{detected_delimiter}'. Tried encodings: {encodings}")
    
    elif name.endswith((".xlsx", ".xls")):
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name or 0)
        return _finalize_dataframe(df)
    
    else:
        raise ValueError("Unsupported file type. Upload CSV, XLSX, or XLS.")


def _finalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up column names and validate DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The selected worksheet did not contain a readable table.")
    
    # Handle empty dataframe
    if df.shape[0] == 0 and df.shape[1] == 0:
        raise ValueError("The dataset is empty (no rows and no columns).")
    
    # Avoid ambiguous duplicate column names in downstream operations.
    cols = pd.Index([
        str(c).strip() if str(c).strip() else f"Unnamed_{i+1}" 
        for i, c in enumerate(df.columns)
    ])
    if cols.duplicated().any():
        counts = {}
        new_cols = []
        for col in cols:
            counts[col] = counts.get(col, 0) + 1
            new_cols.append(col if counts[col] == 1 else f"{col}_{counts[col]}")
        cols = pd.Index(new_cols)
    df.columns = cols
    
    return df


def get_load_info(df: pd.DataFrame, original_name: str) -> dict:
    """Return metadata about the loaded dataset for UI display."""
    return {
        "filename": original_name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
