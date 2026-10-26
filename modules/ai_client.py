from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from openai import APIError, APIConnectionError, RateLimitError, AuthenticationError, APITimeoutError


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_TOKENS = 1500


@dataclass
class AIClientConfig:
    api_key: str | None
    base_url: str = NVIDIA_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_tokens: int = DEFAULT_MAX_TOKENS


@dataclass
class ChatResult:
    success: bool
    content: str | None = None
    error: str | None = None


def get_api_key() -> str | None:
    """Read NVIDIA API key from Streamlit secrets or environment variable."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "NVIDIA_API_KEY" in st.secrets:
            return str(st.secrets["NVIDIA_API_KEY"])
    except Exception:
        pass
    return os.environ.get("NVIDIA_API_KEY")


def create_client(config: AIClientConfig) -> OpenAI:
    """Create an OpenAI client configured for NVIDIA's API."""
    if not config.api_key:
        raise ValueError("NVIDIA_API_KEY is not configured. Set it in .streamlit/secrets.toml or the NVIDIA_API_KEY environment variable.")
    return OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout_seconds,
    )


def build_dataset_context(df, profile, quality_warnings: list[dict], semantic_profile=None) -> str:
    """Build a compact, privacy-safe dataset summary for the AI.

    Only sends essential metadata: column names, types, row count, missing values,
    duplicate count, and key aggregates. Never sends row data or private values.
    """
    lines = [
        "Dataset summary (metadata only):",
        f"Rows: {profile['rows']:,} | Columns: {profile['columns']:,} | Duplicates: {profile['duplicate_rows']:,}",
        "",
        "Columns:",
    ]

    dtype_info = profile.get("dtype_info")
    missing_by_col = profile.get("missing_by_column")

    if dtype_info is not None:
        for _, row in dtype_info.iterrows():
            col_name = row["Column"]
            dtype = row["Data type"]
            non_null = row["Non-null values"]
            unique = row["Unique values"]
            missing_pct = 0.0
            if missing_by_col is not None and not missing_by_col.empty:
                match = missing_by_col[missing_by_col["Column"] == col_name]
                if not match.empty:
                    missing_pct = match.iloc[0]["Missing %"]
            lines.append(f"  - {col_name} ({dtype}): {non_null:,} non-null, {unique:,} unique, {missing_pct:.1f}% missing")

    # Add semantic profile context if available
    if semantic_profile:
        lines.append("")
        lines.append("Semantic profile (inferred column roles):")
        lines.append(f"  Primary domain: {semantic_profile.primary_domain.value} (confidence: {semantic_profile.domain_confidence:.0%})")
        
        if semantic_profile.measures:
            lines.append(f"  Measures: {', '.join(semantic_profile.measures[:5])}{'...' if len(semantic_profile.measures) > 5 else ''}")
        if semantic_profile.dimensions:
            lines.append(f"  Dimensions: {', '.join(semantic_profile.dimensions[:5])}{'...' if len(semantic_profile.dimensions) > 5 else ''}")
        if semantic_profile.dates:
            lines.append(f"  Dates: {', '.join(semantic_profile.dates)}")
        if semantic_profile.currencies:
            lines.append(f"  Currency: {', '.join(semantic_profile.currencies)}")
        if semantic_profile.identifiers:
            lines.append(f"  Identifiers: {', '.join(semantic_profile.identifiers[:3])}{'...' if len(semantic_profile.identifiers) > 3 else ''}")
        if semantic_profile.booleans:
            lines.append(f"  Boolean flags: {', '.join(semantic_profile.booleans[:5])}{'...' if len(semantic_profile.booleans) > 5 else ''}")
        
        # Primary columns for analysis
        primary_measure = semantic_profile.get_primary_measure()
        primary_date = semantic_profile.get_primary_date()
        primary_id = semantic_profile.get_primary_identifier()
        if primary_measure:
            lines.append(f"  Primary measure: {primary_measure}")
        if primary_date:
            lines.append(f"  Primary date: {primary_date}")
        if primary_id:
            lines.append(f"  Primary identifier: {primary_id}")
        
        if semantic_profile.ambiguous_columns:
            lines.append(f"  ⚠️ Ambiguous columns (need review): {', '.join(semantic_profile.ambiguous_columns[:5])}")

    # Only include high-severity quality warnings (errors + warnings, skip info)
    severe_warnings = [w for w in quality_warnings if w["level"] in ("error", "warning")]
    if severe_warnings:
        lines.append("")
        lines.append("Quality issues:")
        for w in severe_warnings:
            lines.append(f"  - [{w['level'].upper()}] {w['message']}")

    lines.append("")
    lines.append("No row data included. Phase 2 = AI interpretation; Phase 3 = verified calculations.")

    return "\n".join(lines)


def chat_with_nemotron(
    messages: list[dict[str, str]],
    config: AIClientConfig,
    dataset_context: str | None = None,
    system_prompt: str | None = None,
) -> ChatResult:
    """Send a chat request to Nemotron and return the result."""
    if not config.api_key:
        return ChatResult(
            success=False,
            error="NVIDIA API key not configured. Add NVIDIA_API_KEY to .streamlit/secrets.toml or set the NVIDIA_API_KEY environment variable.",
        )

    client = create_client(config)

    # If full messages provided, use them directly (for analysis planning)
    if messages and any(m.get("role") == "system" for m in messages):
        full_messages = messages
    else:
        # Default behavior for general chat
        default_system = (
            "You are InsightAI, a concise data analytics assistant. "
            "Answer questions about dataset structure and quality using ONLY the provided metadata. "
            "Do not invent calculations, trends, or statistics. "
            "If a question requires actual computation, say so and note Phase 3 will add verified results. "
            "Keep responses to ~300-500 words. Use bullet points or small markdown tables for clarity. "
            "Be direct and practical."
        )
        full_messages = [{"role": "system", "content": system_prompt or default_system}]
        if dataset_context:
            full_messages.append({"role": "system", "content": dataset_context})
        full_messages.extend(messages)

    try:
        response = client.chat.completions.create(
            model=config.model,
            messages=full_messages,
            max_tokens=config.max_tokens,
            temperature=0.2,
        )
        content = response.choices[0].message.content
        if content is None or not content.strip():
            return ChatResult(success=False, error="Received empty response from the model.")
        return ChatResult(success=True, content=content.strip())

    except AuthenticationError:
        return ChatResult(
            success=False,
            error="Authentication failed. Check that your NVIDIA_API_KEY is valid and has access to the Nemotron model.",
        )
    except RateLimitError:
        return ChatResult(
            success=False,
            error="Rate limit exceeded. Please wait a moment and try again.",
        )
    except APITimeoutError:
        return ChatResult(
            success=False,
            error="Request timed out (30s). The model took too long. Try a simpler question or try again.",
        )
    except APIConnectionError:
        return ChatResult(
            success=False,
            error="Could not connect to the NVIDIA API. Check your network connection and try again.",
        )
    except APIError as e:
        return ChatResult(
            success=False,
            error=f"API error: {e.message if hasattr(e, 'message') else str(e)}",
        )
    except Exception as e:
        return ChatResult(
            success=False,
            error=f"Unexpected error: {type(e).__name__}: {e}",
        )


def is_configured() -> bool:
    """Check if the AI client is configured with a valid API key."""
    return get_api_key() is not None