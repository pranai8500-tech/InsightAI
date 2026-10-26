from __future__ import annotations

import pytest
from unittest.mock import Mock, patch

from modules.ai_client import (
    chat_with_nemotron,
    build_dataset_context,
    AIClientConfig,
    is_configured,
    ChatResult,
)


class TestAIClientConfig:
    def test_default_config(self):
        config = AIClientConfig(api_key="test-key")
        assert config.base_url == "https://integrate.api.nvidia.com/v1"
        assert config.model == "nvidia/nemotron-3-ultra-550b-a55b"
        assert config.timeout_seconds == 30.0
        assert config.max_tokens == 1500

    def test_custom_config(self):
        config = AIClientConfig(
            api_key="custom-key",
            base_url="https://custom.api/v1",
            model="custom-model",
            timeout_seconds=60.0,
            max_tokens=2048,
        )
        assert config.api_key == "custom-key"
        assert config.base_url == "https://custom.api/v1"
        assert config.model == "custom-model"
        assert config.timeout_seconds == 60.0
        assert config.max_tokens == 2048


class TestIsConfigured:
    @patch("modules.ai_client.get_api_key", return_value=None)
    def test_not_configured(self, mock_get_key):
        assert is_configured() is False

    @patch("modules.ai_client.get_api_key", return_value="test-key")
    def test_configured(self, mock_get_key):
        assert is_configured() is True


class TestBuildDatasetContext:
    def test_basic_context(self):
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        profile = {
            "rows": 2,
            "columns": 2,
            "duplicate_rows": 0,
            "total_memory_mb": 0.001,
            "dtype_info": pd.DataFrame({
                "Column": ["a", "b"],
                "Data type": ["int64", "object"],
                "Non-null values": [2, 2],
                "Unique values": [2, 2],
            }),
        }
        warnings = []
        context = build_dataset_context(df, profile, warnings)
        assert "Dataset summary (metadata only):" in context
        assert "Rows: 2" in context
        assert "Columns: 2" in context
        assert "Duplicates: 0" in context
        assert "a (int64)" in context
        assert "b (object)" in context
        assert "non-null" in context
        assert "unique" in context
        assert "missing" in context
        assert "No row data included" in context
        assert "Phase 2 = AI interpretation" in context
        assert "Phase 3 = verified calculations" in context

    def test_context_with_warnings(self):
        import pandas as pd
        df = pd.DataFrame({"a": [1, 2]})
        profile = {
            "rows": 2,
            "columns": 1,
            "duplicate_rows": 0,
            "total_memory_mb": 0.001,
            "dtype_info": pd.DataFrame({
                "Column": ["a"],
                "Data type": ["int64"],
                "Non-null values": [2],
                "Unique values": [2],
            }),
        }
        warnings = [
            {"level": "warning", "message": "Duplicate rows detected"},
            {"level": "info", "message": "Missing values in column b"},
        ]
        context = build_dataset_context(df, profile, warnings)
        assert "[WARNING] Duplicate rows detected" in context
        # Info-level warnings are filtered out in compact context
        assert "[INFO]" not in context

    def test_context_with_date_candidates(self):
        import pandas as pd
        df = pd.DataFrame({"date_col": ["2024-01-01", "2024-02-01"]})
        profile = {
            "rows": 2,
            "columns": 1,
            "duplicate_rows": 0,
            "total_memory_mb": 0.001,
            "dtype_info": pd.DataFrame({
                "Column": ["date_col"],
                "Data type": ["object"],
                "Non-null values": [2],
                "Unique values": [2],
            }),
            "date_candidates": [
                {
                    "column": "date_col",
                    "suspected_format": "ISO (YYYY-MM-DD)",
                    "parse_success_rate": 100.0,
                }
            ],
        }
        warnings = []
        context = build_dataset_context(df, profile, warnings)
        # Date candidates are not included in compact context
        assert "Dataset summary (metadata only):" in context
        assert "date_col" in context
        assert "object" in context


class TestChatWithNemotron:
    @patch("modules.ai_client.get_api_key", return_value=None)
    def test_missing_api_key_returns_error(self, mock_get_key):
        config = AIClientConfig(api_key=None)
        result = chat_with_nemotron(
            messages=[{"role": "user", "content": "test"}],
            config=config,
        )
        assert result.success is False
        assert "not configured" in result.error.lower()

    @patch("modules.ai_client.get_api_key", return_value="test-key")
    @patch("modules.ai_client.OpenAI")
    def test_successful_response(self, mock_openai_class, mock_get_key):
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="  Test response  "))]
        mock_client.chat.completions.create.return_value = mock_response

        config = AIClientConfig(api_key="test-key")
        result = chat_with_nemotron(
            messages=[{"role": "user", "content": "Hello"}],
            config=config,
        )

        assert result.success is True
        assert result.content == "Test response"
        mock_client.chat.completions.create.assert_called_once()

    @patch("modules.ai_client.get_api_key", return_value="test-key")
    @patch("modules.ai_client.OpenAI")
    def test_empty_response_handled(self, mock_openai_class, mock_get_key):
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=None))]
        mock_client.chat.completions.create.return_value = mock_response

        config = AIClientConfig(api_key="test-key")
        result = chat_with_nemotron(
            messages=[{"role": "user", "content": "Hello"}],
            config=config,
        )

        assert result.success is False
        assert "empty response" in result.error.lower()

    @patch("modules.ai_client.get_api_key", return_value="test-key")
    @patch("modules.ai_client.OpenAI")
    def test_authentication_error(self, mock_openai_class, mock_get_key):
        from openai import AuthenticationError
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = AuthenticationError(
            message="Invalid API key",
            response=Mock(status_code=401),
            body={},
        )

        config = AIClientConfig(api_key="test-key")
        result = chat_with_nemotron(
            messages=[{"role": "user", "content": "Hello"}],
            config=config,
        )

        assert result.success is False
        assert "authentication failed" in result.error.lower()

    @patch("modules.ai_client.get_api_key", return_value="test-key")
    @patch("modules.ai_client.OpenAI")
    def test_rate_limit_error(self, mock_openai_class, mock_get_key):
        from openai import RateLimitError
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RateLimitError(
            message="Rate limit exceeded",
            response=Mock(status_code=429),
            body={},
        )

        config = AIClientConfig(api_key="test-key")
        result = chat_with_nemotron(
            messages=[{"role": "user", "content": "Hello"}],
            config=config,
        )

        assert result.success is False
        assert "rate limit" in result.error.lower()

    @patch("modules.ai_client.get_api_key", return_value="test-key")
    @patch("modules.ai_client.OpenAI")
    def test_timeout_error(self, mock_openai_class, mock_get_key):
        from openai import APITimeoutError
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = APITimeoutError(request=Mock())

        config = AIClientConfig(api_key="test-key")
        result = chat_with_nemotron(
            messages=[{"role": "user", "content": "Hello"}],
            config=config,
        )

        assert result.success is False
        assert "timed out" in result.error.lower()

    @patch("modules.ai_client.get_api_key", return_value="test-key")
    @patch("modules.ai_client.OpenAI")
    def test_connection_error(self, mock_openai_class, mock_get_key):
        from openai import APIConnectionError
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = APIConnectionError(request=Mock())

        config = AIClientConfig(api_key="test-key")
        result = chat_with_nemotron(
            messages=[{"role": "user", "content": "Hello"}],
            config=config,
        )

        assert result.success is False
        assert "connect" in result.error.lower()

    @patch("modules.ai_client.get_api_key", return_value="test-key")
    @patch("modules.ai_client.OpenAI")
    def test_generic_api_error(self, mock_openai_class, mock_get_key):
        from openai import APIError
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = APIError(
            message="Internal server error",
            request=Mock(),
            body={},
        )

        config = AIClientConfig(api_key="test-key")
        result = chat_with_nemotron(
            messages=[{"role": "user", "content": "Hello"}],
            config=config,
        )

        assert result.success is False
        assert "api error" in result.error.lower()

    @patch("modules.ai_client.get_api_key", return_value="test-key")
    @patch("modules.ai_client.OpenAI")
    def test_unexpected_error(self, mock_openai_class, mock_get_key):
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = RuntimeError("Unexpected!")

        config = AIClientConfig(api_key="test-key")
        result = chat_with_nemotron(
            messages=[{"role": "user", "content": "Hello"}],
            config=config,
        )

        assert result.success is False
        assert "unexpected error" in result.error.lower()
        assert "RuntimeError" in result.error

    @patch("modules.ai_client.get_api_key", return_value="test-key")
    @patch("modules.ai_client.OpenAI")
    def test_dataset_context_included(self, mock_openai_class, mock_get_key):
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_client.chat.completions.create.return_value = mock_response

        config = AIClientConfig(api_key="test-key")
        dataset_context = "Test dataset context"
        result = chat_with_nemotron(
            messages=[{"role": "user", "content": "Hello"}],
            config=config,
            dataset_context=dataset_context,
        )

        assert result.success is True
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "system"
        assert messages[1]["content"] == dataset_context
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "Hello"


class TestChatResult:
    def test_success_result(self):
        result = ChatResult(success=True, content="Hello world")
        assert result.success is True
        assert result.content == "Hello world"
        assert result.error is None

    def test_error_result(self):
        result = ChatResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.content is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])