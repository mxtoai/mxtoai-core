import os
from unittest.mock import Mock, patch

import pytest

from mxtoai.routed_litellm_model import RoutedLiteLLMModel


@patch.dict(os.environ, {"LITELLM_DEFAULT_MODEL_GROUP": "gpt-4"})
@patch("mxtoai.routed_litellm_model.LiteLLMRouterModel.__init__")
@patch("tomllib.load")
@patch("builtins.open")
@patch("pathlib.Path.exists", return_value=True)
class TestRoutedLiteLLMModel:
    """Test the RoutedLiteLLMModel class functionality."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            "model": [
                {
                    "model_name": "gpt-4",
                    "litellm_params": {
                        "model": "gpt-4",
                        "weight": 1,
                        "api_key": "test_key",
                        "base_url": None,
                        "api_version": None,
                    },
                },
                {
                    "model_name": "thinking",
                    "litellm_params": {
                        "model": "thinking",
                        "weight": 1,
                        "api_key": "test_key",
                        "base_url": None,
                        "api_version": None,
                    },
                },
            ],
            "router_config": {
                "routing_strategy": "simple-shuffle",
                "fallbacks": [],
                "default_litellm_params": {"drop_params": True},
            },
        }

    @pytest.fixture
    def mock_response(self):
        """Mock LiteLLM response for testing."""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message = Mock()
        response.choices[0].message.model_dump.return_value = {
            "role": "assistant",
            "content": "Test response",
            "tool_calls": None,
        }
        response.usage = Mock()
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 50
        return response

    def test_stop_parameter_removal_for_thinking_model(
        self, mock_exists, mock_open, mock_tomllib_load, mock_super_init, mock_config, mock_response
    ):
        """Test that stop parameter is correctly removed for thinking models."""
        # Setup mocks
        mock_tomllib_load.return_value = mock_config
        mock_super_init.return_value = None

        # Create model instance
        model = RoutedLiteLLMModel()

        # Set required attributes that would normally be set by parent class
        model.api_base = "https://test.openai.azure.com"
        model.api_key = "test_api_key"
        model.custom_role_conversions = {}

        # Mock the client and its completion method
        mock_client = Mock()
        mock_client.completion.return_value = mock_response
        model.client = mock_client

        # Mock the _prepare_completion_kwargs method to capture the kwargs
        model._prepare_completion_kwargs = Mock(
            return_value={
                "messages": [{"role": "user", "content": "test"}],
                "model": "thinking",
                "stop": ["stop1", "stop2"],  # This should be removed
                "temperature": 0.7,
            }
        )

        # Set model_id to "thinking" to trigger the stop parameter removal
        model.model_id = "thinking"

        # Call generate method
        messages = [{"role": "user", "content": "test"}]
        stop_sequences = ["stop1", "stop2"]

        result = model.generate(
            messages=messages,
            stop_sequences=stop_sequences,
        )

        # Verify that completion was called
        mock_client.completion.assert_called_once()

        # Get the actual kwargs passed to completion
        completion_kwargs = mock_client.completion.call_args[1]

        # Assert that the stop parameter was removed
        assert "stop" not in completion_kwargs, "Stop parameter should be removed for thinking models"

        # Assert other parameters are still present
        assert "messages" in completion_kwargs
        assert "model" in completion_kwargs
        assert "temperature" in completion_kwargs

        # Verify the response
        assert result.content == "Test response"

    def test_stop_parameter_preserved_for_non_thinking_model(
        self, mock_exists, mock_open, mock_tomllib_load, mock_super_init, mock_config, mock_response
    ):
        """Test that stop parameter is preserved for non-thinking models."""
        # Setup mocks
        mock_tomllib_load.return_value = mock_config
        mock_super_init.return_value = None

        # Create model instance
        model = RoutedLiteLLMModel()

        # Set required attributes that would normally be set by parent class
        model.api_base = "https://test.openai.azure.com"
        model.api_key = "test_api_key"
        model.custom_role_conversions = {}

        # Mock the client and its completion method
        mock_client = Mock()
        mock_client.completion.return_value = mock_response
        model.client = mock_client

        # Mock the _prepare_completion_kwargs method to capture the kwargs
        model._prepare_completion_kwargs = Mock(
            return_value={
                "messages": [{"role": "user", "content": "test"}],
                "model": "gpt-4",
                "stop": ["stop1", "stop2"],  # This should be preserved
                "temperature": 0.7,
            }
        )

        # Set model_id to a non-thinking model
        model.model_id = "gpt-4"

        # Call generate method
        messages = [{"role": "user", "content": "test"}]
        stop_sequences = ["stop1", "stop2"]

        result = model.generate(
            messages=messages,
            stop_sequences=stop_sequences,
        )

        # Verify that completion was called
        mock_client.completion.assert_called_once()

        # Get the actual kwargs passed to completion
        completion_kwargs = mock_client.completion.call_args[1]

        # Assert that the stop parameter was preserved
        assert "stop" in completion_kwargs, "Stop parameter should be preserved for non-thinking models"
        assert completion_kwargs["stop"] == ["stop1", "stop2"]

        # Assert other parameters are still present
        assert "messages" in completion_kwargs
        assert "model" in completion_kwargs
        assert "temperature" in completion_kwargs

        # Verify the response
        assert result.content == "Test response"

    def test_stop_parameter_removal_case_sensitivity(
        self, mock_exists, mock_open, mock_tomllib_load, mock_super_init, mock_config, mock_response
    ):
        """Test that stop parameter removal is case sensitive (only exact 'thinking' match)."""
        # Setup mocks
        mock_tomllib_load.return_value = mock_config
        mock_super_init.return_value = None

        # Create model instance
        model = RoutedLiteLLMModel()

        # Set required attributes that would normally be set by parent class
        model.api_base = "https://test.openai.azure.com"
        model.api_key = "test_api_key"
        model.custom_role_conversions = {}

        # Mock the client and its completion method
        mock_client = Mock()
        mock_client.completion.return_value = mock_response
        model.client = mock_client

        # Mock the _prepare_completion_kwargs method to capture the kwargs
        model._prepare_completion_kwargs = Mock(
            return_value={
                "messages": [{"role": "user", "content": "test"}],
                "model": "Thinking",  # Different case
                "stop": ["stop1", "stop2"],  # This should be preserved
                "temperature": 0.7,
            }
        )

        # Set model_id to "Thinking" (different case)
        model.model_id = "Thinking"

        # Call generate method
        messages = [{"role": "user", "content": "test"}]
        stop_sequences = ["stop1", "stop2"]

        model.generate(
            messages=messages,
            stop_sequences=stop_sequences,
        )

        # Verify that completion was called
        mock_client.completion.assert_called_once()

        # Get the actual kwargs passed to completion
        completion_kwargs = mock_client.completion.call_args[1]

        # Assert that the stop parameter was preserved (case sensitive check)
        assert "stop" in completion_kwargs, "Stop parameter should be preserved for non-exact 'thinking' matches"
        assert completion_kwargs["stop"] == ["stop1", "stop2"]

    def test_stop_parameter_none_handling_for_thinking_model(
        self, mock_exists, mock_open, mock_tomllib_load, mock_super_init, mock_config, mock_response
    ):
        """Test that None stop parameter doesn't cause issues for thinking models."""
        # Setup mocks
        mock_tomllib_load.return_value = mock_config
        mock_super_init.return_value = None

        # Create model instance
        model = RoutedLiteLLMModel()

        # Set required attributes that would normally be set by parent class
        model.api_base = "https://test.openai.azure.com"
        model.api_key = "test_api_key"
        model.custom_role_conversions = {}

        # Mock the client and its completion method
        mock_client = Mock()
        mock_client.completion.return_value = mock_response
        model.client = mock_client

        # Mock the _prepare_completion_kwargs method with no stop parameter
        model._prepare_completion_kwargs = Mock(
            return_value={
                "messages": [{"role": "user", "content": "test"}],
                "model": "thinking",
                "temperature": 0.7,
                # No stop parameter included
            }
        )

        # Set model_id to "thinking"
        model.model_id = "thinking"

        # Call generate method with None stop_sequences
        messages = [{"role": "user", "content": "test"}]

        result = model.generate(
            messages=messages,
            stop_sequences=None,
        )

        # Verify that completion was called without errors
        mock_client.completion.assert_called_once()

        # Get the actual kwargs passed to completion
        completion_kwargs = mock_client.completion.call_args[1]

        # Assert that no stop parameter is present
        assert "stop" not in completion_kwargs

        # Verify the response
        assert result.content == "Test response"
