import os
import tomllib
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from smolagents import ChatMessage, LiteLLMRouterModel, Tool

import mxtoai.schemas
from mxtoai import exceptions
from mxtoai._logging import get_logger
from mxtoai.schemas import ProcessingInstructions

load_dotenv()

logger = get_logger("routed_litellm_model")


class RoutedLiteLLMModel(LiteLLMRouterModel):
    """LiteLLM Model with routing capabilities, using LiteLLMRouterModel from smolagents."""

    def __init__(self, current_handle: ProcessingInstructions | None = None, **kwargs):
        """
        Initialize the routed LiteLLM model.

        Args:
            current_handle: Current email handle configuration being processed
            **kwargs: Additional arguments passed to parent class (e.g., flatten_messages_as_text)

        """
        self.current_handle = current_handle
        self.config_path = os.getenv("LITELLM_CONFIG_PATH", "model.config.toml")
        self.config = self._load_toml_config()

        # Configure model list from environment variables
        model_list = self._load_model_config()
        client_router_kwargs = self._load_router_config()

        # The model_id for LiteLLMRouterModel is the default model group the router will target.
        # Our _get_target_model() will override this per call via the 'model' param in generate().
        default_model_group = os.getenv("LITELLM_DEFAULT_MODEL_GROUP")

        if not default_model_group:
            msg = (
                "LITELLM_DEFAULT_MODEL_GROUP environment variable not found. Please set it to the default model group."
            )
            raise exceptions.EnvironmentVariableNotFoundError(msg)

        # Initialize token count attributes before calling super().__init__
        self._last_input_token_count = 0
        self._last_output_token_count = 0

        # Set environment variables for Azure OpenAI models before initializing the parent class
        # This is required because LiteLLM's Azure provider looks for specific environment variables
        self._set_azure_environment_variables(model_list)

        super().__init__(
            model_id=default_model_group,
            model_list=[model.model_dump() for model in model_list],
            client_kwargs=client_router_kwargs.model_dump(),
            **kwargs,  # Pass through other LiteLLMModel/Model kwargs
        )

    def _load_toml_config(self) -> dict[str, Any]:
        """
        Load configuration from a TOML file.

        Returns:
            Dict[str, Any]: Configuration loaded from the TOML file.

        """
        if not Path(self.config_path).exists():
            msg = f"Model config file not found at {self.config_path}. Please check the path."
            raise exceptions.ModelConfigFileNotFoundException(msg)

        try:
            with Path(self.config_path).open("rb") as f:
                return tomllib.load(f)
        except Exception as e:
            logger.error(f"Failed to load TOML config: {e}")
            return {}

    def _load_model_config(self) -> list[mxtoai.schemas.ModelConfig]:
        """
        Load model configuration from TOML file.

        Returns:
            list[mxtoai.schemas.ModelConfig]: List of model configurations

        """
        model_entries = self.config.get("model", [])
        if not model_entries:
            msg = "No models found in config toml. Please check the configuration."
            raise exceptions.ModelListNotFoundError(msg)

        if isinstance(model_entries, dict):
            # In case there's only one model (TOML parser returns dict)
            model_entries = [model_entries]

        model_list = [
            mxtoai.schemas.ModelConfig(
                model_name=entry.get("model_name"),
                litellm_params=mxtoai.schemas.LiteLLMParams(**entry.get("litellm_params")),
            )
            for entry in model_entries
        ]

        if not model_list:
            msg = "No model list found in config toml. Please check the configuration."
            raise exceptions.ModelListNotFoundError(msg)

        return model_list

    def _load_router_config(self) -> mxtoai.schemas.RouterConfig:
        """
        Load router configuration from environment variables.

        Returns:
           mxtoai.schemas.RouterConfig: Router configuration

        """
        router_config = mxtoai.schemas.RouterConfig(**self.config.get("router_config"))

        if not router_config:
            logger.warning("No router config found in model-config.toml. Using defaults.")
            return mxtoai.schemas.RouterConfig(
                routing_strategy="simple-shuffle",
                fallbacks=[],
                default_litellm_params={"drop_params": True},
            )
        return router_config

    def _set_azure_environment_variables(self, model_list: list[mxtoai.schemas.ModelConfig]) -> None:
        """
        Set Azure OpenAI environment variables from model configurations.

        This is required because LiteLLM's Azure provider checks for specific environment variables
        like AZURE_OPENAI_API_KEY even when API keys are provided in the model configuration.

        Args:
            model_list: List of model configurations from TOML

        """
        for model_config in model_list:
            if model_config.litellm_params.model and model_config.litellm_params.model.startswith("azure/"):
                # Set Azure OpenAI environment variables if they're not already set
                if model_config.litellm_params.api_key and not os.getenv("AZURE_OPENAI_API_KEY"):
                    os.environ["AZURE_OPENAI_API_KEY"] = model_config.litellm_params.api_key
                    logger.debug("Set AZURE_OPENAI_API_KEY from model configuration")

                if model_config.litellm_params.base_url and not os.getenv("AZURE_OPENAI_ENDPOINT"):
                    os.environ["AZURE_OPENAI_ENDPOINT"] = model_config.litellm_params.base_url
                    logger.debug("Set AZURE_OPENAI_ENDPOINT from model configuration")

                if model_config.litellm_params.api_version and not os.getenv("AZURE_OPENAI_API_VERSION"):
                    os.environ["AZURE_OPENAI_API_VERSION"] = model_config.litellm_params.api_version
                    logger.debug("Set AZURE_OPENAI_API_VERSION from model configuration")

                # We only need to set these once for Azure OpenAI
                break

    def _get_target_model(self) -> str:
        """
        Determine which model to route to based on the current handle configuration.

        Returns:
            str: The model name (group) to route to.

        """
        if self.current_handle and self.current_handle.target_model:
            logger.debug(
                f"Using model group {self.current_handle.target_model} for handle {self.current_handle.handle}"
            )
            return self.current_handle.target_model

        return "gpt-4"

    def generate(
        self,
        messages: list[dict[str, str | list[dict]]],
        stop_sequences: list[str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """
        Generate a response using either a local or remote LLM.

        Args:
            messages (list[dict[str, str | list[dict]]]): List of messages to process.
            stop_sequences (list[str] | None): List of stop sequences.
            tools_to_call_from (list[Tool] | None): List of tools available for calling.
            **kwargs: Additional arguments passed to the generate method.

        Returns:
            ChatMessage: The generated chat message.

        """
        # Check if this is a local LLM
        is_local_llm = (
            self.model_id.startswith("ollama")
            or (self.api_base and "localhost" in self.api_base)
            or (self.api_base and "127.0.0.1" in self.api_base)
        )

        # TODO: Get a permanent fix for this. Currently a temporary workaround
        if is_local_llm:
            litellm_messages = []
            for msg in messages:
                if isinstance(msg, str):
                    # skip the buggy char split strings coming in as message. This is a litellm bug.
                    continue
                litellm_messages.append(msg)
            messages = litellm_messages

        # Prepare completion kwargs - for local LLMs, exclude grammar and tools
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            tools_to_call_from=None if is_local_llm else tools_to_call_from,
            model=self.model_id,
            api_base=self.api_base,
            api_key=self.api_key,
            convert_images_to_image_urls=True,
            custom_role_conversions=self.custom_role_conversions,
            **kwargs,
        )

        # models under the 'thinking' group do not support stop sequences
        # This is a workaround for the current limitation in LiteLLMRouterModel
        if self.model_id == "thinking":
            completion_kwargs.pop("stop", None)

        response = self.client.completion(**completion_kwargs)

        # Safely handle response usage tracking
        if response and hasattr(response, "usage") and response.usage:
            self._last_input_token_count = response.usage.prompt_tokens
            self._last_output_token_count = response.usage.completion_tokens
        else:
            # Set default values if usage information is not available
            self._last_input_token_count = 0
            self._last_output_token_count = 0

        return ChatMessage.from_dict(
            response.choices[0].message.model_dump(include={"role", "content", "tool_calls"}),
            raw=response,
        )

    def __call__(
        self,
        messages: list[dict[str, Any]],  # MODIFIED type hint for messages
        stop_sequences: list[str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,  # kwargs from the caller of this RoutedLiteLLMModel instance
    ) -> ChatMessage:
        """
        Generate a response based on the provided messages and other parameters.

        Args:
            messages (list[dict[str, Any]]): List of messages to process.
            stop_sequences (Optional[list[str]]): List of stop sequences.
            tools_to_call_from (Optional[list[Tool]]): List of tools to call from.
            **kwargs: Additional arguments passed to the generate method.

        Returns:
            ChatMessage: The generated chat message.

        """
        try:
            target_model_group = self._get_target_model()

            # Temporarily set self.model_id to the target_model_group for this call.
            # This ensures that when LiteLLMModel.generate calls
            # self._prepare_completion_kwargs(model=self.model_id, ...),
            # it uses our desired target_model_group.
            original_smol_model_id = self.model_id
            self.model_id = target_model_group

            # Remove 'model' from kwargs if present, to prevent conflict with the
            # explicit 'model=self.model_id' passed by LiteLLMModel.generate
            # to _prepare_completion_kwargs.
            kwargs_for_super_generate = {k: v for k, v in kwargs.items() if k != "model"}

            try:
                chat_message = self.generate(
                    messages=messages,
                    stop_sequences=stop_sequences,
                    tools_to_call_from=tools_to_call_from,
                    **kwargs_for_super_generate,
                )
            finally:
                self.model_id = original_smol_model_id

        except Exception as e:
            logger.error(f"Error in RoutedLiteLLMModel completion: {e!s}")
            msg = f"Failed to get completion from LiteLLM router: {e!s}"
            raise RuntimeError(msg) from e
        else:
            return chat_message

    @property
    def last_input_token_count(self) -> int:
        """Safely return the last input token count."""
        return getattr(self, "_last_input_token_count", 0)

    @last_input_token_count.setter
    def last_input_token_count(self, value: int | None) -> None:
        """Safely set the last input token count."""
        self._last_input_token_count = value if value is not None else 0

    @property
    def last_output_token_count(self) -> int:
        """Safely return the last output token count."""
        return getattr(self, "_last_output_token_count", 0)

    @last_output_token_count.setter
    def last_output_token_count(self, value: int | None) -> None:
        """Safely set the last output token count."""
        self._last_output_token_count = value if value is not None else 0

    def __getattr__(self, name: str):
        """Handle any missing attribute access gracefully, especially for token-related properties."""
        # Handle various token-related attribute names that might be accessed
        if name in ("input_tokens", "output_tokens", "total_tokens", "prompt_tokens", "completion_tokens", "usage"):
            return 0
        # For other missing attributes, raise AttributeError as normal
        msg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
        raise AttributeError(msg)
