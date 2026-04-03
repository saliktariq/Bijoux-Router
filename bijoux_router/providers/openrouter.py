"""OpenRouter-specific adapter extending the OpenAI-compatible client."""

from __future__ import annotations

from bijoux_router.config.schema import ProviderConfig
from bijoux_router.providers.openai_compatible import OpenAICompatibleClient


class OpenRouterClient(OpenAICompatibleClient):
    """Adapter for the OpenRouter API.

    OpenRouter uses the OpenAI chat completions format but has
    provider-specific headers and error responses.
    """

    def __init__(self, config: ProviderConfig) -> None:
        if not config.base_url:
            config = config.model_copy(update={"base_url": "https://openrouter.ai/api/v1"})
        # Ensure OpenRouter-required headers
        extra_headers = dict(config.headers)
        if config.api_key:
            extra_headers.setdefault("Authorization", f"Bearer {config.api_key}")
        extra_headers.setdefault("HTTP-Referer", "https://github.com/bijoux-router")
        extra_headers.setdefault("X-Title", "Bijoux Router")
        config = config.model_copy(update={"headers": extra_headers})
        super().__init__(config)
