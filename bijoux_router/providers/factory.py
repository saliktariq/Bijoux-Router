"""Provider factory — maps provider_type strings to adapter classes."""

from __future__ import annotations

from bijoux_router.config.schema import ProviderConfig
from bijoux_router.exceptions.errors import ConfigurationError
from bijoux_router.providers.base import BaseProviderClient
from bijoux_router.providers.anthropic import AnthropicClient
from bijoux_router.providers.cloudflare import CloudflareClient
from bijoux_router.providers.cohere import CohereClient
from bijoux_router.providers.gemini import GeminiClient
from bijoux_router.providers.huggingface import HuggingFaceClient
from bijoux_router.providers.mock import MockProviderClient
from bijoux_router.providers.openai_compatible import OpenAICompatibleClient
from bijoux_router.providers.openrouter import OpenRouterClient
from bijoux_router.providers.replicate import ReplicateClient

_REGISTRY: dict[str, type[BaseProviderClient]] = {
    "openrouter": OpenRouterClient,
    "gemini": GeminiClient,
    "anthropic": AnthropicClient,
    "cohere": CohereClient,
    "cloudflare": CloudflareClient,
    "huggingface": HuggingFaceClient,
    "replicate": ReplicateClient,
    "openai_compatible": OpenAICompatibleClient,
    "openai": OpenAICompatibleClient,
    "mock": MockProviderClient,
}


def register_provider_type(name: str, cls: type[BaseProviderClient]) -> None:
    """Register a custom provider adapter type for plugin extensibility."""
    _REGISTRY[name] = cls


def create_provider(config: ProviderConfig) -> BaseProviderClient:
    """Instantiate a provider client from config."""
    cls = _REGISTRY.get(config.provider_type)
    if cls is None:
        raise ConfigurationError(
            f"Unknown provider_type '{config.provider_type}' for provider '{config.name}'. "
            f"Available types: {', '.join(sorted(_REGISTRY.keys()))}"
        )
    return cls(config)
