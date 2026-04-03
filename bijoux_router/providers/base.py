"""Abstract base class for all LLM provider adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from bijoux_router.config.schema import ProviderConfig
from bijoux_router.exceptions.errors import ProviderErrorCategory
from bijoux_router.models.request_response import LLMRequest, LLMResponse, TokenEstimate, TokenUsage


class BaseProviderClient(ABC):
    """Contract every provider adapter must fulfill."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.name = config.name

    @abstractmethod
    async def send_request(self, request: LLMRequest) -> LLMResponse:
        """Send a normalized request to this provider and return a normalized response.

        Must raise a ProviderError subclass on failure, with the correct
        `category` so the router can decide on retry/failover.
        """

    @abstractmethod
    def classify_error(self, status_code: int | None, body: dict | str | None) -> ProviderErrorCategory:
        """Map a provider-specific error response into the normalized error taxonomy."""

    def estimate_tokens(self, request: LLMRequest) -> TokenEstimate:
        """Estimate token usage for quota reservation. Default uses char heuristic."""
        from bijoux_router.utils.tokens import estimate_tokens
        return estimate_tokens(request)

    def extract_usage(self, raw_response: dict) -> TokenUsage | None:
        """Extract actual token usage from the raw provider response.

        Returns None if the provider did not include usage information.
        """
        usage = raw_response.get("usage")
        if not usage:
            return None
        return TokenUsage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    async def close(self) -> None:
        """Release any resources (HTTP client, etc.)."""
