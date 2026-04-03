"""Mock provider for testing — returns controlled responses and errors."""

from __future__ import annotations

import time
from typing import Any

from bijoux_router.config.schema import ProviderConfig
from bijoux_router.exceptions.errors import (
    ProviderError,
    ProviderErrorCategory,
    QuotaExhaustedError,
    TransientProviderError,
)
from bijoux_router.models.request_response import (
    FinishReason,
    LLMRequest,
    LLMResponse,
    TokenUsage,
)
from bijoux_router.providers.base import BaseProviderClient


class MockProviderClient(BaseProviderClient):
    """Controllable mock provider for tests and simulations.

    Behaviour is driven by metadata on the ProviderConfig or by
    setting class attributes before calling send_request:
      - config.metadata["mock_error"]: ProviderErrorCategory value to raise
      - config.metadata["mock_content"]: response content string
      - config.metadata["mock_latency_ms"]: simulated latency
      - config.metadata["mock_usage"]: dict with prompt_tokens, completion_tokens
    """

    call_count: int = 0
    last_request: LLMRequest | None = None

    # Allow injecting behaviour at instance level
    force_error: ProviderErrorCategory | None = None
    force_content: str | None = None

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self.call_count = 0
        self.last_request = None
        self.force_error = None
        self.force_content = None

    async def send_request(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        self.last_request = request

        meta = self.config.metadata
        error_cat = self.force_error or meta.get("mock_error")
        if error_cat:
            if isinstance(error_cat, str):
                error_cat = ProviderErrorCategory(error_cat)
            self._raise_mock_error(error_cat)

        content = self.force_content or meta.get("mock_content", f"Mock response from {self.name}")
        latency = float(meta.get("mock_latency_ms", 10))
        usage_meta = meta.get("mock_usage", {})
        prompt_t = int(usage_meta.get("prompt_tokens", 10))
        completion_t = int(usage_meta.get("completion_tokens", 20))

        return LLMResponse(
            request_id=request.request_id,
            content=content,
            provider_name=self.name,
            model=self.config.resolve_model(request.model),
            usage=TokenUsage.from_counts(prompt=prompt_t, completion=completion_t),
            finish_reason=FinishReason.STOP,
            latency_ms=latency,
            raw_response={"mock": True},
        )

    def _raise_mock_error(self, category: ProviderErrorCategory) -> None:
        if category in (
            ProviderErrorCategory.QUOTA_EXHAUSTED,
            ProviderErrorCategory.RATE_LIMITED,
        ):
            raise QuotaExhaustedError(
                f"Mock quota exhausted on {self.name}",
                provider_name=self.name,
                status_code=429,
            )
        if category == ProviderErrorCategory.INSUFFICIENT_CREDIT:
            from bijoux_router.exceptions.errors import InsufficientCreditError
            raise InsufficientCreditError(
                f"Mock insufficient credit on {self.name}",
                provider_name=self.name,
                status_code=402,
            )
        if category == ProviderErrorCategory.AUTH_ERROR:
            from bijoux_router.exceptions.errors import AuthenticationError
            raise AuthenticationError(
                f"Mock auth error on {self.name}",
                provider_name=self.name,
                status_code=401,
            )
        if category == ProviderErrorCategory.MODEL_UNAVAILABLE:
            from bijoux_router.exceptions.errors import ModelUnavailableError
            raise ModelUnavailableError(
                f"Mock model unavailable on {self.name}",
                provider_name=self.name,
                status_code=404,
            )
        raise TransientProviderError(
            f"Mock error ({category.value}) on {self.name}",
            provider_name=self.name,
            category=category,
        )

    def classify_error(self, status_code: int | None, body: dict | str | None) -> ProviderErrorCategory:
        return ProviderErrorCategory.UNKNOWN_ERROR
