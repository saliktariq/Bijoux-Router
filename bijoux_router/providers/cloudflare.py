"""Cloudflare Workers AI adapter."""

from __future__ import annotations

import time
from typing import Any

import httpx

from bijoux_router.config.schema import ProviderConfig
from bijoux_router.exceptions.errors import (
    AuthenticationError,
    InsufficientCreditError,
    ModelUnavailableError,
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
from bijoux_router.utils.logging import get_logger

logger = get_logger("providers.cloudflare")


class CloudflareClient(BaseProviderClient):
    """Adapter for Cloudflare Workers AI REST API.

    The Cloudflare AI API uses:
    - Bearer token auth
    - Account-scoped URLs: /client/v4/accounts/{account_id}/ai/run/{model}
    - Its own response wrapper with ``result`` / ``success`` keys
    - config.metadata["account_id"] must be set (or embed account_id in base_url)

    Alternatively, Cloudflare exposes an OpenAI-compatible gateway at
    ``https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_slug}/openai``
    — if you prefer that route, just use provider_type: openai_compatible with
    that base_url.  This adapter uses the *native* Workers AI endpoint.
    """

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._account_id: str = config.metadata.get("account_id", "")
        base = config.base_url or "https://api.cloudflare.com"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
            **config.headers,
        }
        self._client = httpx.AsyncClient(
            base_url=base.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(config.timeout_seconds),
        )

    # ------------------------------------------------------------------ #
    # Request dispatch
    # ------------------------------------------------------------------ #

    async def send_request(self, request: LLMRequest) -> LLMResponse:
        model = self.config.resolve_model(request.model)
        messages_raw = request.effective_messages()

        messages: list[dict[str, str]] = [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages_raw
        ]

        payload: dict[str, Any] = {
            "messages": messages,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p

        # Endpoint: /client/v4/accounts/{account_id}/ai/run/{model}
        url = f"/client/v4/accounts/{self._account_id}/ai/run/{model}"

        timeout = request.timeout_override or self.config.timeout_seconds
        t0 = time.perf_counter()
        try:
            resp = await self._client.post(url, json=payload, timeout=timeout)
        except httpx.TimeoutException as exc:
            raise TransientProviderError(
                f"Timeout calling Cloudflare AI ({self.name})",
                provider_name=self.name,
                category=ProviderErrorCategory.TIMEOUT,
            ) from exc
        except httpx.ConnectError as exc:
            raise TransientProviderError(
                f"Connection error calling Cloudflare AI ({self.name}): {exc}",
                provider_name=self.name,
                category=ProviderErrorCategory.NETWORK_ERROR,
            ) from exc

        latency_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code >= 400:
            self._raise_for_status(resp)

        body = resp.json()

        # Cloudflare wraps responses: {"result": {...}, "success": true}
        result = body.get("result", body)
        content = result.get("response", "")
        usage = self._extract_cf_usage(result)

        return LLMResponse(
            request_id=request.request_id,
            content=content,
            provider_name=self.name,
            model=model,
            usage=usage,
            finish_reason=FinishReason.STOP,
            latency_ms=latency_ms,
            raw_response=body,
        )

    # ------------------------------------------------------------------ #
    # Response helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_cf_usage(result: dict) -> TokenUsage:
        usage = result.get("usage", {})
        prompt_t = usage.get("prompt_tokens", 0)
        completion_t = usage.get("completion_tokens", 0)
        total = usage.get("total_tokens", prompt_t + completion_t)
        return TokenUsage(
            prompt_tokens=prompt_t,
            completion_tokens=completion_t,
            total_tokens=total,
        )

    # ------------------------------------------------------------------ #
    # Error handling
    # ------------------------------------------------------------------ #

    def _raise_for_status(self, resp: httpx.Response) -> None:
        try:
            body = resp.json()
        except Exception:
            body = resp.text

        category = self.classify_error(resp.status_code, body)
        error_msg = ""
        if isinstance(body, dict):
            errors = body.get("errors", [])
            if errors and isinstance(errors[0], dict):
                error_msg = errors[0].get("message", str(errors))
            else:
                error_msg = str(errors or body.get("error", ""))
        else:
            error_msg = str(body)[:500]

        full_msg = f"[{self.name}] Cloudflare HTTP {resp.status_code}: {error_msg}"
        exc_map: dict[ProviderErrorCategory, type[ProviderError]] = {
            ProviderErrorCategory.QUOTA_EXHAUSTED: QuotaExhaustedError,
            ProviderErrorCategory.RATE_LIMITED: QuotaExhaustedError,
            ProviderErrorCategory.INSUFFICIENT_CREDIT: InsufficientCreditError,
            ProviderErrorCategory.AUTH_ERROR: AuthenticationError,
            ProviderErrorCategory.MODEL_UNAVAILABLE: ModelUnavailableError,
        }
        exc_cls = exc_map.get(category, TransientProviderError)
        raise exc_cls(
            full_msg,
            provider_name=self.name,
            category=category,
            status_code=resp.status_code,
            raw_response=body,
        )

    def classify_error(self, status_code: int | None, body: dict | str | None) -> ProviderErrorCategory:
        error_text = ""
        if isinstance(body, dict):
            errors = body.get("errors", [])
            if errors and isinstance(errors[0], dict):
                error_text = errors[0].get("message", "").lower()
            else:
                error_text = str(body).lower()
        elif isinstance(body, str):
            error_text = body.lower()

        if status_code == 429 or "rate limit" in error_text:
            return ProviderErrorCategory.RATE_LIMITED
        if "billing" in error_text or "credit" in error_text:
            return ProviderErrorCategory.INSUFFICIENT_CREDIT
        if status_code == 401 or "authentication" in error_text or "unauthorized" in error_text:
            return ProviderErrorCategory.AUTH_ERROR
        if status_code == 403:
            return ProviderErrorCategory.AUTH_ERROR
        if status_code == 404 or "not found" in error_text or "unknown model" in error_text:
            return ProviderErrorCategory.MODEL_UNAVAILABLE
        if status_code == 400 or "invalid" in error_text:
            return ProviderErrorCategory.INVALID_REQUEST
        if status_code and 500 <= status_code < 600:
            return ProviderErrorCategory.TRANSIENT_ERROR
        return ProviderErrorCategory.UNKNOWN_ERROR

    async def close(self) -> None:
        await self._client.aclose()
