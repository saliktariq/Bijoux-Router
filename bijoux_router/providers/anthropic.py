"""Anthropic Claude Messages API adapter."""

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

logger = get_logger("providers.anthropic")

_FINISH_REASON_MAP: dict[str | None, FinishReason] = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "max_tokens": FinishReason.LENGTH,
    "tool_use": FinishReason.TOOL_CALLS,
}


class AnthropicClient(BaseProviderClient):
    """Adapter for the Anthropic Messages API.

    Anthropic uses a proprietary format:
    - System prompt is a top-level parameter, not a message
    - Content blocks can be structured (text, tool_use, etc.)
    - Usage is reported as input_tokens / output_tokens
    - Auth uses x-api-key header, not Bearer token
    """

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        base = config.base_url or "https://api.anthropic.com"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": config.api_key,
            "anthropic-version": "2023-06-01",
            **config.headers,
        }
        self._client = httpx.AsyncClient(
            base_url=base.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(config.timeout_seconds),
        )

    async def send_request(self, request: LLMRequest) -> LLMResponse:
        model = self.config.resolve_model(request.model)
        messages_raw = request.effective_messages()

        # Anthropic: extract system messages as a top-level param
        system_parts: list[str] = []
        messages: list[dict[str, str]] = []
        for msg in messages_raw:
            if msg.role.value == "system":
                system_parts.append(msg.content)
            else:
                # Anthropic only supports "user" and "assistant" roles
                role = "user" if msg.role.value in ("user", "tool") else "assistant"
                messages.append({"role": role, "content": msg.content})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens or 1024,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop_sequences"] = request.stop

        timeout = request.timeout_override or self.config.timeout_seconds
        t0 = time.perf_counter()
        try:
            resp = await self._client.post(
                "/v1/messages",
                json=payload,
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise TransientProviderError(
                f"Timeout calling Anthropic ({self.name})",
                provider_name=self.name,
                category=ProviderErrorCategory.TIMEOUT,
            ) from exc
        except httpx.ConnectError as exc:
            raise TransientProviderError(
                f"Connection error calling Anthropic ({self.name}): {exc}",
                provider_name=self.name,
                category=ProviderErrorCategory.NETWORK_ERROR,
            ) from exc

        latency_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code >= 400:
            self._raise_for_status(resp)

        body = resp.json()
        content = self._extract_content(body)
        usage = self._extract_anthropic_usage(body)
        stop_reason = body.get("stop_reason")
        finish_reason = _FINISH_REASON_MAP.get(stop_reason, FinishReason.UNKNOWN)

        return LLMResponse(
            request_id=request.request_id,
            content=content,
            provider_name=self.name,
            model=body.get("model", model),
            usage=usage,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            raw_response=body,
        )

    @staticmethod
    def _extract_content(body: dict) -> str:
        """Extract text from Anthropic content blocks."""
        content_blocks = body.get("content", [])
        parts: list[str] = []
        for block in content_blocks:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)

    @staticmethod
    def _extract_anthropic_usage(body: dict) -> TokenUsage:
        usage = body.get("usage", {})
        input_t = usage.get("input_tokens", 0)
        output_t = usage.get("output_tokens", 0)
        return TokenUsage(
            prompt_tokens=input_t,
            completion_tokens=output_t,
            total_tokens=input_t + output_t,
        )

    def _raise_for_status(self, resp: httpx.Response) -> None:
        try:
            body = resp.json()
        except Exception:
            body = resp.text

        category = self.classify_error(resp.status_code, body)
        error_msg = ""
        if isinstance(body, dict):
            err = body.get("error", {})
            error_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        else:
            error_msg = str(body)[:500]

        full_msg = f"[{self.name}] Anthropic HTTP {resp.status_code}: {error_msg}"
        exc_map: dict[ProviderErrorCategory, type[ProviderError]] = {
            ProviderErrorCategory.QUOTA_EXHAUSTED: QuotaExhaustedError,
            ProviderErrorCategory.RATE_LIMITED: QuotaExhaustedError,
            ProviderErrorCategory.INSUFFICIENT_CREDIT: InsufficientCreditError,
            ProviderErrorCategory.BILLING_BLOCKED: InsufficientCreditError,
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
            err = body.get("error", {})
            if isinstance(err, dict):
                error_text = (err.get("message", "") + " " + err.get("type", "")).lower()
            else:
                error_text = str(err).lower()
        elif isinstance(body, str):
            error_text = body.lower()

        if status_code == 429 or "rate_limit" in error_text:
            return ProviderErrorCategory.RATE_LIMITED
        if "credit" in error_text or "billing" in error_text or status_code == 402:
            return ProviderErrorCategory.INSUFFICIENT_CREDIT
        if status_code == 401 or "authentication" in error_text:
            return ProviderErrorCategory.AUTH_ERROR
        if status_code == 403:
            if "credit" in error_text or "billing" in error_text:
                return ProviderErrorCategory.BILLING_BLOCKED
            return ProviderErrorCategory.AUTH_ERROR
        if status_code == 404 or "not_found" in error_text:
            return ProviderErrorCategory.MODEL_UNAVAILABLE
        if status_code == 400 or "invalid_request" in error_text:
            return ProviderErrorCategory.INVALID_REQUEST
        if "overloaded" in error_text or status_code == 529:
            return ProviderErrorCategory.TRANSIENT_ERROR
        if status_code and 500 <= status_code < 600:
            return ProviderErrorCategory.TRANSIENT_ERROR
        return ProviderErrorCategory.UNKNOWN_ERROR

    async def close(self) -> None:
        await self._client.aclose()
