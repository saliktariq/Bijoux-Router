"""Cohere Chat API adapter."""

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

logger = get_logger("providers.cohere")

_FINISH_REASON_MAP: dict[str | None, FinishReason] = {
    "COMPLETE": FinishReason.STOP,
    "MAX_TOKENS": FinishReason.LENGTH,
    "STOP_SEQUENCE": FinishReason.STOP,
    "TOOL_CALL": FinishReason.TOOL_CALLS,
    "ERROR": FinishReason.ERROR,
}


class CohereClient(BaseProviderClient):
    """Adapter for the Cohere v2 Chat API.

    Cohere v2 uses:
    - Bearer token auth
    - Its own message format with role/content
    - Usage as billed_units or tokens in meta
    - finish_reason as a string enum
    """

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        base = config.base_url or "https://api.cohere.com"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
            "Accept": "application/json",
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

        # Cohere v2 chat: messages with role (system, user, assistant, tool)
        messages: list[dict[str, Any]] = []
        for msg in messages_raw:
            role = msg.role.value
            if role == "tool":
                role = "user"
            messages.append({"role": role, "content": msg.content})

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["p"] = request.top_p
        if request.stop:
            payload["stop_sequences"] = request.stop

        timeout = request.timeout_override or self.config.timeout_seconds
        t0 = time.perf_counter()
        try:
            resp = await self._client.post(
                "/v2/chat",
                json=payload,
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise TransientProviderError(
                f"Timeout calling Cohere ({self.name})",
                provider_name=self.name,
                category=ProviderErrorCategory.TIMEOUT,
            ) from exc
        except httpx.ConnectError as exc:
            raise TransientProviderError(
                f"Connection error calling Cohere ({self.name}): {exc}",
                provider_name=self.name,
                category=ProviderErrorCategory.NETWORK_ERROR,
            ) from exc

        latency_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code >= 400:
            self._raise_for_status(resp)

        body = resp.json()
        content = self._extract_content(body)
        usage = self._extract_cohere_usage(body)
        fr = body.get("finish_reason")
        finish_reason = _FINISH_REASON_MAP.get(fr, FinishReason.UNKNOWN)

        return LLMResponse(
            request_id=request.request_id,
            content=content,
            provider_name=self.name,
            model=model,
            usage=usage,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            raw_response=body,
        )

    @staticmethod
    def _extract_content(body: dict) -> str:
        """Extract text from Cohere v2 response."""
        # v2 format: message.content[].text
        message = body.get("message", {})
        content_blocks = message.get("content", [])
        if isinstance(content_blocks, list):
            return "".join(
                block.get("text", "") for block in content_blocks if block.get("type") == "text"
            )
        # Fallback: direct text field
        text = body.get("text", "")
        if text:
            return text
        return ""

    @staticmethod
    def _extract_cohere_usage(body: dict) -> TokenUsage:
        """Extract token usage from Cohere response meta."""
        usage = body.get("usage", {})
        tokens = usage.get("tokens", {})
        input_t = tokens.get("input_tokens", 0)
        output_t = tokens.get("output_tokens", 0)
        if input_t == 0 and output_t == 0:
            # Fallback to billed_units
            billed = usage.get("billed_units", {})
            input_t = billed.get("input_tokens", 0)
            output_t = billed.get("output_tokens", 0)
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
            error_msg = body.get("message", str(body.get("error", "")))
        else:
            error_msg = str(body)[:500]

        full_msg = f"[{self.name}] Cohere HTTP {resp.status_code}: {error_msg}"
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
            error_text = (body.get("message", "") + " " + str(body.get("error", ""))).lower()
        elif isinstance(body, str):
            error_text = body.lower()

        if status_code == 429 or "rate limit" in error_text or "too many requests" in error_text:
            return ProviderErrorCategory.RATE_LIMITED
        if "billing" in error_text or "credit" in error_text or "payment" in error_text:
            return ProviderErrorCategory.INSUFFICIENT_CREDIT
        if status_code == 401:
            return ProviderErrorCategory.AUTH_ERROR
        if status_code == 403:
            return ProviderErrorCategory.AUTH_ERROR
        if status_code == 404 or "not found" in error_text:
            return ProviderErrorCategory.MODEL_UNAVAILABLE
        if status_code == 400:
            return ProviderErrorCategory.INVALID_REQUEST
        if status_code and 500 <= status_code < 600:
            return ProviderErrorCategory.TRANSIENT_ERROR
        return ProviderErrorCategory.UNKNOWN_ERROR

    async def close(self) -> None:
        await self._client.aclose()
