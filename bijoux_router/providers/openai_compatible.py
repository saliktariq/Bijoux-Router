"""OpenAI-compatible provider adapter (covers OpenRouter, vLLM, etc.)."""

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

logger = get_logger("providers.openai_compat")

_FINISH_REASON_MAP: dict[str | None, FinishReason] = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "content_filter": FinishReason.CONTENT_FILTER,
    "tool_calls": FinishReason.TOOL_CALLS,
}


class OpenAICompatibleClient(BaseProviderClient):
    """Adapter for any OpenAI-compatible chat completions API."""

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        headers = {"Content-Type": "application/json", **config.headers}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(config.timeout_seconds),
        )

    async def send_request(self, request: LLMRequest) -> LLMResponse:
        model = self.config.resolve_model(request.model)
        messages = [{"role": m.role.value, "content": m.content} for m in request.effective_messages()]
        payload: dict[str, Any] = {"model": model, "messages": messages}

        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.stop:
            payload["stop"] = request.stop

        timeout = request.timeout_override or self.config.timeout_seconds
        t0 = time.perf_counter()
        try:
            resp = await self._client.post(
                "/chat/completions",
                json=payload,
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise TransientProviderError(
                f"Timeout calling {self.name}",
                provider_name=self.name,
                category=ProviderErrorCategory.TIMEOUT,
            ) from exc
        except httpx.ConnectError as exc:
            raise TransientProviderError(
                f"Connection error calling {self.name}: {exc}",
                provider_name=self.name,
                category=ProviderErrorCategory.NETWORK_ERROR,
            ) from exc

        latency_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code >= 400:
            self._raise_for_status(resp)

        body = resp.json()
        usage = self.extract_usage(body) or TokenUsage()
        content = ""
        finish_reason = FinishReason.UNKNOWN
        choices = body.get("choices", [])
        if choices:
            choice = choices[0]
            msg = choice.get("message", {})
            content = msg.get("content", "") or ""
            fr = choice.get("finish_reason")
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

    def _raise_for_status(self, resp: httpx.Response) -> None:
        try:
            body = resp.json()
        except Exception:
            body = resp.text

        category = self.classify_error(resp.status_code, body)
        error_msg = ""
        if isinstance(body, dict):
            error_msg = body.get("error", {}).get("message", "") if isinstance(body.get("error"), dict) else str(body.get("error", ""))
        else:
            error_msg = str(body)[:500]

        full_msg = f"[{self.name}] HTTP {resp.status_code}: {error_msg}"
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
                error_text = (err.get("message", "") + " " + err.get("code", "")).lower()
            else:
                error_text = str(err).lower()
        elif isinstance(body, str):
            error_text = body.lower()

        # Check for quota/billing signals in the error text
        quota_signals = ["quota", "rate limit", "rate_limit", "too many requests", "resource_exhausted"]
        credit_signals = ["credit", "balance", "billing", "payment", "insufficient", "exceeded"]

        if status_code == 429 or any(s in error_text for s in quota_signals):
            return ProviderErrorCategory.RATE_LIMITED
        if any(s in error_text for s in credit_signals):
            return ProviderErrorCategory.INSUFFICIENT_CREDIT
        if status_code == 401:
            return ProviderErrorCategory.AUTH_ERROR
        if status_code == 403:
            if any(s in error_text for s in credit_signals):
                return ProviderErrorCategory.BILLING_BLOCKED
            return ProviderErrorCategory.AUTH_ERROR
        if status_code == 404 or "model" in error_text and "not found" in error_text:
            return ProviderErrorCategory.MODEL_UNAVAILABLE
        if status_code == 400:
            return ProviderErrorCategory.INVALID_REQUEST
        if status_code and 500 <= status_code < 600:
            return ProviderErrorCategory.TRANSIENT_ERROR
        return ProviderErrorCategory.UNKNOWN_ERROR

    async def close(self) -> None:
        await self._client.aclose()
