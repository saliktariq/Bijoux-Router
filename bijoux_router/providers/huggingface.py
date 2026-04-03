"""Hugging Face Inference API adapter."""

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

logger = get_logger("providers.huggingface")

_FINISH_REASON_MAP: dict[str | None, FinishReason] = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "eos_token": FinishReason.STOP,
}


class HuggingFaceClient(BaseProviderClient):
    """Adapter for the Hugging Face Inference API (serverless & dedicated).

    HF Inference API exposes two forms:
    1. **OpenAI-compatible** ``/v1/chat/completions`` — for chat models on
       dedicated inference endpoints or through the serverless API.  If you
       prefer this route, use ``provider_type: openai_compatible`` with
       ``base_url: https://api-inference.huggingface.co`` (or your dedicated
       endpoint URL) and it will just work.

    2. **Native text-generation-inference (TGI)** format using ``/models/{model}``
       — this adapter handles *this* variant.

    Auth: Bearer token (``api_key``).

    Config:
    - ``base_url``: defaults to ``https://api-inference.huggingface.co``
    - ``metadata.use_chat_endpoint``: if truthy, uses ``/v1/chat/completions``
      instead of the native ``/models/{model}`` endpoint.
    """

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        base = config.base_url or "https://api-inference.huggingface.co"
        self._use_chat = bool(config.metadata.get("use_chat_endpoint", False))
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

        if self._use_chat:
            return await self._send_chat(request, model)
        return await self._send_tgi(request, model)

    async def _send_chat(self, request: LLMRequest, model: str) -> LLMResponse:
        """OpenAI-compatible chat path on HF."""
        messages_raw = request.effective_messages()
        messages = [{"role": m.role.value, "content": m.content} for m in messages_raw]

        payload: dict[str, Any] = {"model": model, "messages": messages}
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.stop:
            payload["stop"] = request.stop

        timeout = request.timeout_override or self.config.timeout_seconds
        t0 = time.perf_counter()
        try:
            resp = await self._client.post("/v1/chat/completions", json=payload, timeout=timeout)
        except httpx.TimeoutException as exc:
            raise TransientProviderError(
                f"Timeout calling HF chat ({self.name})",
                provider_name=self.name,
                category=ProviderErrorCategory.TIMEOUT,
            ) from exc
        except httpx.ConnectError as exc:
            raise TransientProviderError(
                f"Connection error calling HF chat ({self.name}): {exc}",
                provider_name=self.name,
                category=ProviderErrorCategory.NETWORK_ERROR,
            ) from exc

        latency_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code >= 400:
            self._raise_for_status(resp)

        body = resp.json()
        choice = body.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        usage_raw = body.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
        )
        fr = choice.get("finish_reason")
        finish_reason = _FINISH_REASON_MAP.get(fr, FinishReason.UNKNOWN)

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

    async def _send_tgi(self, request: LLMRequest, model: str) -> LLMResponse:
        """Native TGI / HF Inference API path."""
        messages_raw = request.effective_messages()
        # Build a plain-text prompt from messages for the text generation API
        prompt = "\n".join(f"{m.role.value}: {m.content}" for m in messages_raw)

        parameters: dict[str, Any] = {}
        if request.max_tokens is not None:
            parameters["max_new_tokens"] = request.max_tokens
        if request.temperature is not None:
            parameters["temperature"] = request.temperature
        if request.top_p is not None:
            parameters["top_p"] = request.top_p
        if request.stop:
            parameters["stop_sequences"] = request.stop

        payload: dict[str, Any] = {"inputs": prompt}
        if parameters:
            payload["parameters"] = parameters

        url = f"/models/{model}"
        timeout = request.timeout_override or self.config.timeout_seconds
        t0 = time.perf_counter()
        try:
            resp = await self._client.post(url, json=payload, timeout=timeout)
        except httpx.TimeoutException as exc:
            raise TransientProviderError(
                f"Timeout calling HF Inference ({self.name})",
                provider_name=self.name,
                category=ProviderErrorCategory.TIMEOUT,
            ) from exc
        except httpx.ConnectError as exc:
            raise TransientProviderError(
                f"Connection error calling HF Inference ({self.name}): {exc}",
                provider_name=self.name,
                category=ProviderErrorCategory.NETWORK_ERROR,
            ) from exc

        latency_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code >= 400:
            self._raise_for_status(resp)

        body = resp.json()
        content = self._extract_tgi_content(body)

        return LLMResponse(
            request_id=request.request_id,
            content=content,
            provider_name=self.name,
            model=model,
            usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            finish_reason=FinishReason.STOP,
            latency_ms=latency_ms,
            raw_response=body,
        )

    # ------------------------------------------------------------------ #
    # Response helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_tgi_content(body: Any) -> str:
        """Handle both single-result and list-of-results TGI format."""
        if isinstance(body, list):
            parts = [item.get("generated_text", "") for item in body if isinstance(item, dict)]
            return "".join(parts)
        if isinstance(body, dict):
            return body.get("generated_text", "")
        return str(body)

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
            error_msg = body.get("error", str(body))
        elif isinstance(body, str):
            error_msg = body[:500]
        else:
            error_msg = str(body)[:500]

        full_msg = f"[{self.name}] HuggingFace HTTP {resp.status_code}: {error_msg}"
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
            error_text = str(body.get("error", "")).lower()
        elif isinstance(body, str):
            error_text = body.lower()

        if status_code == 429 or "rate limit" in error_text or "too many requests" in error_text:
            return ProviderErrorCategory.RATE_LIMITED
        if "billing" in error_text or "credit" in error_text:
            return ProviderErrorCategory.INSUFFICIENT_CREDIT
        if status_code == 401 or "unauthorized" in error_text:
            return ProviderErrorCategory.AUTH_ERROR
        if status_code == 403:
            return ProviderErrorCategory.AUTH_ERROR
        if status_code == 404 or "not found" in error_text or "model" in error_text and "does not exist" in error_text:
            return ProviderErrorCategory.MODEL_UNAVAILABLE
        if "loading" in error_text or "currently loading" in error_text:
            return ProviderErrorCategory.TRANSIENT_ERROR
        if status_code == 400:
            return ProviderErrorCategory.INVALID_REQUEST
        if status_code == 503 or "service unavailable" in error_text:
            return ProviderErrorCategory.TRANSIENT_ERROR
        if status_code and 500 <= status_code < 600:
            return ProviderErrorCategory.TRANSIENT_ERROR
        return ProviderErrorCategory.UNKNOWN_ERROR

    async def close(self) -> None:
        await self._client.aclose()
