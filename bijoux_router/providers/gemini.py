"""Google Gemini API adapter."""

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

logger = get_logger("providers.gemini")


class GeminiClient(BaseProviderClient):
    """Adapter for the Google Gemini (Generative AI) REST API.

    Uses the generateContent endpoint with API key auth.
    """

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        base = config.base_url or "https://generativelanguage.googleapis.com/v1beta"
        self._base_url = base.rstrip("/")
        self._client = httpx.AsyncClient(
            headers={"Content-Type": "application/json", **config.headers},
            timeout=httpx.Timeout(config.timeout_seconds),
        )

    async def send_request(self, request: LLMRequest) -> LLMResponse:
        model = self.config.resolve_model(request.model)
        # Build Gemini-style contents from messages
        contents = self._build_contents(request)
        generation_config: dict[str, Any] = {}
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.top_p is not None:
            generation_config["topP"] = request.top_p
        if request.max_tokens is not None:
            generation_config["maxOutputTokens"] = request.max_tokens
        if request.stop:
            generation_config["stopSequences"] = request.stop

        payload: dict[str, Any] = {"contents": contents}
        if generation_config:
            payload["generationConfig"] = generation_config

        url = f"{self._base_url}/models/{model}:generateContent"
        params = {"key": self.config.api_key} if self.config.api_key else {}

        timeout = request.timeout_override or self.config.timeout_seconds
        t0 = time.perf_counter()
        try:
            resp = await self._client.post(url, json=payload, params=params, timeout=timeout)
        except httpx.TimeoutException as exc:
            raise TransientProviderError(
                f"Timeout calling Gemini ({self.name})",
                provider_name=self.name,
                category=ProviderErrorCategory.TIMEOUT,
            ) from exc
        except httpx.ConnectError as exc:
            raise TransientProviderError(
                f"Connection error calling Gemini ({self.name}): {exc}",
                provider_name=self.name,
                category=ProviderErrorCategory.NETWORK_ERROR,
            ) from exc

        latency_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code >= 400:
            self._raise_for_status(resp)

        body = resp.json()
        content = self._extract_content(body)
        usage = self._extract_gemini_usage(body)
        finish_reason = self._map_finish_reason(body)

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
    def _build_contents(request: LLMRequest) -> list[dict[str, Any]]:
        """Convert normalized messages to Gemini contents format."""
        contents: list[dict[str, Any]] = []
        for msg in request.effective_messages():
            role = "user" if msg.role.value in ("user", "system") else "model"
            contents.append({"role": role, "parts": [{"text": msg.content}]})
        return contents

    @staticmethod
    def _extract_content(body: dict) -> str:
        candidates = body.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)

    @staticmethod
    def _extract_gemini_usage(body: dict) -> TokenUsage:
        meta = body.get("usageMetadata", {})
        return TokenUsage(
            prompt_tokens=meta.get("promptTokenCount", 0),
            completion_tokens=meta.get("candidatesTokenCount", 0),
            total_tokens=meta.get("totalTokenCount", 0),
        )

    @staticmethod
    def _map_finish_reason(body: dict) -> FinishReason:
        candidates = body.get("candidates", [])
        if not candidates:
            return FinishReason.UNKNOWN
        fr = candidates[0].get("finishReason", "")
        mapping = {
            "STOP": FinishReason.STOP,
            "MAX_TOKENS": FinishReason.LENGTH,
            "SAFETY": FinishReason.CONTENT_FILTER,
        }
        return mapping.get(fr, FinishReason.UNKNOWN)

    def _raise_for_status(self, resp: httpx.Response) -> None:
        try:
            body = resp.json()
        except Exception:
            body = resp.text

        category = self.classify_error(resp.status_code, body)
        error_msg = ""
        if isinstance(body, dict):
            error_obj = body.get("error", {})
            error_msg = error_obj.get("message", str(error_obj)) if isinstance(error_obj, dict) else str(error_obj)
        else:
            error_msg = str(body)[:500]

        full_msg = f"[{self.name}] Gemini HTTP {resp.status_code}: {error_msg}"
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
            err = body.get("error", {})
            if isinstance(err, dict):
                error_text = (err.get("message", "") + " " + str(err.get("status", ""))).lower()
            else:
                error_text = str(err).lower()
        elif isinstance(body, str):
            error_text = body.lower()

        if status_code == 429 or "resource_exhausted" in error_text or "quota" in error_text:
            return ProviderErrorCategory.RATE_LIMITED
        if "billing" in error_text or "payment" in error_text:
            return ProviderErrorCategory.BILLING_BLOCKED
        if status_code == 401 or status_code == 403:
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
