"""Replicate Predictions API adapter."""

from __future__ import annotations

import asyncio
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

logger = get_logger("providers.replicate")

# Default polling settings
_POLL_INTERVAL_S = 1.0
_MAX_POLL_ATTEMPTS = 120  # ~2 minutes at 1s intervals


class ReplicateClient(BaseProviderClient):
    """Adapter for the Replicate HTTP Predictions API.

    Replicate uses an asynchronous predictions model:
    1. ``POST /v1/predictions`` → returns a prediction object with status
    2. Poll ``GET /v1/predictions/{id}`` until status is "succeeded" / "failed"

    Config:
    - ``base_url``: defaults to ``https://api.replicate.com``
    - ``api_key``: Replicate API token (``r8_...``)
    - ``default_model``: The model *version* string, e.g.
      ``meta/meta-llama-3-70b-instruct`` or a full version hash.
    - ``metadata.poll_interval``: seconds between poll attempts (default 1.0)
    - ``metadata.max_poll_attempts``: max polls before timeout (default 120)

    Alternatively, Replicate offers an OpenAI-compatible proxy at
    ``https://openai-proxy.replicate.com/v1`` — you can use
    ``provider_type: openai_compatible`` with that URL if you prefer.
    This adapter uses the **native** predictions endpoint.
    """

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        base = config.base_url or "https://api.replicate.com"
        self._poll_interval = float(config.metadata.get("poll_interval", _POLL_INTERVAL_S))
        self._max_polls = int(config.metadata.get("max_poll_attempts", _MAX_POLL_ATTEMPTS))
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

        # Build a prompt string from messages
        prompt = "\n".join(f"{m.role.value}: {m.content}" for m in messages_raw)

        # Replicate input format
        input_payload: dict[str, Any] = {"prompt": prompt}
        if request.max_tokens is not None:
            input_payload["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            input_payload["temperature"] = request.temperature
        if request.top_p is not None:
            input_payload["top_p"] = request.top_p

        # System prompt — extract if present
        system_parts = [m.content for m in messages_raw if m.role.value == "system"]
        if system_parts:
            input_payload["system_prompt"] = "\n\n".join(system_parts)
            # Rebuild prompt without system messages
            non_system = [m for m in messages_raw if m.role.value != "system"]
            input_payload["prompt"] = "\n".join(f"{m.role.value}: {m.content}" for m in non_system)

        payload: dict[str, Any] = {
            "model": model,
            "input": input_payload,
        }

        t0 = time.perf_counter()
        try:
            resp = await self._client.post("/v1/predictions", json=payload)
        except httpx.TimeoutException as exc:
            raise TransientProviderError(
                f"Timeout creating Replicate prediction ({self.name})",
                provider_name=self.name,
                category=ProviderErrorCategory.TIMEOUT,
            ) from exc
        except httpx.ConnectError as exc:
            raise TransientProviderError(
                f"Connection error calling Replicate ({self.name}): {exc}",
                provider_name=self.name,
                category=ProviderErrorCategory.NETWORK_ERROR,
            ) from exc

        if resp.status_code >= 400:
            self._raise_for_status(resp)

        prediction = resp.json()
        prediction_id = prediction.get("id", "")
        status = prediction.get("status", "")

        # If already succeeded (unlikely for heavy models, but possible)
        if status == "succeeded":
            return self._build_response(prediction, request, model, t0)

        if status in ("failed", "canceled"):
            error_msg = prediction.get("error", "Prediction failed immediately")
            raise TransientProviderError(
                f"[{self.name}] Replicate prediction {prediction_id} {status}: {error_msg}",
                provider_name=self.name,
                category=ProviderErrorCategory.TRANSIENT_ERROR,
            )

        # Poll for completion
        result = await self._poll_prediction(prediction_id, t0)
        return self._build_response(result, request, model, t0)

    # ------------------------------------------------------------------ #
    # Polling
    # ------------------------------------------------------------------ #

    async def _poll_prediction(self, prediction_id: str, t0: float) -> dict:
        """Poll GET /v1/predictions/{id} until terminal status."""
        for _ in range(self._max_polls):
            await asyncio.sleep(self._poll_interval)
            try:
                resp = await self._client.get(f"/v1/predictions/{prediction_id}")
            except httpx.TimeoutException as exc:
                raise TransientProviderError(
                    f"Timeout polling Replicate prediction ({self.name})",
                    provider_name=self.name,
                    category=ProviderErrorCategory.TIMEOUT,
                ) from exc
            except httpx.ConnectError as exc:
                raise TransientProviderError(
                    f"Connection error polling Replicate ({self.name}): {exc}",
                    provider_name=self.name,
                    category=ProviderErrorCategory.NETWORK_ERROR,
                ) from exc

            if resp.status_code >= 400:
                self._raise_for_status(resp)

            prediction = resp.json()
            status = prediction.get("status", "")

            if status == "succeeded":
                return prediction
            if status in ("failed", "canceled"):
                error_msg = prediction.get("error", f"Prediction {status}")
                raise TransientProviderError(
                    f"[{self.name}] Replicate prediction {prediction_id} {status}: {error_msg}",
                    provider_name=self.name,
                    category=ProviderErrorCategory.TRANSIENT_ERROR,
                )
            # status is "starting" or "processing" — keep polling

        raise TransientProviderError(
            f"[{self.name}] Replicate prediction {prediction_id} timed out after {self._max_polls} polls",
            provider_name=self.name,
            category=ProviderErrorCategory.TIMEOUT,
        )

    # ------------------------------------------------------------------ #
    # Response building
    # ------------------------------------------------------------------ #

    def _build_response(self, prediction: dict, request: LLMRequest, model: str, t0: float) -> LLMResponse:
        latency_ms = (time.perf_counter() - t0) * 1000
        output = prediction.get("output", "")

        # Replicate output can be a string or a list of string tokens
        if isinstance(output, list):
            content = "".join(str(tok) for tok in output)
        else:
            content = str(output)

        # Replicate includes metrics in some models
        metrics = prediction.get("metrics", {})
        input_tokens = metrics.get("input_token_count", 0)
        output_tokens = metrics.get("output_token_count", 0)

        return LLMResponse(
            request_id=request.request_id,
            content=content,
            provider_name=self.name,
            model=model,
            usage=TokenUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            finish_reason=FinishReason.STOP,
            latency_ms=latency_ms,
            raw_response=prediction,
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
            error_msg = body.get("detail", body.get("error", str(body)))
        elif isinstance(body, str):
            error_msg = body[:500]
        else:
            error_msg = str(body)[:500]

        full_msg = f"[{self.name}] Replicate HTTP {resp.status_code}: {error_msg}"
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
            error_text = str(body.get("detail", body.get("error", ""))).lower()
        elif isinstance(body, str):
            error_text = body.lower()

        if status_code == 429 or "rate limit" in error_text or "too many requests" in error_text:
            return ProviderErrorCategory.RATE_LIMITED
        if "billing" in error_text or "payment" in error_text or "credit" in error_text:
            return ProviderErrorCategory.INSUFFICIENT_CREDIT
        if status_code == 401 or "unauthorized" in error_text or "unauthenticated" in error_text:
            return ProviderErrorCategory.AUTH_ERROR
        if status_code == 403:
            return ProviderErrorCategory.AUTH_ERROR
        if status_code == 404 or "not found" in error_text or "does not exist" in error_text:
            return ProviderErrorCategory.MODEL_UNAVAILABLE
        if status_code == 422 or "invalid" in error_text or "validation" in error_text:
            return ProviderErrorCategory.INVALID_REQUEST
        if status_code and 500 <= status_code < 600:
            return ProviderErrorCategory.TRANSIENT_ERROR
        return ProviderErrorCategory.UNKNOWN_ERROR

    async def close(self) -> None:
        await self._client.aclose()
