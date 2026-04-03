"""Core router / orchestration engine for Bijoux LLM Gateway."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from bijoux_router.config.loader import load_config
from bijoux_router.config.schema import GatewayConfig, ProviderConfig
from bijoux_router.exceptions.errors import (
    AllProvidersExhaustedError,
    BijouxError,
    NoViableProviderError,
    ProviderError,
    ProviderErrorCategory,
    RequestValidationError,
)
from bijoux_router.models.request_response import (
    FailoverAttempt,
    LLMRequest,
    LLMResponse,
    TokenUsage,
)
from bijoux_router.providers.base import BaseProviderClient
from bijoux_router.providers.factory import create_provider
from bijoux_router.quota.tracker import QuotaTracker
from bijoux_router.router.selection import ProviderSelector
from bijoux_router.storage.base import StorageBackend
from bijoux_router.storage.sqlite_backend import SQLiteStorage
from bijoux_router.utils.logging import get_logger, redact_dict
from bijoux_router.utils.tokens import estimate_tokens

logger = get_logger("router.engine")


class BijouxRouter:
    """Main entry point — quota-aware LLM request router with transparent failover.

    Usage:
        router = BijouxRouter.from_yaml("config/providers.yaml")
        response = await router.process(LLMRequest(prompt="Hello"))
        router.close()
    """

    def __init__(
        self,
        config: GatewayConfig,
        storage: StorageBackend | None = None,
    ) -> None:
        self._config = config
        # Storage
        self._storage = storage or SQLiteStorage(config.storage_path)
        self._storage.initialize()
        # Quota tracker
        self._quota = QuotaTracker(self._storage)
        # Provider clients
        self._clients: dict[str, BaseProviderClient] = {}
        self._provider_configs: dict[str, ProviderConfig] = {}
        for pc in config.providers:
            if pc.enabled:
                client = create_provider(pc)
                self._clients[pc.name] = client
                self._provider_configs[pc.name] = pc
        # Selector
        self._selector = ProviderSelector(
            providers=list(self._provider_configs.values()),
            quota_tracker=self._quota,
            storage=self._storage,
            strategy_config=config.selection_strategy,
        )
        logger.info(
            "BijouxRouter initialized with %d enabled providers: %s",
            len(self._clients),
            ", ".join(self._clients.keys()),
        )

    @classmethod
    def from_yaml(cls, path: str | Path, storage: StorageBackend | None = None) -> BijouxRouter:
        """Create a router from a YAML configuration file."""
        config = load_config(path)
        return cls(config, storage=storage)

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    async def process(self, request: LLMRequest) -> LLMResponse:
        """Route an LLM request to the best available provider.

        This is the central contract:
        - Select best provider by quota + policy
        - Reserve estimated tokens
        - Dispatch request
        - Reconcile actual usage
        - On quota/credit failure → failover to next provider
        - Return normalized response
        """
        self._validate_request(request)
        logger.info("Processing request %s (model=%s)", request.request_id, request.model)

        # Estimate tokens for quota reservation
        token_est = estimate_tokens(request)
        estimated_total = token_est.estimated_total

        # Select ordered providers
        ordered = self._selector.select_ordered(
            estimated_tokens=estimated_total,
            request_model=request.model,
        )
        if not ordered:
            raise NoViableProviderError(
                "No viable provider found for request",
                details={"request_id": request.request_id, "model": request.model},
            )

        failover_attempts: list[FailoverAttempt] = []
        last_error: ProviderError | None = None

        max_attempts = min(len(ordered), self._config.max_failover_attempts)

        for attempt_idx, provider_config in enumerate(ordered[:max_attempts]):
            provider_name = provider_config.name
            client = self._clients[provider_name]

            # Reserve tokens
            reservation_id = self._quota.create_reservation(provider_config, estimated_total)

            try:
                response = await self._dispatch_with_retry(client, provider_config, request)
            except ProviderError as exc:
                # Release reservation on failure
                self._quota.release_reservation(reservation_id)
                last_error = exc
                latency = 0.0

                logger.warning(
                    "Provider %s failed for request %s: [%s] %s",
                    provider_name, request.request_id, exc.category.value, exc,
                )

                failover_attempts.append(FailoverAttempt(
                    provider_name=provider_name,
                    error_category=exc.category.value,
                    error_message=str(exc),
                    latency_ms=latency,
                ))

                # Handle error categories
                self._handle_provider_failure(provider_config, exc)

                # Decide whether to failover
                if not self._should_failover(provider_config, exc):
                    logger.info("Not failing over for %s error on %s", exc.category.value, provider_name)
                    raise AllProvidersExhaustedError(
                        f"Provider {provider_name} failed with non-failover error: {exc.category.value}",
                        attempts=[a.model_dump() for a in failover_attempts],
                    ) from exc

                logger.info(
                    "Failing over from %s (attempt %d/%d): %s",
                    provider_name, attempt_idx + 1, max_attempts, exc.category.value,
                )
                continue

            # SUCCESS — reconcile reservation with actual usage
            actual_usage = response.usage
            if actual_usage.total_tokens == 0:
                # Provider didn't report usage; use estimate
                actual_usage = TokenUsage.from_counts(
                    prompt=token_est.estimated_prompt_tokens,
                    completion=token_est.estimated_completion_tokens,
                )
                response.usage = actual_usage

            self._quota.reconcile(
                config=provider_config,
                reservation_id=reservation_id,
                actual_prompt=actual_usage.prompt_tokens,
                actual_completion=actual_usage.completion_tokens,
                actual_total=actual_usage.total_tokens,
                request_id=request.request_id,
                model=response.model,
            )

            # Update fairness cursor and reset failures
            self._storage.set_last_used_provider(provider_name)
            self._storage.reset_failures(provider_name)

            # Attach failover metadata
            response.failover_attempts = failover_attempts

            logger.info(
                "Request %s completed via %s (model=%s, tokens=%d, failovers=%d)",
                request.request_id, provider_name, response.model,
                actual_usage.total_tokens, len(failover_attempts),
            )
            return response

        # All attempts exhausted
        raise AllProvidersExhaustedError(
            f"All {max_attempts} provider attempts exhausted for request {request.request_id}",
            attempts=[a.model_dump() for a in failover_attempts],
        )

    # ------------------------------------------------------------------
    # Dispatch with per-provider retry
    # ------------------------------------------------------------------

    async def _dispatch_with_retry(
        self,
        client: BaseProviderClient,
        config: ProviderConfig,
        request: LLMRequest,
    ) -> LLMResponse:
        """Dispatch request to a single provider with retry on transient errors."""
        max_retries = config.retry_policy.max_retries
        backoff_base = config.retry_policy.backoff_base
        backoff_max = config.retry_policy.backoff_max

        last_exc: ProviderError | None = None
        for attempt in range(1 + max_retries):
            try:
                return await client.send_request(request)
            except ProviderError as exc:
                last_exc = exc
                if not exc.category.is_retriable_transient:
                    raise
                if attempt >= max_retries:
                    raise
                if not config.retry_policy.retry_on_transient:
                    raise
                delay = min(backoff_base * (2 ** attempt), backoff_max)
                logger.debug(
                    "Transient error from %s (attempt %d/%d), retrying in %.1fs: %s",
                    config.name, attempt + 1, 1 + max_retries, delay, exc,
                )
                await asyncio.sleep(delay)

        # Should not reach here, but satisfy type checker
        assert last_exc is not None
        raise last_exc

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------

    def _handle_provider_failure(self, config: ProviderConfig, exc: ProviderError) -> None:
        """Update provider health state after a failure."""
        failure_count = self._storage.increment_failure(config.name)

        if exc.category.is_quota_related:
            # Put into quota-exhaustion cooldown
            cooldown = config.cooldown_policy.quota_exhaustion_cooldown_seconds
            until = time.time() + cooldown
            self._storage.set_cooldown(config.name, until, reason=exc.category.value)
            logger.info("Provider %s entering quota cooldown for %.0fs", config.name, cooldown)

        elif failure_count >= config.cooldown_policy.failure_threshold:
            # Repeated failures → cooldown
            cooldown = config.cooldown_policy.cooldown_seconds
            until = time.time() + cooldown
            self._storage.set_cooldown(config.name, until, reason="repeated_failures")
            self._storage.reset_failures(config.name)
            logger.info(
                "Provider %s entering cooldown for %.0fs after %d failures",
                config.name, cooldown, failure_count,
            )

    def _should_failover(self, config: ProviderConfig, exc: ProviderError) -> bool:
        """Determine whether to try the next provider after this error."""
        if not config.failover_enabled:
            return False

        if exc.category.should_failover:
            return True

        if exc.category == ProviderErrorCategory.AUTH_ERROR:
            return config.continue_on_auth_error

        if exc.category == ProviderErrorCategory.INVALID_REQUEST:
            return config.continue_on_invalid_request

        return False

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_request(request: LLMRequest) -> None:
        if not request.prompt and not request.messages:
            raise RequestValidationError("Request must have either 'prompt' or 'messages'")

    # ------------------------------------------------------------------
    # Status / admin
    # ------------------------------------------------------------------

    def get_provider_status(self) -> list[dict[str, Any]]:
        """Return status for all configured providers."""
        statuses: list[dict[str, Any]] = []
        now = time.time()
        for name, config in self._provider_configs.items():
            cooldown_until = self._storage.get_cooldown(name)
            failures = self._storage.get_failure_count(name)
            quota_status = self._quota.get_quota_status(config)
            statuses.append({
                "name": name,
                "enabled": config.enabled,
                "provider_type": config.provider_type,
                "priority": config.priority,
                "in_cooldown": cooldown_until is not None and cooldown_until > now,
                "cooldown_until": cooldown_until,
                "failure_count": failures,
                **quota_status,
            })
        return statuses

    def get_quota_status(self) -> list[dict[str, Any]]:
        """Return quota status for all providers."""
        return [
            self._quota.get_quota_status(config)
            for config in self._provider_configs.values()
        ]

    def reload_config(self, path: str | Path) -> None:
        """Reload configuration from YAML. Re-creates provider clients."""
        logger.info("Reloading configuration from %s", path)
        new_config = load_config(path)

        # Close old clients
        for client in self._clients.values():
            # fire-and-forget close in sync context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(client.close())
                else:
                    loop.run_until_complete(client.close())
            except RuntimeError:
                pass

        self._config = new_config
        self._clients.clear()
        self._provider_configs.clear()
        for pc in new_config.providers:
            if pc.enabled:
                self._clients[pc.name] = create_provider(pc)
                self._provider_configs[pc.name] = pc

        self._selector = ProviderSelector(
            providers=list(self._provider_configs.values()),
            quota_tracker=self._quota,
            storage=self._storage,
            strategy_config=new_config.selection_strategy,
        )
        logger.info("Configuration reloaded: %d providers", len(self._clients))

    def reset_provider_usage(self, provider_name: str) -> None:
        """Reset all usage, reservations, cooldowns, failures for a provider."""
        self._storage.reset_provider_usage(provider_name)

    def close(self) -> None:
        """Release all resources."""
        for client in self._clients.values():
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(client.close())
                else:
                    loop.run_until_complete(client.close())
            except RuntimeError:
                pass
        self._storage.close()
        logger.info("BijouxRouter closed")
