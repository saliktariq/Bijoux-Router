"""Integration tests for the BijouxRouter engine — routing, failover, quota, cooldown."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from bijoux_router.config.schema import (
    CooldownPolicyConfig,
    GatewayConfig,
    PeriodType,
    ProviderConfig,
    QuotaConfig,
    ResetMode,
    RetryPolicyConfig,
    SelectionStrategyConfig,
)
from bijoux_router.exceptions.errors import (
    AllProvidersExhaustedError,
    NoViableProviderError,
    ProviderErrorCategory,
    RequestValidationError,
)
from bijoux_router.models.request_response import LLMRequest
from bijoux_router.providers.mock import MockProviderClient
from bijoux_router.router.engine import BijouxRouter
from bijoux_router.storage.sqlite_backend import SQLiteStorage


def _mock_config(
    name: str,
    priority: int = 0,
    token_limit: int = 100_000,
    mock_error: str | None = None,
    mock_content: str | None = None,
    failover_enabled: bool = True,
    continue_on_auth_error: bool = False,
    continue_on_invalid_request: bool = False,
    max_retries: int = 0,
    cooldown_seconds: float = 5.0,
    failure_threshold: int = 3,
    quota_exhaustion_cooldown: float = 10.0,
    enabled: bool = True,
) -> ProviderConfig:
    metadata: dict[str, Any] = {}
    if mock_error:
        metadata["mock_error"] = mock_error
    if mock_content:
        metadata["mock_content"] = mock_content
    return ProviderConfig(
        name=name,
        enabled=enabled,
        provider_type="mock",
        base_url="http://mock",
        api_key="mock-key",
        default_model="mock-model",
        priority=priority,
        timeout_seconds=5,
        retry_policy=RetryPolicyConfig(max_retries=max_retries, backoff_base=0.01, backoff_max=0.05),
        cooldown_policy=CooldownPolicyConfig(
            cooldown_seconds=cooldown_seconds,
            failure_threshold=failure_threshold,
            quota_exhaustion_cooldown_seconds=quota_exhaustion_cooldown,
        ),
        quota=QuotaConfig(token_limit=token_limit, period_type=PeriodType.DAY, period_value=1, reset_mode=ResetMode.FIXED),
        metadata=metadata,
        failover_enabled=failover_enabled,
        continue_on_auth_error=continue_on_auth_error,
        continue_on_invalid_request=continue_on_invalid_request,
    )


def _build_router(providers: list[ProviderConfig], storage: SQLiteStorage, **kwargs: Any) -> BijouxRouter:
    config = GatewayConfig(
        providers=providers,
        storage_path=":memory:",
        max_failover_attempts=kwargs.get("max_failover_attempts", 5),
        selection_strategy=kwargs.get(
            "selection_strategy",
            SelectionStrategyConfig(fairness_cursor=False),
        ),
    )
    return BijouxRouter(config, storage=storage)


class TestBasicRouting:
    @pytest.mark.asyncio
    async def test_routes_to_highest_priority(self, storage: SQLiteStorage) -> None:
        router = _build_router(
            [
                _mock_config("b", priority=2, mock_content="from B"),
                _mock_config("a", priority=1, mock_content="from A"),
            ],
            storage,
        )
        resp = await router.process(LLMRequest(prompt="hello"))
        assert resp.provider_name == "a"
        assert resp.content == "from A"
        router.close()

    @pytest.mark.asyncio
    async def test_returns_normalized_response(self, storage: SQLiteStorage) -> None:
        router = _build_router([_mock_config("p1", priority=1)], storage)
        resp = await router.process(LLMRequest(prompt="test"))
        assert resp.request_id
        assert resp.provider_name == "p1"
        assert resp.model == "mock-model"
        assert resp.usage.total_tokens > 0
        assert resp.finish_reason.value == "stop"
        assert resp.latency_ms >= 0
        assert resp.failover_attempts == []
        router.close()

    @pytest.mark.asyncio
    async def test_rejects_empty_request(self, storage: SQLiteStorage) -> None:
        router = _build_router([_mock_config("p1")], storage)
        with pytest.raises(RequestValidationError):
            await router.process(LLMRequest())
        router.close()


class TestFailover:
    @pytest.mark.asyncio
    async def test_failover_on_quota_exhausted(self, storage: SQLiteStorage) -> None:
        router = _build_router(
            [
                _mock_config("a", priority=1, mock_error="QUOTA_EXHAUSTED"),
                _mock_config("b", priority=2, mock_content="from B"),
            ],
            storage,
        )
        resp = await router.process(LLMRequest(prompt="hello"))
        assert resp.provider_name == "b"
        assert len(resp.failover_attempts) == 1
        assert resp.failover_attempts[0].provider_name == "a"
        assert resp.failover_attempts[0].error_category == "QUOTA_EXHAUSTED"
        router.close()

    @pytest.mark.asyncio
    async def test_failover_on_rate_limited(self, storage: SQLiteStorage) -> None:
        router = _build_router(
            [
                _mock_config("a", priority=1, mock_error="RATE_LIMITED"),
                _mock_config("b", priority=2, mock_content="ok"),
            ],
            storage,
        )
        resp = await router.process(LLMRequest(prompt="hello"))
        assert resp.provider_name == "b"
        router.close()

    @pytest.mark.asyncio
    async def test_failover_on_insufficient_credit(self, storage: SQLiteStorage) -> None:
        router = _build_router(
            [
                _mock_config("a", priority=1, mock_error="INSUFFICIENT_CREDIT"),
                _mock_config("b", priority=2, mock_content="ok"),
            ],
            storage,
        )
        resp = await router.process(LLMRequest(prompt="hello"))
        assert resp.provider_name == "b"
        router.close()

    @pytest.mark.asyncio
    async def test_failover_on_transient_error(self, storage: SQLiteStorage) -> None:
        router = _build_router(
            [
                _mock_config("a", priority=1, mock_error="TRANSIENT_ERROR"),
                _mock_config("b", priority=2, mock_content="ok"),
            ],
            storage,
        )
        resp = await router.process(LLMRequest(prompt="hi"))
        assert resp.provider_name == "b"
        router.close()

    @pytest.mark.asyncio
    async def test_all_providers_exhausted(self, storage: SQLiteStorage) -> None:
        router = _build_router(
            [
                _mock_config("a", priority=1, mock_error="QUOTA_EXHAUSTED"),
                _mock_config("b", priority=2, mock_error="QUOTA_EXHAUSTED"),
            ],
            storage,
        )
        with pytest.raises(AllProvidersExhaustedError) as exc_info:
            await router.process(LLMRequest(prompt="hello"))
        assert len(exc_info.value.attempts) == 2
        router.close()

    @pytest.mark.asyncio
    async def test_failover_chain_three_providers(self, storage: SQLiteStorage) -> None:
        router = _build_router(
            [
                _mock_config("a", priority=1, mock_error="RATE_LIMITED"),
                _mock_config("b", priority=2, mock_error="INSUFFICIENT_CREDIT"),
                _mock_config("c", priority=3, mock_content="from C"),
            ],
            storage,
        )
        resp = await router.process(LLMRequest(prompt="hello"))
        assert resp.provider_name == "c"
        assert len(resp.failover_attempts) == 2
        router.close()


class TestAuthErrorPolicy:
    @pytest.mark.asyncio
    async def test_auth_error_does_not_failover_by_default(self, storage: SQLiteStorage) -> None:
        router = _build_router(
            [
                _mock_config("a", priority=1, mock_error="AUTH_ERROR"),
                _mock_config("b", priority=2, mock_content="ok"),
            ],
            storage,
        )
        with pytest.raises(AllProvidersExhaustedError):
            await router.process(LLMRequest(prompt="hello"))
        router.close()

    @pytest.mark.asyncio
    async def test_auth_error_failover_when_configured(self, storage: SQLiteStorage) -> None:
        router = _build_router(
            [
                _mock_config("a", priority=1, mock_error="AUTH_ERROR", continue_on_auth_error=True),
                _mock_config("b", priority=2, mock_content="ok"),
            ],
            storage,
        )
        resp = await router.process(LLMRequest(prompt="hello"))
        assert resp.provider_name == "b"
        router.close()


class TestInvalidRequestPolicy:
    @pytest.mark.asyncio
    async def test_invalid_request_does_not_failover_by_default(self, storage: SQLiteStorage) -> None:
        router = _build_router(
            [
                _mock_config("a", priority=1, mock_error="INVALID_REQUEST"),
                _mock_config("b", priority=2, mock_content="ok"),
            ],
            storage,
        )
        with pytest.raises(AllProvidersExhaustedError):
            await router.process(LLMRequest(prompt="hello"))
        router.close()

    @pytest.mark.asyncio
    async def test_invalid_request_failover_when_configured(self, storage: SQLiteStorage) -> None:
        router = _build_router(
            [
                _mock_config("a", priority=1, mock_error="INVALID_REQUEST", continue_on_invalid_request=True),
                _mock_config("b", priority=2, mock_content="ok"),
            ],
            storage,
        )
        resp = await router.process(LLMRequest(prompt="hello"))
        assert resp.provider_name == "b"
        router.close()


class TestTransientRetry:
    @pytest.mark.asyncio
    async def test_retries_transient_then_failsover(self, storage: SQLiteStorage) -> None:
        """Provider A gets retries, then fails over to B."""
        router = _build_router(
            [
                _mock_config("a", priority=1, mock_error="TRANSIENT_ERROR", max_retries=2),
                _mock_config("b", priority=2, mock_content="ok"),
            ],
            storage,
        )
        resp = await router.process(LLMRequest(prompt="hello"))
        assert resp.provider_name == "b"
        # Provider A's mock client should have been called 3 times (1 + 2 retries)
        client_a = router._clients["a"]
        assert isinstance(client_a, MockProviderClient)
        assert client_a.call_count == 3
        router.close()


class TestQuotaTracking:
    @pytest.mark.asyncio
    async def test_usage_recorded_after_success(self, storage: SQLiteStorage) -> None:
        router = _build_router([_mock_config("p1", priority=1, token_limit=100000)], storage)
        await router.process(LLMRequest(prompt="hello"))
        statuses = router.get_quota_status()
        assert len(statuses) == 1
        assert statuses[0]["used_tokens"] > 0
        assert statuses[0]["reserved_tokens"] == 0  # reservation reconciled
        router.close()

    @pytest.mark.asyncio
    async def test_reservation_released_on_failure(self, storage: SQLiteStorage) -> None:
        router = _build_router(
            [
                _mock_config("a", priority=1, mock_error="QUOTA_EXHAUSTED"),
                _mock_config("b", priority=2, mock_content="ok"),
            ],
            storage,
        )
        await router.process(LLMRequest(prompt="hello"))
        # Provider A's reservation should have been released
        statuses = router.get_quota_status()
        a_status = next(s for s in statuses if s["provider_name"] == "a")
        assert a_status["reserved_tokens"] == 0
        assert a_status["used_tokens"] == 0  # never succeeded
        router.close()

    @pytest.mark.asyncio
    async def test_skips_quota_exhausted_provider(self, storage: SQLiteStorage) -> None:
        """Provider with near-zero remaining quota gets skipped at selection time."""
        router = _build_router(
            [
                _mock_config("a", priority=1, token_limit=10, mock_content="from A"),
                _mock_config("b", priority=2, token_limit=100000, mock_content="from B"),
            ],
            storage,
        )
        # Exhaust provider A's quota
        quota_tracker = router._quota
        config_a = router._provider_configs["a"]
        wk = quota_tracker.get_window_key(config_a)
        storage.record_usage("a", wk, 5, 5, 10, "r-pre", "m", time.time())

        resp = await router.process(LLMRequest(prompt="hello"))
        assert resp.provider_name == "b"
        router.close()

    @pytest.mark.asyncio
    async def test_no_viable_provider_when_all_exhausted(self, storage: SQLiteStorage) -> None:
        router = _build_router(
            [_mock_config("a", priority=1, token_limit=10)],
            storage,
        )
        config_a = router._provider_configs["a"]
        wk = router._quota.get_window_key(config_a)
        storage.record_usage("a", wk, 5, 5, 10, "r-pre", "m", time.time())

        with pytest.raises(NoViableProviderError):
            await router.process(LLMRequest(prompt="hello"))
        router.close()


class TestCooldown:
    @pytest.mark.asyncio
    async def test_quota_exhaustion_triggers_cooldown(self, storage: SQLiteStorage) -> None:
        router = _build_router(
            [
                _mock_config("a", priority=1, mock_error="QUOTA_EXHAUSTED", quota_exhaustion_cooldown=10),
                _mock_config("b", priority=2, mock_content="ok"),
            ],
            storage,
        )
        await router.process(LLMRequest(prompt="hello"))
        # Provider A should now be in cooldown
        cooldown = storage.get_cooldown("a")
        assert cooldown is not None
        assert cooldown > time.time()
        router.close()

    @pytest.mark.asyncio
    async def test_repeated_failures_trigger_cooldown(self, storage: SQLiteStorage) -> None:
        """Non-quota failures accumulate and trigger cooldown at threshold."""
        router = _build_router(
            [
                _mock_config("a", priority=1, mock_error="TRANSIENT_ERROR", failure_threshold=2, cooldown_seconds=10),
                _mock_config("b", priority=2, mock_content="ok"),
            ],
            storage,
        )
        # First request: provider A fails, failover to B
        await router.process(LLMRequest(prompt="r1"))
        assert storage.get_cooldown("a") is None  # Below threshold

        # Second request: another failure pushes A past threshold
        await router.process(LLMRequest(prompt="r2"))
        cooldown = storage.get_cooldown("a")
        assert cooldown is not None
        router.close()

    @pytest.mark.asyncio
    async def test_provider_in_cooldown_is_skipped(self, storage: SQLiteStorage) -> None:
        storage.set_cooldown("a", time.time() + 3600, "test")
        router = _build_router(
            [
                _mock_config("a", priority=1, mock_content="from A"),
                _mock_config("b", priority=2, mock_content="from B"),
            ],
            storage,
        )
        resp = await router.process(LLMRequest(prompt="hello"))
        assert resp.provider_name == "b"
        router.close()


class TestPersistenceAcrossRestart:
    @pytest.mark.asyncio
    async def test_usage_persists_across_router_instances(self, tmp_db: str) -> None:
        """Verify that usage data survives closing and re-creating the router."""
        s1 = SQLiteStorage(tmp_db)
        s1.initialize()
        r1 = _build_router(
            [_mock_config("p1", priority=1, token_limit=100000)],
            s1,
        )
        await r1.process(LLMRequest(prompt="first"))
        statuses1 = r1.get_quota_status()
        used1 = statuses1[0]["used_tokens"]
        r1.close()
        s1.close()

        # Reopen
        s2 = SQLiteStorage(tmp_db)
        s2.initialize()
        r2 = _build_router(
            [_mock_config("p1", priority=1, token_limit=100000)],
            s2,
        )
        statuses2 = r2.get_quota_status()
        assert statuses2[0]["used_tokens"] == used1
        r2.close()
        s2.close()


class TestProviderStatus:
    @pytest.mark.asyncio
    async def test_get_provider_status(self, storage: SQLiteStorage) -> None:
        router = _build_router(
            [
                _mock_config("a", priority=1),
                _mock_config("b", priority=2),
            ],
            storage,
        )
        statuses = router.get_provider_status()
        assert len(statuses) == 2
        assert statuses[0]["name"] == "a"
        assert statuses[0]["enabled"] is True
        assert "remaining_tokens" in statuses[0]
        router.close()

    @pytest.mark.asyncio
    async def test_reset_provider_usage(self, storage: SQLiteStorage) -> None:
        router = _build_router([_mock_config("p1", priority=1)], storage)
        await router.process(LLMRequest(prompt="hello"))
        assert router.get_quota_status()[0]["used_tokens"] > 0
        router.reset_provider_usage("p1")
        assert router.get_quota_status()[0]["used_tokens"] == 0
        router.close()


class TestYAMLLoading:
    @pytest.mark.asyncio
    async def test_from_yaml(self, sample_yaml_path: Path) -> None:
        router = BijouxRouter.from_yaml(sample_yaml_path)
        resp = await router.process(LLMRequest(prompt="hello"))
        assert resp.provider_name == "test-mock"
        router.close()


class TestAdapterErrorClassification:
    def test_openai_compatible_classify_429(self) -> None:
        from bijoux_router.providers.openai_compatible import OpenAICompatibleClient
        config = _mock_config("t", priority=1)
        client = OpenAICompatibleClient(config)
        assert client.classify_error(429, {"error": {"message": "rate limit"}}) == ProviderErrorCategory.RATE_LIMITED

    def test_openai_compatible_classify_401(self) -> None:
        from bijoux_router.providers.openai_compatible import OpenAICompatibleClient
        config = _mock_config("t", priority=1)
        client = OpenAICompatibleClient(config)
        assert client.classify_error(401, {}) == ProviderErrorCategory.AUTH_ERROR

    def test_openai_compatible_classify_500(self) -> None:
        from bijoux_router.providers.openai_compatible import OpenAICompatibleClient
        config = _mock_config("t", priority=1)
        client = OpenAICompatibleClient(config)
        assert client.classify_error(500, {}) == ProviderErrorCategory.TRANSIENT_ERROR

    def test_openai_compatible_classify_credit(self) -> None:
        from bijoux_router.providers.openai_compatible import OpenAICompatibleClient
        config = _mock_config("t", priority=1)
        client = OpenAICompatibleClient(config)
        cat = client.classify_error(402, {"error": {"message": "insufficient credit balance"}})
        assert cat == ProviderErrorCategory.INSUFFICIENT_CREDIT

    def test_openai_compatible_classify_model_not_found(self) -> None:
        from bijoux_router.providers.openai_compatible import OpenAICompatibleClient
        config = _mock_config("t", priority=1)
        client = OpenAICompatibleClient(config)
        assert client.classify_error(404, {}) == ProviderErrorCategory.MODEL_UNAVAILABLE

    def test_gemini_classify_429(self) -> None:
        from bijoux_router.providers.gemini import GeminiClient
        config = _mock_config("t", priority=1)
        client = GeminiClient(config)
        assert client.classify_error(429, {"error": {"message": "RESOURCE_EXHAUSTED"}}) == ProviderErrorCategory.RATE_LIMITED

    def test_gemini_classify_auth(self) -> None:
        from bijoux_router.providers.gemini import GeminiClient
        config = _mock_config("t", priority=1)
        client = GeminiClient(config)
        assert client.classify_error(403, {}) == ProviderErrorCategory.AUTH_ERROR


class TestTokenEstimation:
    def test_estimate_with_prompt(self) -> None:
        from bijoux_router.utils.tokens import estimate_tokens
        req = LLMRequest(prompt="Hello world", max_tokens=100)
        est = estimate_tokens(req)
        assert est.estimated_prompt_tokens > 0
        assert est.estimated_completion_tokens == 100
        assert est.estimated_total == est.estimated_prompt_tokens + est.estimated_completion_tokens

    def test_estimate_default_completion(self) -> None:
        from bijoux_router.utils.tokens import estimate_tokens
        req = LLMRequest(prompt="Hello")
        est = estimate_tokens(req)
        assert est.estimated_completion_tokens == 256  # default

    def test_estimate_empty(self) -> None:
        from bijoux_router.utils.tokens import estimate_tokens
        req = LLMRequest(prompt="")
        est = estimate_tokens(req)
        assert est.estimated_prompt_tokens == 0
