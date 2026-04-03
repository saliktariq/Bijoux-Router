"""Shared fixtures for Bijoux Router tests."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Generator

import pytest
import yaml

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
from bijoux_router.router.engine import BijouxRouter
from bijoux_router.storage.sqlite_backend import SQLiteStorage


def _make_mock_provider(
    name: str,
    priority: int = 0,
    token_limit: int = 100_000,
    enabled: bool = True,
    mock_error: str | None = None,
    mock_content: str | None = None,
    failover_enabled: bool = True,
    continue_on_auth_error: bool = False,
    continue_on_invalid_request: bool = False,
    cooldown_seconds: float = 5.0,
    failure_threshold: int = 3,
    quota_exhaustion_cooldown_seconds: float = 10.0,
    max_retries: int = 0,
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
        retry_policy=RetryPolicyConfig(max_retries=max_retries, backoff_base=0.01, backoff_max=0.1),
        cooldown_policy=CooldownPolicyConfig(
            cooldown_seconds=cooldown_seconds,
            failure_threshold=failure_threshold,
            quota_exhaustion_cooldown_seconds=quota_exhaustion_cooldown_seconds,
        ),
        quota=QuotaConfig(
            token_limit=token_limit,
            period_type=PeriodType.DAY,
            period_value=1,
            reset_mode=ResetMode.FIXED,
        ),
        metadata=metadata,
        failover_enabled=failover_enabled,
        continue_on_auth_error=continue_on_auth_error,
        continue_on_invalid_request=continue_on_invalid_request,
    )


@pytest.fixture
def tmp_db(tmp_path: Path) -> str:
    return str(tmp_path / "test_bijoux.db")


@pytest.fixture
def storage(tmp_db: str) -> Generator[SQLiteStorage, None, None]:
    s = SQLiteStorage(tmp_db)
    s.initialize()
    yield s
    s.close()


@pytest.fixture
def two_provider_config(tmp_db: str) -> GatewayConfig:
    """Config with two mock providers: primary (priority 1) and secondary (priority 2)."""
    return GatewayConfig(
        providers=[
            _make_mock_provider("provider-a", priority=1, token_limit=50_000),
            _make_mock_provider("provider-b", priority=2, token_limit=100_000),
        ],
        storage_path=tmp_db,
        max_failover_attempts=5,
    )


@pytest.fixture
def three_provider_config(tmp_db: str) -> GatewayConfig:
    return GatewayConfig(
        providers=[
            _make_mock_provider("provider-a", priority=1),
            _make_mock_provider("provider-b", priority=2),
            _make_mock_provider("provider-c", priority=3),
        ],
        storage_path=tmp_db,
    )


@pytest.fixture
def router_two(two_provider_config: GatewayConfig, storage: SQLiteStorage) -> Generator[BijouxRouter, None, None]:
    r = BijouxRouter(two_provider_config, storage=storage)
    yield r
    r.close()


@pytest.fixture
def router_three(three_provider_config: GatewayConfig, storage: SQLiteStorage) -> Generator[BijouxRouter, None, None]:
    r = BijouxRouter(three_provider_config, storage=storage)
    yield r
    r.close()


@pytest.fixture
def sample_yaml_path(tmp_path: Path) -> Path:
    """Write a minimal valid YAML config to a temp file."""
    config = {
        "providers": [
            {
                "name": "test-mock",
                "enabled": True,
                "provider_type": "mock",
                "base_url": "http://mock",
                "api_key": "test-key",
                "default_model": "mock-model",
                "priority": 1,
                "quota": {
                    "token_limit": 100000,
                    "period_type": "day",
                    "period_value": 1,
                    "reset_mode": "fixed",
                },
            }
        ],
        "storage_path": str(tmp_path / "yaml_test.db"),
    }
    p = tmp_path / "test_providers.yaml"
    p.write_text(yaml.dump(config), encoding="utf-8")
    return p
