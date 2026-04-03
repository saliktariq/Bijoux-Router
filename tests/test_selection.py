"""Tests for provider selection strategy."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from bijoux_router.config.schema import (
    PeriodType,
    ProviderConfig,
    QuotaConfig,
    ResetMode,
    SelectionStrategyConfig,
    SelectionStrategyType,
)
from bijoux_router.quota.tracker import QuotaTracker
from bijoux_router.router.selection import ProviderSelector
from bijoux_router.storage.sqlite_backend import SQLiteStorage


def _prov(name: str, priority: int = 0, token_limit: int = 100000, enabled: bool = True) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        provider_type="mock",
        priority=priority,
        enabled=enabled,
        quota=QuotaConfig(token_limit=token_limit, period_type=PeriodType.DAY, period_value=1, reset_mode=ResetMode.FIXED),
    )


class TestProviderSelector:
    def test_selects_by_priority(self, storage: SQLiteStorage) -> None:
        providers = [_prov("c", priority=3), _prov("a", priority=1), _prov("b", priority=2)]
        qt = QuotaTracker(storage)
        sel = ProviderSelector(providers, qt, storage, SelectionStrategyConfig(fairness_cursor=False))
        ordered = sel.select_ordered(estimated_tokens=100)
        assert [p.name for p in ordered] == ["a", "b", "c"]

    def test_skips_disabled(self, storage: SQLiteStorage) -> None:
        providers = [_prov("a", priority=1), _prov("b", priority=2, enabled=False)]
        qt = QuotaTracker(storage)
        sel = ProviderSelector(providers, qt, storage, SelectionStrategyConfig())
        ordered = sel.select_ordered(estimated_tokens=100)
        assert [p.name for p in ordered] == ["a"]

    def test_skips_exhausted_quota(self, storage: SQLiteStorage) -> None:
        providers = [_prov("a", priority=1, token_limit=50), _prov("b", priority=2, token_limit=100000)]
        qt = QuotaTracker(storage)
        # Exhaust provider a
        wk = qt.get_window_key(providers[0])
        storage.record_usage("a", wk, 20, 30, 50, "r1", "m", time.time())
        sel = ProviderSelector(providers, qt, storage, SelectionStrategyConfig())
        ordered = sel.select_ordered(estimated_tokens=100)
        assert [p.name for p in ordered] == ["b"]

    def test_skips_cooldown(self, storage: SQLiteStorage) -> None:
        providers = [_prov("a", priority=1), _prov("b", priority=2)]
        storage.set_cooldown("a", time.time() + 3600, "test")
        qt = QuotaTracker(storage)
        sel = ProviderSelector(providers, qt, storage, SelectionStrategyConfig())
        ordered = sel.select_ordered(estimated_tokens=100)
        assert [p.name for p in ordered] == ["b"]

    def test_fairness_cursor_rotates(self, storage: SQLiteStorage) -> None:
        providers = [_prov("a", priority=1), _prov("b", priority=1), _prov("c", priority=1)]
        storage.set_last_used_provider("a")
        qt = QuotaTracker(storage)
        sel = ProviderSelector(providers, qt, storage, SelectionStrategyConfig(fairness_cursor=True))
        ordered = sel.select_ordered(estimated_tokens=100)
        # After using 'a', should rotate to [b, c, a]
        assert ordered[0].name == "b"
        assert ordered[-1].name == "a"

    def test_returns_empty_when_all_exhausted(self, storage: SQLiteStorage) -> None:
        providers = [_prov("a", priority=1, token_limit=10)]
        qt = QuotaTracker(storage)
        wk = qt.get_window_key(providers[0])
        storage.record_usage("a", wk, 5, 5, 10, "r1", "m", time.time())
        sel = ProviderSelector(providers, qt, storage, SelectionStrategyConfig())
        ordered = sel.select_ordered(estimated_tokens=100)
        assert ordered == []

    def test_priority_quota_prefers_more_remaining(self, storage: SQLiteStorage) -> None:
        # Same priority, different remaining quota
        providers = [_prov("a", priority=1, token_limit=1000), _prov("b", priority=1, token_limit=50000)]
        qt = QuotaTracker(storage)
        # Use up some of 'a' quota
        wk = qt.get_window_key(providers[0])
        storage.record_usage("a", wk, 200, 300, 500, "r1", "m", time.time())
        sel = ProviderSelector(providers, qt, storage, SelectionStrategyConfig(fairness_cursor=False))
        ordered = sel.select_ordered(estimated_tokens=100)
        # 'b' has more remaining (50000 vs 500), so should come first
        assert ordered[0].name == "b"
