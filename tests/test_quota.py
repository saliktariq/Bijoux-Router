"""Tests for the quota tracker."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from bijoux_router.config.schema import (
    PeriodType,
    ProviderConfig,
    QuotaConfig,
    ResetMode,
)
from bijoux_router.quota.tracker import QuotaTracker
from bijoux_router.storage.sqlite_backend import SQLiteStorage


def _make_provider(name: str = "test-p", token_limit: int = 1000, request_limit: int | None = None) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        provider_type="mock",
        quota=QuotaConfig(
            token_limit=token_limit,
            period_type=PeriodType.DAY,
            period_value=1,
            reset_mode=ResetMode.FIXED,
            request_limit=request_limit,
        ),
    )


class TestQuotaTracker:
    def test_initial_remaining_is_full(self, storage: SQLiteStorage) -> None:
        tracker = QuotaTracker(storage)
        p = _make_provider(token_limit=5000)
        assert tracker.get_remaining_tokens(p) == 5000

    def test_has_budget_with_capacity(self, storage: SQLiteStorage) -> None:
        tracker = QuotaTracker(storage)
        p = _make_provider(token_limit=5000)
        assert tracker.has_budget(p, 1000) is True

    def test_has_budget_insufficient(self, storage: SQLiteStorage) -> None:
        tracker = QuotaTracker(storage)
        p = _make_provider(token_limit=100)
        # Use up most of the quota
        window_key = tracker.get_window_key(p)
        storage.record_usage(p.name, window_key, 30, 60, 90, "r1", "m", time.time())
        assert tracker.has_budget(p, 50) is False

    def test_reservation_reduces_remaining(self, storage: SQLiteStorage) -> None:
        tracker = QuotaTracker(storage)
        p = _make_provider(token_limit=1000)
        r_id = tracker.create_reservation(p, 400)
        assert tracker.get_remaining_tokens(p) == 600

    def test_reconcile_releases_reservation_and_records(self, storage: SQLiteStorage) -> None:
        tracker = QuotaTracker(storage)
        p = _make_provider(token_limit=1000)
        r_id = tracker.create_reservation(p, 400)
        # Reconcile with actual usage (less than estimated)
        tracker.reconcile(p, r_id, actual_prompt=50, actual_completion=100, actual_total=150, request_id="r1", model="m")
        # Remaining should be 1000 - 150 (actual) + 0 (reservation released)
        assert tracker.get_remaining_tokens(p) == 850

    def test_release_reservation_on_failure(self, storage: SQLiteStorage) -> None:
        tracker = QuotaTracker(storage)
        p = _make_provider(token_limit=1000)
        r_id = tracker.create_reservation(p, 400)
        assert tracker.get_remaining_tokens(p) == 600
        tracker.release_reservation(r_id)
        assert tracker.get_remaining_tokens(p) == 1000

    def test_request_limit(self, storage: SQLiteStorage) -> None:
        tracker = QuotaTracker(storage)
        p = _make_provider(token_limit=100000, request_limit=2)
        window_key = tracker.get_window_key(p)
        storage.record_usage(p.name, window_key, 10, 10, 20, "r1", "m", time.time())
        storage.record_usage(p.name, window_key, 10, 10, 20, "r2", "m", time.time())
        assert tracker.get_remaining_requests(p) == 0
        assert tracker.has_budget(p, 10) is False

    def test_quota_status_report(self, storage: SQLiteStorage) -> None:
        tracker = QuotaTracker(storage)
        p = _make_provider(token_limit=10000)
        window_key = tracker.get_window_key(p)
        storage.record_usage(p.name, window_key, 100, 200, 300, "r1", "m", time.time())
        tracker.create_reservation(p, 500)
        status = tracker.get_quota_status(p)
        assert status["token_limit"] == 10000
        assert status["used_tokens"] == 300
        assert status["reserved_tokens"] == 500
        assert status["remaining_tokens"] == 9200
        assert status["utilization_pct"] == 8.0
