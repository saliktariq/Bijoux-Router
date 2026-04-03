"""Quota tracker — manages token budgets, reservations, and window arithmetic."""

from __future__ import annotations

import time
import uuid
from calendar import monthrange
from datetime import datetime, timezone
from typing import Any

from bijoux_router.config.schema import PeriodType, ProviderConfig, ResetMode
from bijoux_router.storage.base import StorageBackend
from bijoux_router.utils.logging import get_logger

logger = get_logger("quota.tracker")


def _compute_window_key(period_type: PeriodType, period_value: int, reset_mode: ResetMode) -> str:
    """Compute a deterministic window key for the current time period.

    For FIXED reset mode, windows align to calendar boundaries.
    For ROLLING mode, windows are based on epoch seconds (more complex; here we
    approximate with short-lived fixed windows sized to period_value).
    """
    now = datetime.now(timezone.utc)
    if reset_mode == ResetMode.ROLLING:
        # Rolling windows are approximated as epoch-based buckets
        period_seconds = _period_to_seconds(period_type, period_value)
        bucket = int(time.time()) // period_seconds
        return f"rolling_{period_type.value}_{period_value}_{bucket}"

    # Fixed calendar-aligned windows
    if period_type == PeriodType.MINUTE:
        return now.strftime(f"%Y%m%d%H%M_m{period_value}")
    if period_type == PeriodType.HOUR:
        return now.strftime(f"%Y%m%d%H_h{period_value}")
    if period_type == PeriodType.DAY:
        return now.strftime(f"%Y%m%d_d{period_value}")
    if period_type == PeriodType.MONTH:
        return now.strftime(f"%Y%m_M{period_value}")
    # CUSTOM falls back to epoch bucket
    period_seconds = _period_to_seconds(period_type, period_value)
    bucket = int(time.time()) // period_seconds
    return f"custom_{period_value}s_{bucket}"


def _period_to_seconds(period_type: PeriodType, period_value: int) -> int:
    """Convert a period specification to seconds."""
    if period_type == PeriodType.MINUTE:
        return 60 * period_value
    if period_type == PeriodType.HOUR:
        return 3600 * period_value
    if period_type == PeriodType.DAY:
        return 86400 * period_value
    if period_type == PeriodType.MONTH:
        # Approximate month as 30 days
        return 86400 * 30 * period_value
    # CUSTOM: period_value is seconds
    return period_value


class QuotaTracker:
    """Manages per-provider token quota budgets.

    Key design: Quota tracking is approximate.  Before dispatch, we reserve
    estimated tokens.  After dispatch, we reconcile with actual usage.
    If a provider rejects a request due to real quota exhaustion, the
    router handles failover.
    """

    def __init__(self, storage: StorageBackend) -> None:
        self._storage = storage

    def get_window_key(self, config: ProviderConfig) -> str:
        """Return the current window key for a provider's quota config."""
        return _compute_window_key(
            config.quota.period_type,
            config.quota.period_value,
            config.quota.reset_mode,
        )

    def get_remaining_tokens(self, config: ProviderConfig) -> int:
        """Return estimated remaining tokens in the current window.

        Accounts for both committed usage and active reservations.
        """
        window_key = self.get_window_key(config)
        usage = self._storage.get_window_usage(config.name, window_key)
        reservations = self._storage.get_active_reservations(config.name, window_key)
        used = usage["total_tokens"] + reservations
        remaining = config.quota.token_limit - used
        return max(0, remaining)

    def get_remaining_requests(self, config: ProviderConfig) -> int | None:
        """Return remaining request count, or None if not request-limited."""
        if config.quota.request_limit is None:
            return None
        window_key = self.get_window_key(config)
        usage = self._storage.get_window_usage(config.name, window_key)
        return max(0, config.quota.request_limit - usage["request_count"])

    def has_budget(self, config: ProviderConfig, estimated_tokens: int) -> bool:
        """Check whether there is likely enough quota for a request."""
        remaining = self.get_remaining_tokens(config)
        if remaining < estimated_tokens:
            return False
        remaining_reqs = self.get_remaining_requests(config)
        if remaining_reqs is not None and remaining_reqs <= 0:
            return False
        return True

    def create_reservation(self, config: ProviderConfig, estimated_tokens: int) -> str:
        """Reserve estimated tokens before dispatching.  Returns reservation ID."""
        reservation_id = uuid.uuid4().hex
        window_key = self.get_window_key(config)
        self._storage.create_reservation(
            reservation_id=reservation_id,
            provider_name=config.name,
            window_key=window_key,
            estimated_tokens=estimated_tokens,
            timestamp=time.time(),
        )
        logger.debug(
            "Created reservation %s for %s: %d tokens in window %s",
            reservation_id, config.name, estimated_tokens, window_key,
        )
        return reservation_id

    def reconcile(
        self,
        config: ProviderConfig,
        reservation_id: str,
        actual_prompt: int,
        actual_completion: int,
        actual_total: int,
        request_id: str,
        model: str,
    ) -> None:
        """Reconcile a reservation with actual provider-reported usage.

        Releases the reservation and records actual usage.
        """
        window_key = self.get_window_key(config)
        # Release reservation
        self._storage.release_reservation(reservation_id)
        # Record actual usage
        self._storage.record_usage(
            provider_name=config.name,
            window_key=window_key,
            prompt_tokens=actual_prompt,
            completion_tokens=actual_completion,
            total_tokens=actual_total,
            request_id=request_id,
            model=model,
            timestamp=time.time(),
        )
        logger.debug(
            "Reconciled reservation %s for %s: actual=%d tokens",
            reservation_id, config.name, actual_total,
        )

    def release_reservation(self, reservation_id: str) -> None:
        """Release a reservation without recording usage (e.g., on failure)."""
        self._storage.release_reservation(reservation_id)
        logger.debug("Released reservation %s (no usage recorded)", reservation_id)

    def get_quota_status(self, config: ProviderConfig) -> dict[str, Any]:
        """Return a detailed quota status dict for a provider."""
        window_key = self.get_window_key(config)
        usage = self._storage.get_window_usage(config.name, window_key)
        reservations = self._storage.get_active_reservations(config.name, window_key)
        remaining = max(0, config.quota.token_limit - usage["total_tokens"] - reservations)
        return {
            "provider_name": config.name,
            "window_key": window_key,
            "token_limit": config.quota.token_limit,
            "used_tokens": usage["total_tokens"],
            "reserved_tokens": reservations,
            "remaining_tokens": remaining,
            "request_count": usage["request_count"],
            "request_limit": config.quota.request_limit,
            "period_type": config.quota.period_type.value,
            "period_value": config.quota.period_value,
            "utilization_pct": round(
                (usage["total_tokens"] + reservations) / max(1, config.quota.token_limit) * 100, 2
            ),
        }
