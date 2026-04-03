"""Abstract storage interface for Bijoux Router persistence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StorageBackend(ABC):
    """Abstract persistence interface.

    All methods are synchronous. In a high-concurrency async context,
    wrap calls via asyncio.to_thread or use an async-native backend.
    """

    @abstractmethod
    def initialize(self) -> None:
        """Create tables/schema if they do not exist."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""

    # -- Usage records ------------------------------------------------------

    @abstractmethod
    def record_usage(
        self,
        provider_name: str,
        window_key: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        request_id: str,
        model: str,
        timestamp: float,
    ) -> None:
        """Persist a completed-request usage record."""

    @abstractmethod
    def get_window_usage(self, provider_name: str, window_key: str) -> dict[str, int]:
        """Return aggregate {prompt_tokens, completion_tokens, total_tokens, request_count} for a window."""

    @abstractmethod
    def delete_window_usage(self, provider_name: str, window_key: str) -> None:
        """Delete all usage records for a specific window (e.g., on reset)."""

    # -- Reservations -------------------------------------------------------

    @abstractmethod
    def create_reservation(
        self, reservation_id: str, provider_name: str, window_key: str, estimated_tokens: int, timestamp: float,
    ) -> None:
        """Create a token reservation before dispatching a request."""

    @abstractmethod
    def release_reservation(self, reservation_id: str) -> None:
        """Release (delete) a reservation after reconciliation or failure."""

    @abstractmethod
    def get_active_reservations(self, provider_name: str, window_key: str) -> int:
        """Return the total reserved tokens for a provider/window."""

    # -- Cooldown state -----------------------------------------------------

    @abstractmethod
    def set_cooldown(self, provider_name: str, until_timestamp: float, reason: str) -> None:
        """Mark a provider as in cooldown until the given timestamp."""

    @abstractmethod
    def get_cooldown(self, provider_name: str) -> float | None:
        """Return cooldown-until timestamp, or None if not in cooldown."""

    @abstractmethod
    def clear_cooldown(self, provider_name: str) -> None:
        """Remove cooldown for a provider."""

    # -- Failure counters ---------------------------------------------------

    @abstractmethod
    def increment_failure(self, provider_name: str) -> int:
        """Increment and return the consecutive failure count."""

    @abstractmethod
    def reset_failures(self, provider_name: str) -> None:
        """Reset consecutive failure counter to 0."""

    @abstractmethod
    def get_failure_count(self, provider_name: str) -> int:
        """Return current consecutive failure count."""

    # -- Fairness cursor ----------------------------------------------------

    @abstractmethod
    def get_last_used_provider(self) -> str | None:
        """Return the name of the last successfully used provider."""

    @abstractmethod
    def set_last_used_provider(self, provider_name: str) -> None:
        """Record the last successfully used provider."""

    # -- Bulk / admin -------------------------------------------------------

    @abstractmethod
    def reset_provider_usage(self, provider_name: str) -> None:
        """Delete all usage, reservations, cooldowns, and failures for a provider."""

    @abstractmethod
    def get_all_provider_states(self) -> list[dict[str, Any]]:
        """Return a summary of all provider states for status inspection."""
