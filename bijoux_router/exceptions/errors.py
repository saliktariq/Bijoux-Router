"""Bijoux Router — normalized error taxonomy and exception hierarchy."""

from __future__ import annotations

from enum import Enum
from typing import Any


class ProviderErrorCategory(str, Enum):
    """Normalized error categories that all provider adapters map into."""

    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    RATE_LIMITED = "RATE_LIMITED"
    INSUFFICIENT_CREDIT = "INSUFFICIENT_CREDIT"
    BILLING_BLOCKED = "BILLING_BLOCKED"
    AUTH_ERROR = "AUTH_ERROR"
    INVALID_REQUEST = "INVALID_REQUEST"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    TRANSIENT_ERROR = "TRANSIENT_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    TIMEOUT = "TIMEOUT"
    PROVIDER_MISCONFIGURATION = "PROVIDER_MISCONFIGURATION"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"

    @property
    def is_quota_related(self) -> bool:
        return self in _QUOTA_RELATED

    @property
    def is_retriable_transient(self) -> bool:
        return self in _TRANSIENT_RETRIABLE

    @property
    def should_failover(self) -> bool:
        return self in _FAILOVER_ELIGIBLE


_QUOTA_RELATED = frozenset({
    ProviderErrorCategory.QUOTA_EXHAUSTED,
    ProviderErrorCategory.RATE_LIMITED,
    ProviderErrorCategory.INSUFFICIENT_CREDIT,
    ProviderErrorCategory.BILLING_BLOCKED,
})

_TRANSIENT_RETRIABLE = frozenset({
    ProviderErrorCategory.TRANSIENT_ERROR,
    ProviderErrorCategory.NETWORK_ERROR,
    ProviderErrorCategory.TIMEOUT,
})

_FAILOVER_ELIGIBLE = _QUOTA_RELATED | _TRANSIENT_RETRIABLE | frozenset({
    ProviderErrorCategory.MODEL_UNAVAILABLE,
})


# ---------------------------------------------------------------------------
# Base exception
# ---------------------------------------------------------------------------

class BijouxError(Exception):
    """Root exception for all Bijoux Router errors."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------

class ConfigurationError(BijouxError):
    """Raised when configuration is invalid or cannot be loaded."""


class SecretResolutionError(ConfigurationError):
    """Raised when an environment variable referenced in config is missing."""


# ---------------------------------------------------------------------------
# Provider errors
# ---------------------------------------------------------------------------

class ProviderError(BijouxError):
    """Base class for errors originating from or related to a provider."""

    def __init__(
        self,
        message: str,
        *,
        provider_name: str,
        category: ProviderErrorCategory,
        status_code: int | None = None,
        raw_response: Any = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.provider_name = provider_name
        self.category = category
        self.status_code = status_code
        self.raw_response = raw_response


class QuotaExhaustedError(ProviderError):
    """Provider quota or rate limit exhausted."""

    def __init__(self, message: str, *, provider_name: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            provider_name=provider_name,
            category=ProviderErrorCategory.QUOTA_EXHAUSTED,
            **kwargs,
        )


class InsufficientCreditError(ProviderError):
    """Provider billing/credit issue."""

    def __init__(self, message: str, *, provider_name: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            provider_name=provider_name,
            category=ProviderErrorCategory.INSUFFICIENT_CREDIT,
            **kwargs,
        )


class AuthenticationError(ProviderError):
    """Provider authentication/authorization failure."""

    def __init__(self, message: str, *, provider_name: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            provider_name=provider_name,
            category=ProviderErrorCategory.AUTH_ERROR,
            **kwargs,
        )


class ModelUnavailableError(ProviderError):
    """Requested model not available on provider."""

    def __init__(self, message: str, *, provider_name: str, **kwargs: Any) -> None:
        super().__init__(
            message,
            provider_name=provider_name,
            category=ProviderErrorCategory.MODEL_UNAVAILABLE,
            **kwargs,
        )


class TransientProviderError(ProviderError):
    """Transient/retryable provider error (5xx, network issue, timeout)."""

    def __init__(
        self,
        message: str,
        *,
        provider_name: str,
        category: ProviderErrorCategory = ProviderErrorCategory.TRANSIENT_ERROR,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, provider_name=provider_name, category=category, **kwargs)


# ---------------------------------------------------------------------------
# Router-level errors
# ---------------------------------------------------------------------------

class AllProvidersExhaustedError(BijouxError):
    """Raised when no viable provider remains for a request."""

    def __init__(
        self,
        message: str = "All providers exhausted or unavailable",
        *,
        attempts: list[dict[str, Any]] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.attempts = attempts or []


class NoViableProviderError(BijouxError):
    """Raised when provider selection finds zero candidates before any attempt."""


class RequestValidationError(BijouxError):
    """Raised when the inbound LLM request fails validation."""


class StorageError(BijouxError):
    """Raised on persistence layer failures."""
