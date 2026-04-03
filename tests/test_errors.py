"""Tests for error taxonomy and exception hierarchy."""

from __future__ import annotations

from bijoux_router.exceptions.errors import (
    AllProvidersExhaustedError,
    AuthenticationError,
    BijouxError,
    ConfigurationError,
    InsufficientCreditError,
    ModelUnavailableError,
    NoViableProviderError,
    ProviderError,
    ProviderErrorCategory,
    QuotaExhaustedError,
    RequestValidationError,
    SecretResolutionError,
    StorageError,
    TransientProviderError,
)


class TestErrorCategories:
    def test_quota_related(self) -> None:
        assert ProviderErrorCategory.QUOTA_EXHAUSTED.is_quota_related
        assert ProviderErrorCategory.RATE_LIMITED.is_quota_related
        assert ProviderErrorCategory.INSUFFICIENT_CREDIT.is_quota_related
        assert ProviderErrorCategory.BILLING_BLOCKED.is_quota_related
        assert not ProviderErrorCategory.AUTH_ERROR.is_quota_related

    def test_retriable_transient(self) -> None:
        assert ProviderErrorCategory.TRANSIENT_ERROR.is_retriable_transient
        assert ProviderErrorCategory.NETWORK_ERROR.is_retriable_transient
        assert ProviderErrorCategory.TIMEOUT.is_retriable_transient
        assert not ProviderErrorCategory.QUOTA_EXHAUSTED.is_retriable_transient

    def test_should_failover(self) -> None:
        assert ProviderErrorCategory.QUOTA_EXHAUSTED.should_failover
        assert ProviderErrorCategory.TRANSIENT_ERROR.should_failover
        assert ProviderErrorCategory.MODEL_UNAVAILABLE.should_failover
        assert not ProviderErrorCategory.AUTH_ERROR.should_failover
        assert not ProviderErrorCategory.INVALID_REQUEST.should_failover


class TestExceptionHierarchy:
    def test_all_inherit_from_bijoux_error(self) -> None:
        assert issubclass(ConfigurationError, BijouxError)
        assert issubclass(SecretResolutionError, BijouxError)
        assert issubclass(ProviderError, BijouxError)
        assert issubclass(AllProvidersExhaustedError, BijouxError)
        assert issubclass(NoViableProviderError, BijouxError)
        assert issubclass(RequestValidationError, BijouxError)
        assert issubclass(StorageError, BijouxError)

    def test_provider_errors_inherit_from_provider_error(self) -> None:
        assert issubclass(QuotaExhaustedError, ProviderError)
        assert issubclass(InsufficientCreditError, ProviderError)
        assert issubclass(AuthenticationError, ProviderError)
        assert issubclass(ModelUnavailableError, ProviderError)
        assert issubclass(TransientProviderError, ProviderError)

    def test_provider_error_has_fields(self) -> None:
        exc = QuotaExhaustedError("out of quota", provider_name="test-p", status_code=429)
        assert exc.provider_name == "test-p"
        assert exc.category == ProviderErrorCategory.QUOTA_EXHAUSTED
        assert exc.status_code == 429

    def test_all_providers_exhausted_has_attempts(self) -> None:
        exc = AllProvidersExhaustedError(attempts=[{"provider": "a"}, {"provider": "b"}])
        assert len(exc.attempts) == 2
