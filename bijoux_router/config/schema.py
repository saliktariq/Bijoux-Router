"""Configuration schema models for Bijoux Router."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class PeriodType(str, Enum):
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    MONTH = "month"
    CUSTOM = "custom"


class ResetMode(str, Enum):
    FIXED = "fixed"
    ROLLING = "rolling"


class QuotaConfig(BaseModel):
    token_limit: int = 1_000_000
    request_limit: int | None = None
    period_type: PeriodType = PeriodType.DAY
    period_value: int = 1
    reset_mode: ResetMode = ResetMode.FIXED


class RetryPolicyConfig(BaseModel):
    max_retries: int = 2
    backoff_base: float = 1.0
    backoff_max: float = 10.0
    retry_on_transient: bool = True


class CooldownPolicyConfig(BaseModel):
    cooldown_seconds: float = 60.0
    failure_threshold: int = 3
    quota_exhaustion_cooldown_seconds: float = 300.0


class CostMetadata(BaseModel):
    """Optional cost-per-token metadata for future cheapest-provider routing."""
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0
    currency: str = "USD"


class ProviderConfig(BaseModel):
    """Configuration for a single provider/account."""

    name: str
    enabled: bool = True
    provider_type: str  # "openrouter", "gemini", "openai_compatible", "mock"
    base_url: str = ""
    api_key: str = ""  # may contain ${ENV_VAR} references
    default_model: str = ""
    model_map: dict[str, str] = Field(default_factory=dict)
    priority: int = 0  # lower = higher priority
    selection_weight: float = 1.0
    timeout_seconds: float = 60.0
    retry_policy: RetryPolicyConfig = Field(default_factory=RetryPolicyConfig)
    cooldown_policy: CooldownPolicyConfig = Field(default_factory=CooldownPolicyConfig)
    quota: QuotaConfig = Field(default_factory=QuotaConfig)
    headers: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    failover_enabled: bool = True
    continue_on_auth_error: bool = False
    continue_on_invalid_request: bool = False
    cost: CostMetadata | None = None

    def resolve_model(self, requested_model: str | None) -> str:
        """Map a caller-requested model name to this provider's actual model name."""
        if requested_model and requested_model in self.model_map:
            return self.model_map[requested_model]
        if requested_model:
            return requested_model
        return self.default_model


class SelectionStrategyType(str, Enum):
    PRIORITY_QUOTA = "priority_quota"
    WEIGHTED = "weighted"
    CHEAPEST = "cheapest"
    ROUND_ROBIN = "round_robin"


class SelectionStrategyConfig(BaseModel):
    strategy_type: SelectionStrategyType = SelectionStrategyType.PRIORITY_QUOTA
    fairness_cursor: bool = True


class GatewayConfig(BaseModel):
    """Root configuration for the Bijoux Router."""

    providers: list[ProviderConfig]
    selection_strategy: SelectionStrategyConfig = Field(default_factory=SelectionStrategyConfig)
    default_timeout_seconds: float = 60.0
    max_failover_attempts: int = 5
    storage_path: str = "bijoux_state.db"
    log_level: str = "INFO"
    redact_secrets: bool = True

    @model_validator(mode="after")
    def _validate_at_least_one_provider(self) -> GatewayConfig:
        if not self.providers:
            raise ValueError("At least one provider must be configured")
        names = [p.name for p in self.providers]
        if len(names) != len(set(names)):
            raise ValueError("Provider names must be unique")
        return self
