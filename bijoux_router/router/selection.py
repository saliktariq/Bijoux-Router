"""Provider selection strategies for the Bijoux Router."""

from __future__ import annotations

import time
from typing import Any

from bijoux_router.config.schema import ProviderConfig, SelectionStrategyConfig, SelectionStrategyType
from bijoux_router.quota.tracker import QuotaTracker
from bijoux_router.storage.base import StorageBackend
from bijoux_router.utils.logging import get_logger
from bijoux_router.utils.tokens import estimate_tokens

logger = get_logger("router.selection")


class ProviderSelector:
    """Selects viable providers ordered by strategy.

    Responsibilities:
    1. Filter out disabled providers
    2. Filter out providers in cooldown
    3. Filter out providers with insufficient quota
    4. Order remaining providers by strategy
    5. Apply fairness cursor if enabled
    """

    def __init__(
        self,
        providers: list[ProviderConfig],
        quota_tracker: QuotaTracker,
        storage: StorageBackend,
        strategy_config: SelectionStrategyConfig,
    ) -> None:
        self._providers = providers
        self._quota = quota_tracker
        self._storage = storage
        self._strategy = strategy_config

    def select_ordered(
        self,
        estimated_tokens: int,
        *,
        request_model: str | None = None,
        tags: list[str] | None = None,
    ) -> list[ProviderConfig]:
        """Return an ordered list of viable providers for this request.

        The first element is the "best" choice.  The router will try
        them in order, failing over as needed.
        """
        now = time.time()
        viable: list[ProviderConfig] = []

        for p in self._providers:
            if not p.enabled:
                logger.debug("Skipping %s: disabled", p.name)
                continue

            # Cooldown check
            cooldown_until = self._storage.get_cooldown(p.name)
            if cooldown_until is not None and cooldown_until > now:
                logger.debug("Skipping %s: in cooldown until %.0f", p.name, cooldown_until)
                continue

            # Check if model is resolvable
            if request_model and not p.resolve_model(request_model):
                logger.debug("Skipping %s: cannot resolve model '%s'", p.name, request_model)
                continue

            # Quota check
            if not self._quota.has_budget(p, estimated_tokens):
                logger.debug("Skipping %s: insufficient quota (need %d)", p.name, estimated_tokens)
                continue

            viable.append(p)

        if not viable:
            return []

        # Order by strategy
        ordered = self._apply_strategy(viable)

        # Apply fairness cursor
        if self._strategy.fairness_cursor:
            ordered = self._apply_fairness(ordered)

        return ordered

    def _apply_strategy(self, providers: list[ProviderConfig]) -> list[ProviderConfig]:
        """Sort providers according to the selected strategy."""
        strategy = self._strategy.strategy_type

        if strategy == SelectionStrategyType.PRIORITY_QUOTA:
            # Sort by priority (lower = better), then by remaining quota (higher = better)
            def score(p: ProviderConfig) -> tuple[int, int]:
                remaining = self._quota.get_remaining_tokens(p)
                return (p.priority, -remaining)  # negate remaining so higher remaining sorts first
            return sorted(providers, key=score)

        if strategy == SelectionStrategyType.CHEAPEST:
            def cost_score(p: ProviderConfig) -> float:
                if p.cost:
                    return p.cost.input_cost_per_1k + p.cost.output_cost_per_1k
                return float("inf")
            return sorted(providers, key=cost_score)

        if strategy == SelectionStrategyType.ROUND_ROBIN:
            return sorted(providers, key=lambda p: p.priority)

        # Default: priority order
        return sorted(providers, key=lambda p: p.priority)

    def _apply_fairness(self, providers: list[ProviderConfig]) -> list[ProviderConfig]:
        """Rotate provider order so the last-used provider is not always first.

        If providers [A, B, C] are ordered and A was last used, rotate to [B, C, A].
        This distributes load across providers with equal priority.
        """
        if len(providers) <= 1:
            return providers

        last_used = self._storage.get_last_used_provider()
        if not last_used:
            return providers

        # Find last_used index
        names = [p.name for p in providers]
        if last_used not in names:
            return providers

        idx = names.index(last_used)
        # Only rotate among providers with the same priority as the first one
        top_priority = providers[0].priority
        same_priority = [p for p in providers if p.priority == top_priority]
        rest = [p for p in providers if p.priority != top_priority]

        if len(same_priority) <= 1:
            return providers

        # Rotate same-priority group if last_used is in it
        sp_names = [p.name for p in same_priority]
        if last_used in sp_names:
            sp_idx = sp_names.index(last_used)
            same_priority = same_priority[sp_idx + 1:] + same_priority[:sp_idx + 1]

        return same_priority + rest
