# Bijoux Router

A production-grade Python library for multi-provider LLM routing with quota tracking, transparent failover, and persistent accounting.

## What It Does

You have accounts with multiple LLM providers (OpenRouter, Gemini, OpenAI-compatible endpoints, etc.), each with different quotas, rate limits, and pricing. Bijoux provides:

- **One normalized interface** — callers never deal with provider-specific APIs
- **Quota-aware provider selection** — routes to the provider most likely to succeed within its token budget
- **Transparent failover** — if a provider rejects a request (quota exhausted, rate limited, billing blocked), the same request is automatically replayed against the next viable provider
- **Persistent usage tracking** — token accounting survives process restarts via SQLite
- **Reservation/reconciliation** — estimates tokens before dispatch, reconciles with actual provider-reported usage after
- **Cooldown and health tracking** — backs off from repeatedly-failing providers
- **Extensible adapter architecture** — add new providers by implementing one abstract class

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   Caller Code                       │
│         router.process(LLMRequest(...))             │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              BijouxRouter (engine.py)                │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │  Selection   │  │    Quota     │  │  Retry /  │  │
│  │  Strategy    │  │   Tracker    │  │ Failover  │  │
│  └──────┬──────┘  └──────┬───────┘  └─────┬─────┘  │
└─────────┼────────────────┼─────────────────┼────────┘
          │                │                 │
┌─────────▼────────────────▼─────────────────▼────────┐
│              Provider Adapters                       │
│  ┌───────────┐ ┌────────┐ ┌──────────────┐ ┌─────┐ │
│  │OpenRouter │ │ Gemini │ │OpenAI-Compat │ │Mock │ │
│  └───────────┘ └────────┘ └──────────────┘ └─────┘ │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│           Storage (SQLite)                           │
│  usage_records │ reservations │ cooldowns │ kv_store │
└─────────────────────────────────────────────────────┘
```

## Project Structure

```
bijoux_router/
├── __init__.py              # Public API re-exports
├── config/
│   ├── loader.py            # YAML loading, env interpolation, validation
│   └── schema.py            # Pydantic config models
├── models/
│   └── request_response.py  # LLMRequest, LLMResponse, TokenUsage, etc.
├── providers/
│   ├── base.py              # BaseProviderClient abstract class
│   ├── factory.py           # Provider registry and factory
│   ├── openai_compatible.py # OpenAI-compatible adapter
│   ├── openrouter.py        # OpenRouter-specific adapter
│   ├── gemini.py            # Google Gemini adapter
│   └── mock.py              # Mock provider for tests
├── quota/
│   └── tracker.py           # Quota management, reservation, reconciliation
├── router/
│   ├── engine.py            # Core orchestrator (BijouxRouter)
│   └── selection.py         # Provider selection strategies
├── storage/
│   ├── base.py              # Abstract storage interface
│   └── sqlite_backend.py    # SQLite implementation
├── cli/
│   └── main.py              # Click CLI commands
├── utils/
│   ├── logging.py           # Structured logging with secret redaction
│   └── tokens.py            # Token estimation utilities
└── exceptions/
    └── errors.py            # Error taxonomy and exception hierarchy

tests/
├── conftest.py              # Shared fixtures
├── test_config.py           # Configuration loading/validation
├── test_storage.py          # SQLite backend
├── test_quota.py            # Quota tracker
├── test_selection.py        # Provider selection strategy
├── test_router.py           # Full router integration tests
├── test_errors.py           # Error taxonomy
└── test_concurrent.py       # Concurrent access
```

## Quick Start

### Installation

```bash
pip install -e ".[dev]"
```

### Configuration

Create `config/providers.yaml`:

```yaml
selection_strategy:
  strategy_type: priority_quota
  fairness_cursor: true

storage_path: bijoux_state.db

providers:
  - name: openrouter-primary
    enabled: true
    provider_type: openrouter
    api_key: "${OPENROUTER_API_KEY}"
    default_model: openai/gpt-4o-mini
    priority: 1
    quota:
      token_limit: 500000
      period_type: day
      period_value: 1

  - name: gemini-free
    enabled: true
    provider_type: gemini
    api_key: "${GEMINI_API_KEY}"
    default_model: gemini-1.5-flash
    priority: 2
    quota:
      token_limit: 1000000
      period_type: day
      period_value: 1
```

### Usage

```python
import asyncio
from bijoux_router import BijouxRouter, LLMRequest

async def main():
    router = BijouxRouter.from_yaml("config/providers.yaml")

    response = await router.process(
        LLMRequest(
            prompt="Summarize this document",
            model="gpt-4o-mini",
            max_tokens=500,
            temperature=0.2,
        )
    )

    print(f"Provider: {response.provider_name}")
    print(f"Content: {response.content}")
    print(f"Tokens: {response.usage.total_tokens}")
    print(f"Failovers: {len(response.failover_attempts)}")

    router.close()

asyncio.run(main())
```

The caller does not need to know which provider was chosen, which failed first, how quota was calculated, or how provider-specific request formatting works.

### CLI

```bash
# Validate configuration
bijoux -c config/providers.yaml validate-config

# Show provider status
bijoux -c config/providers.yaml show-provider-status

# Show quota usage
bijoux -c config/providers.yaml show-quota

# Simulate a request
bijoux -c config/providers.yaml simulate-request -p "Hello world" -m gpt-4o-mini

# Reset usage for a provider
bijoux -c config/providers.yaml reset-provider-usage openrouter-primary
```

### Running Tests

```bash
pytest tests/ -v
```

## Configuration Reference

### Provider Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Unique provider identifier |
| `enabled` | bool | Whether to include in routing |
| `provider_type` | string | `openrouter`, `gemini`, `openai_compatible`, `mock` |
| `base_url` | string | API base URL |
| `api_key` | string | API key (supports `${ENV_VAR}` interpolation) |
| `default_model` | string | Default model when caller doesn't specify |
| `model_map` | dict | Map caller model names → provider model names |
| `priority` | int | Lower = higher priority |
| `selection_weight` | float | For weighted selection (future) |
| `timeout_seconds` | float | HTTP timeout |
| `retry_policy` | object | Transient retry config |
| `cooldown_policy` | object | Cooldown behavior config |
| `quota` | object | Token/request budget config |
| `headers` | dict | Extra HTTP headers |
| `tags` | list | Metadata tags |
| `failover_enabled` | bool | Allow failover from this provider |
| `continue_on_auth_error` | bool | Failover on auth errors |
| `continue_on_invalid_request` | bool | Failover on invalid request errors |
| `cost` | object | Cost-per-token metadata (future) |

### Quota Configuration

| Field | Type | Options |
|-------|------|---------|
| `token_limit` | int | Max tokens per period |
| `request_limit` | int? | Max requests per period |
| `period_type` | enum | `minute`, `hour`, `day`, `month`, `custom` |
| `period_value` | int | Number of periods |
| `reset_mode` | enum | `fixed` (calendar-aligned) or `rolling` |

### Error Taxonomy

All provider-specific errors are classified into:

| Category | Triggers Failover | Triggers Cooldown |
|----------|:-:|:-:|
| `QUOTA_EXHAUSTED` | ✓ | ✓ (quota cooldown) |
| `RATE_LIMITED` | ✓ | ✓ (quota cooldown) |
| `INSUFFICIENT_CREDIT` | ✓ | ✓ (quota cooldown) |
| `BILLING_BLOCKED` | ✓ | ✓ (quota cooldown) |
| `AUTH_ERROR` | Policy-driven | ✗ |
| `INVALID_REQUEST` | Policy-driven | ✗ |
| `MODEL_UNAVAILABLE` | ✓ | ✗ |
| `TRANSIENT_ERROR` | ✓ | After threshold |
| `NETWORK_ERROR` | ✓ | After threshold |
| `TIMEOUT` | ✓ | After threshold |

## Design Tradeoffs

### Estimated vs. Provider-Reported Token Usage

The system estimates tokens before dispatch (using a character-count heuristic at ~4 chars/token) and reserves that estimate against quota. After the provider responds, actual reported usage replaces the estimate. If the provider omits usage (some free-tier APIs do), the estimate is persisted as best-effort. This means:

- **Quota can drift** slightly from reality over time
- **Conservative estimates** avoid under-counting but may cause premature provider skipping
- **Failover catches the real discrepancy** — if estimates were wrong and the provider rejects, we simply try the next one

### Fixed Windows vs. Rolling Windows

Fixed windows align to calendar boundaries (start of day, hour, etc.) and are simpler to implement and query. Rolling windows prevent burst-at-boundary problems but require more complex bookkeeping. The system supports both via `reset_mode`, with fixed windows using deterministic window keys and rolling windows approximated via epoch-second buckets.

### SQLite Concurrency

SQLite supports multiple readers but only one writer at a time. The implementation uses:
- WAL mode for better read concurrency
- A threading lock for write serialization within a process
- `busy_timeout` for brief cross-connection contention

For high-concurrency multi-process scenarios, substitute a PostgreSQL `StorageBackend` implementation. The abstract interface makes this a drop-in replacement.

### Invalid Request Errors: Fail Fast vs. Continue

By default, `INVALID_REQUEST` (HTTP 400) errors do **not** trigger failover — the assumption is that if the request is malformed, it will likely fail on all providers. The `continue_on_invalid_request` flag overrides this for cases where provider-specific validation differs.

### Model Portability

The `model_map` configuration bridges model name differences across providers. When a caller requests `gpt-4o-mini`, each provider maps it to its own equivalent. This works well for capability-equivalent models but cannot guarantee identical output quality or behavior across providers.

### Idempotency

LLM completions are inherently non-deterministic. Replaying the same request on a different provider will produce different output but satisfies the same semantic intent. The `request_id` is preserved across failover attempts for correlation/debugging, not for deduplication.

## Extending with New Providers

```python
from bijoux_router.providers.base import BaseProviderClient
from bijoux_router.providers.factory import register_provider_type

class MyCustomClient(BaseProviderClient):
    async def send_request(self, request):
        # Implement provider-specific HTTP call
        ...

    def classify_error(self, status_code, body):
        # Map errors to ProviderErrorCategory
        ...

register_provider_type("my_provider", MyCustomClient)
```

Then in YAML:
```yaml
- name: my-endpoint
  provider_type: my_provider
  ...
```

## Integration Targets

Bijoux is designed as a reusable internal component. It can sit behind:

- **FastAPI service** — wrap `router.process()` in an endpoint
- **CLI tool** — the built-in CLI or custom Click commands
- **Background worker** — Celery/RQ tasks calling `router.process()`
- **Any Python application** — import and call directly

The central contract: *"One call in, best available provider selected, quota tracked, graceful automatic failover if provider quota/credits are actually exhausted."*
