"""YAML configuration loader with environment variable interpolation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from bijoux_router.config.schema import GatewayConfig
from bijoux_router.exceptions.errors import ConfigurationError, SecretResolutionError

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _interpolate_env(value: Any) -> Any:
    """Recursively replace ${ENV_VAR} placeholders with environment values."""
    if isinstance(value, str):
        def _replace(match: re.Match[str]) -> str:
            var_name = match.group(1)
            env_val = os.environ.get(var_name)
            if env_val is None:
                raise SecretResolutionError(
                    f"Environment variable '{var_name}' referenced in config is not set"
                )
            return env_val
        return _ENV_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(item) for item in value]
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load and parse a YAML file, returning raw dict."""
    path = Path(path)
    if not path.exists():
        raise ConfigurationError(f"Configuration file not found: {path}")
    if not path.is_file():
        raise ConfigurationError(f"Configuration path is not a file: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"YAML parsing error in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError(f"Expected top-level mapping in {path}, got {type(data).__name__}")
    return data


def load_config(path: str | Path) -> GatewayConfig:
    """Load, interpolate, validate, and return a GatewayConfig from a YAML file."""
    raw = load_yaml(path)
    try:
        interpolated = _interpolate_env(raw)
    except SecretResolutionError:
        raise
    except Exception as exc:
        raise ConfigurationError(f"Error interpolating config: {exc}") from exc
    try:
        config = GatewayConfig.model_validate(interpolated)
    except Exception as exc:
        raise ConfigurationError(f"Configuration validation failed: {exc}") from exc
    return config


def validate_config(path: str | Path) -> list[str]:
    """Validate config file and return list of warnings (empty if clean)."""
    warnings: list[str] = []
    config = load_config(path)
    enabled = [p for p in config.providers if p.enabled]
    if not enabled:
        warnings.append("No providers are enabled")
    for p in config.providers:
        if p.enabled and not p.api_key:
            warnings.append(f"Provider '{p.name}' is enabled but has no api_key")
        if p.enabled and not p.default_model and not p.model_map:
            warnings.append(f"Provider '{p.name}' has no default_model and no model_map")
    return warnings
