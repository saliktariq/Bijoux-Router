"""Tests for configuration loading and validation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from bijoux_router.config.loader import load_config, validate_config
from bijoux_router.exceptions.errors import ConfigurationError, SecretResolutionError


class TestConfigLoading:
    def test_load_valid_config(self, sample_yaml_path: Path) -> None:
        config = load_config(sample_yaml_path)
        assert len(config.providers) == 1
        assert config.providers[0].name == "test-mock"

    def test_load_nonexistent_file(self) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            load_config("/nonexistent/path.yaml")

    def test_load_invalid_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("{{{{invalid yaml", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="YAML parsing error"):
            load_config(p)

    def test_load_non_mapping_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "list.yaml"
        p.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="Expected top-level mapping"):
            load_config(p)

    def test_empty_providers_fails(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.yaml"
        p.write_text(yaml.dump({"providers": []}), encoding="utf-8")
        with pytest.raises(ConfigurationError, match="validation failed"):
            load_config(p)

    def test_duplicate_provider_names_fails(self, tmp_path: Path) -> None:
        config = {
            "providers": [
                {"name": "dup", "provider_type": "mock"},
                {"name": "dup", "provider_type": "mock"},
            ]
        }
        p = tmp_path / "dup.yaml"
        p.write_text(yaml.dump(config), encoding="utf-8")
        with pytest.raises(ConfigurationError, match="validation failed"):
            load_config(p)

    def test_env_var_interpolation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BIJOUX_TEST_KEY", "secret-api-key-123")
        config = {
            "providers": [
                {
                    "name": "env-test",
                    "provider_type": "mock",
                    "api_key": "${BIJOUX_TEST_KEY}",
                    "default_model": "m",
                }
            ],
            "storage_path": str(tmp_path / "env.db"),
        }
        p = tmp_path / "env.yaml"
        p.write_text(yaml.dump(config), encoding="utf-8")
        loaded = load_config(p)
        assert loaded.providers[0].api_key == "secret-api-key-123"

    def test_missing_env_var_raises(self, tmp_path: Path) -> None:
        config = {
            "providers": [
                {
                    "name": "env-test",
                    "provider_type": "mock",
                    "api_key": "${BIJOUX_NONEXISTENT_VAR_XYZ}",
                }
            ],
        }
        p = tmp_path / "missing_env.yaml"
        p.write_text(yaml.dump(config), encoding="utf-8")
        with pytest.raises(SecretResolutionError, match="BIJOUX_NONEXISTENT_VAR_XYZ"):
            load_config(p)


class TestConfigValidation:
    def test_validate_reports_no_api_key(self, tmp_path: Path) -> None:
        config = {
            "providers": [
                {"name": "nokey", "provider_type": "mock", "enabled": True, "api_key": "", "default_model": "m"}
            ],
            "storage_path": str(tmp_path / "v.db"),
        }
        p = tmp_path / "nokey.yaml"
        p.write_text(yaml.dump(config), encoding="utf-8")
        warnings = validate_config(p)
        assert any("no api_key" in w for w in warnings)

    def test_validate_reports_no_model(self, tmp_path: Path) -> None:
        config = {
            "providers": [
                {"name": "nomodel", "provider_type": "mock", "enabled": True, "api_key": "k"}
            ],
            "storage_path": str(tmp_path / "v2.db"),
        }
        p = tmp_path / "nomodel.yaml"
        p.write_text(yaml.dump(config), encoding="utf-8")
        warnings = validate_config(p)
        assert any("no default_model" in w for w in warnings)
