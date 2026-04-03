"""Bijoux Router CLI — configuration validation, status, and admin commands."""

from __future__ import annotations

import asyncio
import json
import sys

import click

from bijoux_router.config.loader import load_config, validate_config
from bijoux_router.models.request_response import LLMRequest
from bijoux_router.router.engine import BijouxRouter
from bijoux_router.utils.logging import get_logger

logger = get_logger("cli")


@click.group()
@click.option("--config", "-c", default="config/providers.yaml", help="Path to providers YAML config.")
@click.pass_context
def cli(ctx: click.Context, config: str) -> None:
    """Bijoux Quota-Aware LLM Gateway CLI."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


@cli.command("validate-config")
@click.pass_context
def validate_config_cmd(ctx: click.Context) -> None:
    """Validate the YAML configuration file."""
    path = ctx.obj["config_path"]
    try:
        warnings = validate_config(path)
        click.echo(f"Configuration is valid: {path}")
        if warnings:
            for w in warnings:
                click.echo(f"  WARNING: {w}")
        else:
            click.echo("  No warnings.")
    except Exception as exc:
        click.echo(f"Configuration error: {exc}", err=True)
        sys.exit(1)


@cli.command("show-provider-status")
@click.pass_context
def show_provider_status(ctx: click.Context) -> None:
    """Show the current status of all configured providers."""
    path = ctx.obj["config_path"]
    try:
        router = BijouxRouter.from_yaml(path)
        statuses = router.get_provider_status()
        click.echo(json.dumps(statuses, indent=2, default=str))
        router.close()
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command("show-quota")
@click.pass_context
def show_quota(ctx: click.Context) -> None:
    """Show quota status for all providers."""
    path = ctx.obj["config_path"]
    try:
        router = BijouxRouter.from_yaml(path)
        quotas = router.get_quota_status()
        click.echo(json.dumps(quotas, indent=2, default=str))
        router.close()
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command("simulate-request")
@click.option("--prompt", "-p", required=True, help="Prompt text to simulate.")
@click.option("--model", "-m", default=None, help="Model name.")
@click.option("--max-tokens", default=100, help="Max tokens for response.")
@click.pass_context
def simulate_request(ctx: click.Context, prompt: str, model: str | None, max_tokens: int) -> None:
    """Simulate an LLM request through the router."""
    path = ctx.obj["config_path"]
    try:
        router = BijouxRouter.from_yaml(path)
        request = LLMRequest(prompt=prompt, model=model, max_tokens=max_tokens)
        response = asyncio.run(router.process(request))
        output = {
            "request_id": response.request_id,
            "provider": response.provider_name,
            "model": response.model,
            "content": response.content[:200] + ("..." if len(response.content) > 200 else ""),
            "usage": response.usage.model_dump(),
            "finish_reason": response.finish_reason.value,
            "latency_ms": round(response.latency_ms, 2),
            "failover_attempts": len(response.failover_attempts),
        }
        click.echo(json.dumps(output, indent=2))
        router.close()
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command("reset-provider-usage")
@click.argument("provider_name")
@click.pass_context
def reset_provider_usage(ctx: click.Context, provider_name: str) -> None:
    """Reset all usage tracking for a specific provider."""
    path = ctx.obj["config_path"]
    try:
        router = BijouxRouter.from_yaml(path)
        router.reset_provider_usage(provider_name)
        click.echo(f"Usage reset for provider: {provider_name}")
        router.close()
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@cli.command("reload-config")
@click.pass_context
def reload_config_cmd(ctx: click.Context) -> None:
    """Reload configuration from YAML file."""
    path = ctx.obj["config_path"]
    try:
        router = BijouxRouter.from_yaml(path)
        router.reload_config(path)
        click.echo("Configuration reloaded successfully.")
        router.close()
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
