"""Example usage of Bijoux Quota-Aware LLM Gateway."""

import asyncio

from bijoux_router import BijouxRouter, LLMRequest


async def main() -> None:
    # Initialize from YAML
    router = BijouxRouter.from_yaml("config/providers.yaml")

    # ── Simple prompt ─────────────────────────────────────────
    response = await router.process(
        LLMRequest(
            prompt="Explain the difference between TCP and UDP in two sentences.",
            max_tokens=200,
            temperature=0.3,
        )
    )
    print(f"Provider: {response.provider_name}")
    print(f"Model:    {response.model}")
    print(f"Content:  {response.content}")
    print(f"Tokens:   {response.usage.total_tokens}")
    print(f"Latency:  {response.latency_ms:.0f}ms")
    print(f"Failover: {len(response.failover_attempts)} attempts")
    print()

    # ── Model-specific request ────────────────────────────────
    response = await router.process(
        LLMRequest(
            prompt="Write a haiku about distributed systems.",
            model="gpt-4o-mini",
            max_tokens=100,
            temperature=0.7,
        )
    )
    print(f"Provider: {response.provider_name}")
    print(f"Model:    {response.model}")
    print(f"Content:  {response.content}")
    print()

    # ── Chat-style messages ───────────────────────────────────
    from bijoux_router import ChatMessage, MessageRole

    response = await router.process(
        LLMRequest(
            messages=[
                ChatMessage(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
                ChatMessage(role=MessageRole.USER, content="What is 2+2?"),
            ],
            max_tokens=50,
        )
    )
    print(f"Chat response: {response.content}")
    print()

    # ── Status inspection ─────────────────────────────────────
    import json
    print("Provider status:")
    print(json.dumps(router.get_provider_status(), indent=2, default=str))

    print("\nQuota status:")
    print(json.dumps(router.get_quota_status(), indent=2, default=str))

    router.close()


if __name__ == "__main__":
    asyncio.run(main())
