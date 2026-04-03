"""Token estimation utilities for pre-dispatch quota reservation."""

from __future__ import annotations

from bijoux_router.models.request_response import LLMRequest, TokenEstimate


# Average chars-per-token ratio for common models (conservative estimate).
_CHARS_PER_TOKEN = 4.0


def estimate_prompt_tokens(request: LLMRequest) -> int:
    """Estimate the number of prompt tokens in a request.

    Uses a simple character-based heuristic. For production accuracy,
    tiktoken can be integrated per-model, but this provides a safe
    conservative estimate for quota reservation purposes.
    """
    total_chars = 0
    for msg in request.effective_messages():
        total_chars += len(msg.content)
        # Role and formatting overhead: ~4 tokens per message
        total_chars += 16
    if total_chars == 0:
        return 0
    return max(1, int(total_chars / _CHARS_PER_TOKEN))


def estimate_completion_tokens(request: LLMRequest) -> int:
    """Estimate completion tokens based on max_tokens or a default."""
    if request.max_tokens:
        # Reserve for the full requested max (conservative)
        return request.max_tokens
    # Default assumption: 256 tokens if not specified
    return 256


def estimate_tokens(request: LLMRequest) -> TokenEstimate:
    """Produce a full token estimate for a request."""
    prompt = estimate_prompt_tokens(request)
    completion = estimate_completion_tokens(request)
    return TokenEstimate(
        estimated_prompt_tokens=prompt,
        estimated_completion_tokens=completion,
        estimated_total=prompt + completion,
    )
