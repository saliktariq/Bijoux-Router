"""Core data models for Bijoux Router — requests, responses, and token usage."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Message / chat models
# ---------------------------------------------------------------------------

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str
    name: str | None = None


# ---------------------------------------------------------------------------
# LLM Request
# ---------------------------------------------------------------------------

class LLMRequest(BaseModel):
    """Normalized LLM request understood by every provider adapter."""

    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timeout_override: float | None = None
    tenant: str | None = None
    tags: list[str] = Field(default_factory=list)

    def effective_messages(self) -> list[ChatMessage]:
        """Return messages list, synthesizing from prompt if needed."""
        if self.messages:
            return list(self.messages)
        if self.prompt:
            return [ChatMessage(role=MessageRole.USER, content=self.prompt)]
        return []


# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------

class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_counts(cls, prompt: int = 0, completion: int = 0) -> TokenUsage:
        return cls(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion)


class TokenEstimate(BaseModel):
    """Pre-dispatch estimate so quota can be reserved."""
    estimated_prompt_tokens: int = 0
    estimated_completion_tokens: int = 0
    estimated_total: int = 0


# ---------------------------------------------------------------------------
# LLM Response
# ---------------------------------------------------------------------------

class FinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"
    ERROR = "error"
    UNKNOWN = "unknown"


class FailoverAttempt(BaseModel):
    provider_name: str
    error_category: str
    error_message: str
    latency_ms: float = 0.0


class LLMResponse(BaseModel):
    """Normalized LLM response returned to every caller."""

    request_id: str
    content: str = ""
    provider_name: str = ""
    model: str = ""
    usage: TokenUsage = Field(default_factory=TokenUsage)
    finish_reason: FinishReason = FinishReason.UNKNOWN
    latency_ms: float = 0.0
    failover_attempts: list[FailoverAttempt] = Field(default_factory=list)
    raw_response: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
