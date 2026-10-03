"""The model interface. Field names are the contract (``docs/contracts.md`` section 2)."""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]
FinishReason = Literal["stop", "tool_calls", "length"]

#: The closed error taxonomy (failure table of the contracts).
ERROR_KINDS: tuple[str, ...] = (
    "timeout",
    "rate_limited",
    "server_error",
    "bad_request",
    "auth",
    "context_length",
    "invalid_output",
    "tool_error",
    "tool_timeout",
    "budget_exhausted",
    "cancelled",
    "permission_denied",
    "subagent_failed",
)
PROVIDER_ERROR_KINDS: tuple[str, ...] = (
    "timeout",
    "rate_limited",
    "server_error",
    "bad_request",
    "auth",
    "context_length",
)
RETRYABLE_KINDS: frozenset[str] = frozenset({"timeout", "rate_limited", "server_error"})

RUN_STATES: tuple[str, ...] = ("running", "succeeded", "failed", "cancelled", "budget_exhausted")
#: Closed list of ``failure_reason`` values of a failed run.
FAILURE_REASONS: tuple[str, ...] = (
    "provider_unavailable",
    "bad_request",
    "auth",
    "context_length",
    "invalid_output",
    "step_failed",
    "plan_invalid",
    "verifier_rejected",
)


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reasoning: str | None = None
    tool_call_id: str | None = None


class ToolSpec(BaseModel):
    """A tool as offered to a model: name, description, JSON Schema of the arguments."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class ModelRequest(BaseModel):
    model: str
    messages: list[Message]
    tools: list[ToolSpec] = Field(default_factory=list)
    response_schema: dict[str, Any] | None = None
    max_output_tokens: int = 512
    timeout_ms: int = 10_000
    meta: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    input_tokens: int = 0  # uncached input only
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    estimated: bool = False

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens


class ModelResponse(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: FinishReason = "stop"
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    provider: str = ""
    #: reasoning text returned by a dialect that has one (passed back where the dialect requires it)
    reasoning: str | None = None


class ProviderError(Exception):
    """The only exception type that crosses the adapter boundary."""

    def __init__(
        self,
        kind: str,
        message: str = "",
        *,
        status: int | None = None,
        retry_after_ms: int | None = None,
        retryable: bool | None = None,
    ) -> None:
        if kind not in ERROR_KINDS:
            raise ValueError(f"unknown error kind {kind!r}")
        super().__init__(f"{kind}: {message}" if message else kind)
        self.kind = kind
        self.message = message
        self.status = status
        self.retry_after_ms = retry_after_ms
        self.retryable = kind in RETRYABLE_KINDS if retryable is None else retryable

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "retryable": self.retryable,
            "status": self.status,
            "retry_after_ms": self.retry_after_ms,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ProviderError:
        return cls(
            d["kind"],
            d.get("message", ""),
            status=d.get("status"),
            retry_after_ms=d.get("retry_after_ms"),
            retryable=d.get("retryable"),
        )


class ModelAdapter(Protocol):
    """A wire format or a simulated provider. Raises only :class:`ProviderError`."""

    async def complete(self, request: ModelRequest, spec: Any) -> ModelResponse: ...
