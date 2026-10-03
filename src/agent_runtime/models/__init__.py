"""Provider-neutral model interface: typed requests and responses, usage, the closed error
taxonomy, the model registry, and the accounting rule (``docs/contracts.md``)."""

from agent_runtime.models.accounting import cost_ucu, estimate_tokens
from agent_runtime.models.registry import ModelRegistry, ModelSpec
from agent_runtime.models.types import (
    ERROR_KINDS,
    FAILURE_REASONS,
    RUN_STATES,
    Message,
    ModelAdapter,
    ModelRequest,
    ModelResponse,
    ProviderError,
    ToolCall,
    ToolSpec,
    Usage,
)

__all__ = [
    "ERROR_KINDS",
    "FAILURE_REASONS",
    "RUN_STATES",
    "Message",
    "ModelAdapter",
    "ModelRegistry",
    "ModelRequest",
    "ModelResponse",
    "ModelSpec",
    "ProviderError",
    "ToolCall",
    "ToolSpec",
    "Usage",
    "cost_ucu",
    "estimate_tokens",
]
