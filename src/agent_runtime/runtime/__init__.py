"""The run engine: model and tool pipelines, budgets, cancellation, traces, replay.

``run_once`` is the scaffold's single-call helper, kept on the new model interface.
"""

from __future__ import annotations

from typing import Any

from agent_runtime.models import Message, ModelAdapter, ModelRequest, ModelResponse, ModelSpec

_DEFAULT_SPEC = ModelSpec(id="default", provider="local", wire="scripted")


async def run_once(model: ModelAdapter, prompt: str, spec: Any = None) -> ModelResponse:
    if not prompt:
        raise ValueError("prompt must not be empty")
    spec = spec or _DEFAULT_SPEC
    request = ModelRequest(model=spec.target, messages=[Message(role="user", content=prompt)])
    return await model.complete(request, spec)
