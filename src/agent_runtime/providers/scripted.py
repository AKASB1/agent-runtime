"""The scripted model: plays a fixed list of responses (or provider errors) in order.

A script item is a ``ModelResponse``, a ``ProviderError``, a dict (``{"text": ..., "tool_calls": [...]}``,
or ``{"error": {"kind": ...}}``), or a callable ``(request) -> item``. Token counts that the item
does not give are estimated (bytes / 4) and flagged ``estimated``. It stands in for a language model
in tests, the demo, and the worked example; it says nothing about any model.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent_runtime.canon import canonical
from agent_runtime.models.accounting import estimate_tokens
from agent_runtime.models.types import ModelRequest, ModelResponse, ProviderError, ToolCall, Usage

ScriptItem = ModelResponse | ProviderError | dict[str, Any] | Callable[[ModelRequest], Any]


class ScriptExhausted(Exception):
    pass


def _estimate_input(request: ModelRequest) -> int:
    return estimate_tokens(canonical([m.model_dump(exclude_none=True) for m in request.messages]))


def response_from_item(item: Any, request: ModelRequest, spec: Any) -> ModelResponse:
    if isinstance(item, ProviderError):
        raise item
    if isinstance(item, ModelResponse):
        resp = item
    else:
        d = dict(item)
        if "error" in d:
            raise ProviderError.from_dict(d["error"])
        calls = [
            c
            if isinstance(c, ToolCall)
            else ToolCall(id=c["id"], name=c["name"], arguments=c.get("arguments", {}))
            for c in d.get("tool_calls", [])
        ]
        text = d.get("text", "")
        if not isinstance(text, str):
            text = json.dumps(text)
        usage = d.get("usage")
        if usage is None:
            out = estimate_tokens(text + (canonical([c.model_dump() for c in calls]) if calls else ""))
            usage = Usage(input_tokens=_estimate_input(request), output_tokens=max(out, 1), estimated=True)
        elif isinstance(usage, dict):
            usage = Usage(**usage)
        resp = ModelResponse(
            text=text,
            tool_calls=calls,
            finish_reason=d.get("finish_reason", "tool_calls" if calls else "stop"),
            usage=usage,
            reasoning=d.get("reasoning"),
        )
    return resp.model_copy(update={"model": spec.id, "provider": spec.provider})


class ScriptedModel:
    """Plays ``script`` in order; ``latency_ms`` (int or list) passes on the run's clock."""

    def __init__(
        self, script: list[ScriptItem], *, clock: Any = None, latency_ms: int | list[int] = 0
    ) -> None:
        self._script = list(script)
        self._i = 0
        self.clock = clock
        self.latency_ms = latency_ms
        self.requests: list[ModelRequest] = []

    @classmethod
    def from_file(cls, path: str | Path, **kw: Any) -> ScriptedModel:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["steps"] if isinstance(data, dict) else data, **kw)

    @property
    def remaining(self) -> int:
        return len(self._script) - self._i

    async def complete(self, request: ModelRequest, spec: Any) -> ModelResponse:
        if self._i >= len(self._script):
            raise ProviderError("bad_request", "scripted model: script exhausted")
        item = self._script[self._i]
        idx = self._i
        self._i += 1
        self.requests.append(request)
        if self.clock is not None:
            lat = (
                self.latency_ms[idx % len(self.latency_ms)]
                if isinstance(self.latency_ms, list)
                else self.latency_ms
            )
            if lat:
                await self.clock.sleep(lat)
        if callable(item) and not isinstance(item, ModelResponse | ProviderError | dict):
            item = item(request)
        return response_from_item(item, request, spec)


class RoleScriptedModel:
    """Plays one script per request role (``meta.role``), each in order: for workflows whose calls
    have different roles (the worked example's ``formulate`` and ``report``)."""

    def __init__(
        self, scripts: dict[str, list[ScriptItem]], *, clock: Any = None, latency_ms: int = 0
    ) -> None:
        self.models = {
            role: ScriptedModel(items, clock=clock, latency_ms=latency_ms) for role, items in scripts.items()
        }

    async def complete(self, request: ModelRequest, spec: Any) -> ModelResponse:
        role = request.meta.get("role", "")
        if role not in self.models:
            raise ProviderError("bad_request", f"scripted model: no script for role {role!r}")
        return await self.models[role].complete(request, spec)
