"""Subagents: ``delegate(agent, task)`` runs a child with a fresh context (the spec's system prompt and
the task, nothing of the parent's history), a tool allowlist, a target, ``max_steps``, and a share of
the parent's budget. The child's calls are billed to the parent's run; its spans nest in the parent's
trace under a ``subagent`` span; what returns is the final text (cut to 400 tokens), the status, and
the step count."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.tools import Tool

RESULT_CAP_TOKENS = 400
CAP_MARKER = " [... cut to 400 tokens]"


class AgentSpec(BaseModel):
    name: str
    description: str = ""
    system_prompt: str = "You are a focused helper agent. Use your tools and answer briefly."
    tools: list[str] = Field(default_factory=list)
    target: str | None = None  # a fixed target; None = the parent's router
    fallback: list[str] = Field(default_factory=list)  # used when the fixed target is not offered
    max_steps: int = 8
    max_cost_ucu: int | None = None
    max_model_calls: int | None = None


class DelegateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: str
    task: str = Field(min_length=1)


async def _never(value: Any) -> Any:  # the engine runs delegate itself
    raise RuntimeError("delegate is executed by the run engine")


def delegate_tool(catalog: dict[str, AgentSpec]) -> Tool:
    names = ", ".join(f"{n} ({s.description})" if s.description else n for n, s in sorted(catalog.items()))
    return Tool(
        name="delegate",
        handler=_never,
        input_model=DelegateIn,
        description=f"Delegate a self-contained task to a helper agent and get its final answer. Agents: {names}.",
        side_effect="write",
        capability="none",
        timeout_ms=10**9,
        meta={"runtime": "delegate"},
    )


def cap_text(text: str, tokens: int = RESULT_CAP_TOKENS) -> str:
    """Cut to ``tokens`` by the documented estimator (4 bytes per token) with a marker."""
    b = text.encode("utf-8")
    if len(b) <= tokens * 4:
        return text
    return b[: tokens * 4].decode("utf-8", errors="ignore") + CAP_MARKER
