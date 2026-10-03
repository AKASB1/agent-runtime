"""Owner's opt-in live smoke test of the OpenAI-compatible adapter.

Reads exactly three environment variables, by name: AGENT_RUNTIME_LIVE_BASE_URL,
AGENT_RUNTIME_LIVE_API_KEY, AGENT_RUNTIME_LIVE_MODELS (comma-separated entries ``id``,
``id:price_in:price_out:context``, or ``id:price_in:price_out:context:price_cache_read``; prices per
million tokens in the endpoint's own currency, treated as cu). For each model it makes three small
calls (plain, a tool-call round trip, structured output) and prints usage (cached input where
reported), cost, and latency. The key is never printed or logged.

    python scripts/live_smoke.py [--dialect openai|deepseek]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

if __package__ in (None, ""):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from agent_runtime.clock import SystemClock
from agent_runtime.models import Message, ModelRequest, ModelSpec, ProviderError, ToolSpec, cost_ucu
from agent_runtime.models.structured import StructuredOutputError, parse_structured
from agent_runtime.providers.openai_chat import OpenAIChatAdapter, OpenAIChatConfig

VARS = ("AGENT_RUNTIME_LIVE_BASE_URL", "AGENT_RUNTIME_LIVE_API_KEY", "AGENT_RUNTIME_LIVE_MODELS")

WEATHER = ToolSpec(
    name="get_temperature",
    description="Return the temperature in Celsius for a city.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"capital": {"type": "string"}, "country": {"type": "string"}},
    "required": ["capital", "country"],
}


def parse_models(value: str, dialect: str) -> list[ModelSpec]:
    specs = []
    for entry in [e.strip() for e in value.split(",") if e.strip()]:
        parts = entry.split(":")
        mid = parts[0]
        kw: dict[str, Any] = {"id": mid, "provider": "live", "wire": "openai_chat", "dialect": dialect}
        if len(parts) >= 4:
            kw["price_in_ucu"] = round(float(parts[1]) * 1_000_000)
            kw["price_out_ucu"] = round(float(parts[2]) * 1_000_000)
            kw["context_tokens"] = int(parts[3])
            if len(parts) >= 5:
                kw["price_cache_read_ucu"] = round(float(parts[4]) * 1_000_000)
        elif len(parts) == 1:
            kw["priced"] = False
        else:
            raise ValueError(
                f"bad model entry {entry!r}: expected id or id:price_in:price_out:context[:price_cache_read]"
            )
        specs.append(ModelSpec(**kw))
    return specs


def default_dialect(base_url: str) -> str:
    return "deepseek" if (urlparse(base_url).hostname or "") == "api.deepseek.com" else "openai"


async def smoke_model(
    adapter: OpenAIChatAdapter, spec: ModelSpec, clock: SystemClock, out: list[str]
) -> dict:
    totals = {"calls": 0, "input": 0, "cached": 0, "output": 0, "cost_ucu": 0, "errors": {}}

    async def call(name: str, req: ModelRequest):
        t0 = clock.now_ms()
        try:
            r = await adapter.complete(req, spec)
        except ProviderError as e:
            totals["errors"][e.kind] = totals["errors"].get(e.kind, 0) + 1
            out.append(f"  {name}: error {e.kind} (status {e.status})")
            return None
        lat = clock.now_ms() - t0
        c = cost_ucu(r.usage, spec)
        totals["calls"] += 1
        totals["input"] += r.usage.input_tokens
        totals["cached"] += r.usage.cache_read_tokens
        totals["output"] += r.usage.output_tokens
        totals["cost_ucu"] += c
        est = " (estimated)" if r.usage.estimated else ""
        price = f"cost {c / 1_000_000:.6f}" if spec.priced else "unpriced"
        out.append(
            f"  {name}: finish={r.finish_reason} in={r.usage.input_tokens} cached={r.usage.cache_read_tokens} "
            f"out={r.usage.output_tokens}{est} {price} latency={lat / 1000:.3f}s"
        )
        return r

    sys_msg = Message(role="system", content="You are a terse assistant.")
    await call(
        "plain",
        ModelRequest(
            model=spec.target,
            messages=[sys_msg, Message(role="user", content="Say OK.")],
            max_output_tokens=64,
        ),
    )
    msgs = [sys_msg, Message(role="user", content="What is the temperature in Paris? Use the tool.")]
    r = await call(
        "tool_call", ModelRequest(model=spec.target, messages=msgs, tools=[WEATHER], max_output_tokens=256)
    )
    if r is not None and r.tool_calls:
        tc = r.tool_calls[0]
        msgs = msgs + [
            Message(role="assistant", content=r.text, tool_calls=r.tool_calls, reasoning=r.reasoning),
            Message(role="tool", tool_call_id=tc.id, content=json.dumps({"celsius": 18})),
        ]
        await call(
            "tool_result",
            ModelRequest(model=spec.target, messages=msgs, tools=[WEATHER], max_output_tokens=128),
        )
    elif r is not None:
        out.append("  tool_call: the model answered without calling the tool")
    sr = await call(
        "structured",
        ModelRequest(
            model=spec.target,
            messages=[
                sys_msg,
                Message(role="user", content='Reply with JSON {"capital": ..., "country": ...} for France.'),
            ],
            response_schema=ANSWER_SCHEMA,
            max_output_tokens=128,
        ),
    )
    if sr is not None:
        try:
            parse_structured(sr.text, ANSWER_SCHEMA)
            out.append("  structured: valid against the schema")
        except StructuredOutputError as e:
            out.append(f"  structured: invalid ({e})")
    return totals


async def run(
    environ: Mapping[str, str], dialect: str | None, transport: Any = None
) -> tuple[int, list[str]]:
    out: list[str] = []
    if not all(environ.get(v) for v in VARS):
        out.append("not run: no endpoint configured")
        return 0, out
    base_url = environ["AGENT_RUNTIME_LIVE_BASE_URL"]
    dialect = dialect or default_dialect(base_url)
    specs = parse_models(environ["AGENT_RUNTIME_LIVE_MODELS"], dialect)
    adapter = OpenAIChatAdapter(
        OpenAIChatConfig(
            base_url=base_url,
            api_key=environ["AGENT_RUNTIME_LIVE_API_KEY"],
            dialect=dialect,
            trust_env=urlparse(base_url).hostname not in ("127.0.0.1", "localhost"),
        ),
        transport=transport,
    )
    clock = SystemClock()
    out.append(f"endpoint host: {urlparse(base_url).hostname}  dialect: {dialect}")
    try:
        for spec in specs:
            out.append(f"model {spec.id}:")
            t = await smoke_model(adapter, spec, clock, out)
            out.append(
                f"  total: calls={t['calls']} in={t['input']} cached={t['cached']} out={t['output']} "
                f"cost={t['cost_ucu'] / 1_000_000:.6f} errors={t['errors']}"
            )
    finally:
        await adapter.aclose()
    return 0, out


def main(
    argv: list[str] | None = None, environ: Mapping[str, str] | None = None, transport: Any = None
) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dialect", choices=["openai", "deepseek"], default=None)
    args = p.parse_args(argv)
    env = {v: (environ if environ is not None else os.environ).get(v, "") for v in VARS}
    code, lines = asyncio.run(run(env, args.dialect, transport))
    for line in lines:
        print(line)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
