"""Check 7 (canary): a workflow through the OpenAI-compatible adapter with a fake key against the
stand-in; the key must appear in no trace, result, error message, or exception text. The API part
of the canary is in ``test_api.py``."""

import asyncio
import io
import logging

import httpx

from agent_runtime.clock import SystemClock
from agent_runtime.mcp.twin import native_lookup_tool
from agent_runtime.models import ModelRegistry, ModelSpec
from agent_runtime.providers.openai_chat import OpenAIChatAdapter, OpenAIChatConfig
from agent_runtime.providers.standin import StandIn
from agent_runtime.runtime.engine import Env
from agent_runtime.runtime.runner import RunSpec, execute
from agent_runtime.store.memory import MemoryStore
from agent_runtime.tools import ToolRegistry

KEY = "sk-CANARY-7f3a9c2e5b8d1f4a"


def test_fake_key_appears_nowhere():
    standin = StandIn("deepseek", expected_key=KEY)
    standin.enqueue(
        {
            "tool_calls": [{"id": "t1", "name": "lookup", "arguments": {"key": "beta"}}],
            "reasoning": "thinking",
        },
        {"status": 503},
        {"content": "beta is 22", "cached": 3, "prompt_tokens": 30},
    )
    adapter = OpenAIChatAdapter(
        OpenAIChatConfig(base_url="http://standin/v1", api_key=KEY, dialect="deepseek"),
        transport=httpx.ASGITransport(app=standin.app),
    )
    bad_adapter = OpenAIChatAdapter(
        OpenAIChatConfig(base_url="http://standin/v1", api_key=KEY + "x", dialect="deepseek"),
        transport=httpx.ASGITransport(app=standin.app),
    )
    reg = ModelRegistry(
        [
            ModelSpec(
                id="flash",
                provider="live",
                wire="openai_chat",
                dialect="deepseek",
                price_in_ucu=150_000,
                price_out_ucu=600_000,
            ),
            ModelSpec(id="flash", provider="wrongkey", wire="openai_chat", dialect="deepseek"),
        ]
    )
    log = io.StringIO()
    handler = logging.StreamHandler(log)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.DEBUG)
    texts = []
    try:

        async def go():
            store = MemoryStore()
            env = Env(
                registry=reg,
                adapters={"live": adapter, "wrongkey": bad_adapter},
                tools=ToolRegistry([native_lookup_tool()]),
            )
            r1 = await execute(
                RunSpec(
                    run_id="r-1",
                    workflow="react",
                    input={"task": "beta?"},
                    default_target="live/flash",
                    clock="system",
                ),
                env,
                SystemClock(),
                store,
                store.create_session(),
            )
            r2 = await execute(
                RunSpec(
                    run_id="r-2",
                    workflow="chat",
                    input={"message": "hi"},
                    default_target="wrongkey/flash",
                    clock="system",
                ),
                env,
                SystemClock(),
                store,
                store.create_session(),
            )
            await adapter.aclose()
            await bad_adapter.aclose()
            return r1, r2, store

        r1, r2, store = asyncio.run(go())
        assert (r1.status, r1.text) == ("succeeded", "beta is 22")
        assert (r2.status, r2.failure_reason) == ("failed", "auth")
        for r in (r1, r2):
            texts.append(r.trace.jsonl())
            texts.append(str(r.to_dict()))
            texts.append(repr(r))
        texts.append(str(store._sessions))
        texts.append(repr(adapter.config))
    finally:
        logging.getLogger().removeHandler(handler)
    texts.append(log.getvalue())
    blob = "\n".join(texts)
    assert KEY not in blob
    assert "thinking" in blob  # the trace did capture the payloads it should
