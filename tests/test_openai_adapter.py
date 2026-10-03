"""Checks 7 and 16: the OpenAI-compatible adapter and its dialects against the stand-in."""

import asyncio

import httpx
import pytest

from agent_runtime.models import (
    Message,
    ModelRequest,
    ModelSpec,
    ProviderError,
    ToolCall,
    ToolSpec,
    Usage,
    cost_ucu,
)
from agent_runtime.providers.openai_chat import OpenAIChatAdapter, OpenAIChatConfig, get_dialect
from agent_runtime.providers.standin import StandIn, ThreadedServer

TOOL = ToolSpec(
    name="lookup",
    description="look up",
    parameters={"type": "object", "properties": {"q": {"type": "string"}}},
)


def make(dialect="openai", key="sk-test"):
    standin = StandIn(dialect)
    adapter = OpenAIChatAdapter(
        OpenAIChatConfig(base_url="http://standin/v1", api_key=key, dialect=dialect),
        transport=httpx.ASGITransport(app=standin.app),
    )
    spec = ModelSpec(
        id="echo",
        provider="standin",
        wire="openai_chat",
        dialect=dialect,
        price_in_ucu=1_000_000,
        price_out_ucu=2_000_000,
    )
    return standin, adapter, spec


def req(**kw):
    base = dict(
        model="standin/echo",
        messages=[Message(role="system", content="s"), Message(role="user", content="hi")],
    )
    base.update(kw)
    return ModelRequest(**base)


def run(coro):
    return asyncio.run(coro)


def test_request_body_has_exactly_the_documented_fields():
    standin, adapter, spec = make()
    run(adapter.complete(req(), spec))
    assert set(standin.seen[0]) == {"model", "messages", "max_tokens"}
    run(adapter.complete(req(tools=[TOOL], response_schema={"type": "object"}), spec))
    body = standin.seen[1]
    assert set(body) == {"model", "messages", "max_tokens", "tools", "response_format"}
    assert body["tools"][0] == {
        "type": "function",
        "function": {"name": "lookup", "description": "look up", "parameters": TOOL.parameters},
    }
    assert body["response_format"]["type"] == "json_schema"
    assert body["messages"] == [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]


def test_response_parsing_content_tool_calls_finish_reason_usage():
    standin, adapter, spec = make()
    r = run(adapter.complete(req(), spec))
    assert (r.text, r.finish_reason, r.provider, r.model) == ("echo: hi", "stop", "standin", "echo")
    assert not r.usage.estimated and r.usage.output_tokens == 5
    standin.enqueue(
        {
            "tool_calls": [{"id": "c1", "name": "lookup", "arguments": '{"q": "x"}'}],
            "prompt_tokens": 40,
            "completion_tokens": 7,
        }
    )
    r = run(adapter.complete(req(tools=[TOOL]), spec))
    assert r.finish_reason == "tool_calls"
    assert r.tool_calls == [ToolCall(id="c1", name="lookup", arguments={"q": "x"})]
    assert (r.usage.input_tokens, r.usage.output_tokens) == (40, 7)
    standin.enqueue({"tool_calls": [{"id": "c2", "name": "lookup", "arguments": "{not json"}]})
    r = run(adapter.complete(req(tools=[TOOL]), spec))
    assert r.tool_calls[0].arguments == {"_invalid_json": "{not json"}
    standin.enqueue({"content": "partial", "finish_reason": "length"})
    assert run(adapter.complete(req(), spec)).finish_reason == "length"


@pytest.mark.parametrize(
    "directive,kind,status",
    [
        ({"status": 400}, "bad_request", 400),
        (
            {"status": 400, "message": "This model's maximum context length is 4096 tokens"},
            "context_length",
            400,
        ),
        ({"status": 401}, "auth", 401),
        ({"status": 402}, "auth", 402),
        ({"status": 403}, "auth", 403),
        ({"status": 404}, "bad_request", 404),
        ({"status": 408}, "timeout", 408),
        ({"status": 422}, "bad_request", 422),
        ({"status": 429, "retry_after_ms": 1500}, "rate_limited", 429),
        ({"status": 500}, "server_error", 500),
        ({"status": 502}, "server_error", 502),
        ({"status": 503}, "server_error", 503),
        ({"malformed": "not_json"}, "server_error", None),
        ({"malformed": "no_choices"}, "server_error", None),
    ],
)
def test_error_statuses_and_malformed_bodies_map_to_the_taxonomy(directive, kind, status):
    standin, adapter, spec = make()
    standin.enqueue(directive)
    with pytest.raises(ProviderError) as ei:
        run(adapter.complete(req(), spec))
    assert ei.value.kind == kind
    assert ei.value.status == status
    if kind == "rate_limited":
        assert ei.value.retry_after_ms == 1500
    assert "sk-test" not in str(ei.value)


def test_timeout_maps_to_timeout_over_a_real_socket():
    # the in-process ASGI transport has no network timeouts, so this one uses the socket
    standin = StandIn("openai")
    standin.enqueue({"delay_ms": 1500})
    with ThreadedServer(standin.app) as srv:
        spec = ModelSpec(id="echo", provider="standin", wire="openai_chat")

        async def go():
            a = OpenAIChatAdapter(OpenAIChatConfig(base_url=srv.base_url + "/v1", api_key="k"))
            try:
                with pytest.raises(ProviderError) as ei:
                    await a.complete(req(timeout_ms=200), spec)
                return ei.value
            finally:
                await a.aclose()

        err = run(go())
    assert err.kind == "timeout" and err.retryable


def test_real_local_socket_round_trip_and_errors():
    standin = StandIn("openai", expected_key="sk-real")
    with ThreadedServer(standin.app) as srv:
        spec = ModelSpec(id="echo", provider="standin", wire="openai_chat")

        async def go():
            ok = OpenAIChatAdapter(OpenAIChatConfig(base_url=srv.base_url + "/v1", api_key="sk-real"))
            bad = OpenAIChatAdapter(OpenAIChatConfig(base_url=srv.base_url + "/v1", api_key="sk-wrong"))
            try:
                r = await ok.complete(req(), spec)
                with pytest.raises(ProviderError) as ei:
                    await bad.complete(req(), spec)
                return r, ei.value
            finally:
                await ok.aclose()
                await bad.aclose()

        r, err = run(go())
    assert r.text == "echo: hi"
    assert err.kind == "auth" and "sk-wrong" not in str(err)


# ---- check 16: dialects and usage -----------------------------------------------------------


@pytest.mark.parametrize("dialect", ["openai", "deepseek"])
def test_usage_with_and_without_cache_and_without_usage_block(dialect):
    standin, adapter, spec = make(dialect)
    standin.enqueue({"prompt_tokens": 100, "completion_tokens": 9, "cached": 60, "usage": "cached"})
    u = run(adapter.complete(req(), spec)).usage
    assert (u.input_tokens, u.cache_read_tokens, u.output_tokens, u.cache_write_tokens, u.estimated) == (
        40,
        60,
        9,
        0,
        False,
    )
    standin.enqueue({"prompt_tokens": 100, "completion_tokens": 9})
    u = run(adapter.complete(req(), spec)).usage
    assert (u.input_tokens, u.cache_read_tokens, u.estimated) == (100, 0, False)
    standin.enqueue({"usage": "none", "content": "abcdefgh"})
    u = run(adapter.complete(req(), spec)).usage
    assert u.estimated and u.output_tokens == 2 and u.input_tokens > 0


def test_deepseek_cache_fields_nested_under_prompt_tokens_details_are_read_too():
    from agent_runtime.providers.openai_chat import parse_usage

    u = parse_usage(
        {
            "prompt_tokens": 50,
            "completion_tokens": 3,
            "prompt_tokens_details": {"prompt_cache_hit_tokens": 20, "prompt_cache_miss_tokens": 30},
        },
        get_dialect("deepseek"),
    )
    assert (u.input_tokens, u.cache_read_tokens) == (30, 20)


def test_reasoning_text_is_parsed_and_passed_back_only_in_a_tool_loop_for_deepseek():
    standin, adapter, spec = make("deepseek")
    standin.enqueue(
        {"tool_calls": [{"id": "c1", "name": "lookup", "arguments": {"q": "x"}}], "reasoning": "think-1"}
    )
    r = run(adapter.complete(req(tools=[TOOL]), spec))
    assert r.reasoning == "think-1"
    history = [
        Message(role="system", content="s"),
        Message(role="user", content="hi"),
        Message(role="assistant", tool_calls=r.tool_calls, reasoning=r.reasoning),
        Message(role="tool", tool_call_id="c1", content='{"ok": true}'),
    ]
    run(adapter.complete(req(messages=history, tools=[TOOL]), spec))
    assistant = standin.seen[-1]["messages"][2]
    assert assistant["reasoning_content"] == "think-1"
    assert assistant["tool_calls"][0]["function"]["arguments"] == '{"q": "x"}'
    # without tools in the request the reasoning is dropped
    run(adapter.complete(req(messages=history[:3] + [Message(role="user", content="next")]), spec))
    assert "reasoning_content" not in standin.seen[-1]["messages"][2]
    assert standin.seen[-1]["response_format"] if "response_format" in standin.seen[-1] else True


def test_openai_dialect_never_sends_reasoning_and_uses_json_schema():
    standin, adapter, spec = make("openai")
    history = [
        Message(role="user", content="hi"),
        Message(role="assistant", content="", tool_calls=[ToolCall(id="c", name="lookup")], reasoning="r"),
        Message(role="tool", tool_call_id="c", content="{}"),
    ]
    run(adapter.complete(req(messages=history, tools=[TOOL], response_schema={"type": "object"}), spec))
    assert "reasoning_content" not in standin.seen[-1]["messages"][1]
    assert standin.seen[-1]["response_format"]["type"] == "json_schema"
    standin2, adapter2, spec2 = make("deepseek")
    run(adapter2.complete(req(response_schema={"type": "object"}), spec2))
    assert standin2.seen[-1]["response_format"] == {"type": "json_object"}


def test_cost_formula_with_cache_terms_hand_computed():
    spec = ModelSpec(
        id="m",
        provider="p",
        wire="openai_chat",
        price_in_ucu=270_000,
        price_out_ucu=1_100_000,
        price_cache_read_ucu=70_000,
    )
    # 333*0.27 + 1000*0.07 + 77*1.1 = 89.91 + 70 + 84.7 = 244.61 -> 245
    assert cost_ucu(Usage(input_tokens=333, cache_read_tokens=1000, output_tokens=77), spec) == 245


def test_unknown_dialect_is_a_configuration_error():
    with pytest.raises(ValueError, match="unknown dialect"):
        OpenAIChatAdapter(OpenAIChatConfig(base_url="http://x", dialect="nope"))
    with pytest.raises(ValueError):
        get_dialect("nope")
