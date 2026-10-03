"""``scripts/live_smoke.py`` against the stand-in only (it never runs live in the test suite)."""

import importlib.util
import pathlib

import httpx

from agent_runtime.providers.standin import StandIn

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("live_smoke", ROOT / "scripts" / "live_smoke.py")
live_smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live_smoke)

FAKE_KEY = "sk-canary-0123456789abcdef"


def test_not_configured_says_not_run(capsys):
    assert live_smoke.main([], environ={}) == 0
    assert "not run: no endpoint configured" in capsys.readouterr().out


def test_smoke_against_standin_never_prints_the_key(capsys):
    standin = StandIn("deepseek", expected_key=FAKE_KEY)
    standin.enqueue(
        {"content": "OK", "cached": 4, "prompt_tokens": 20},
        {
            "tool_calls": [{"id": "t1", "name": "get_temperature", "arguments": {"city": "Paris"}}],
            "reasoning": "r",
        },
        {"content": "18 C"},
        {"content": '{"capital": "Paris", "country": "France"}'},
    )
    env = {
        "AGENT_RUNTIME_LIVE_BASE_URL": "http://standin/v1",
        "AGENT_RUNTIME_LIVE_API_KEY": FAKE_KEY,
        "AGENT_RUNTIME_LIVE_MODELS": "deepseek-flash:0.15:0.6:65536:0.003,other",
    }
    code = live_smoke.main(
        ["--dialect", "deepseek"], environ=env, transport=httpx.ASGITransport(app=standin.app)
    )
    out = capsys.readouterr().out
    assert code == 0
    assert FAKE_KEY not in out
    assert "cached=4" in out and "structured: valid" in out and "unpriced" in out
    # the reasoning was passed back inside the tool loop (deepseek rule)
    assert standin.seen[2]["messages"][2]["reasoning_content"] == "r"


def test_model_entry_parsing_and_dialect_default():
    specs = live_smoke.parse_models("a,b:1:2:1000,c:1:2:1000:0.5", "openai")
    assert [s.priced for s in specs] == [False, True, True]
    assert specs[1].price_in_ucu == 1_000_000 and specs[2].price_cache_read_ucu == 500_000
    assert live_smoke.default_dialect("https://api.deepseek.com/v1") == "deepseek"
    assert live_smoke.default_dialect("http://127.0.0.1:11434/v1") == "openai"
