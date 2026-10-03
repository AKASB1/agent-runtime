import json

import pytest

from agent_runtime.models import ModelRegistry, ModelSpec, ProviderError, Usage, cost_ucu, estimate_tokens
from agent_runtime.models.structured import StructuredOutputError, parse_structured

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "n": {"type": "integer"}},
    "required": ["answer", "n"],
    "additionalProperties": False,
}

# Every malformed shape the repair loop must handle (check 5); each is rejected with a message.
MALFORMED_FIXTURES = [
    "",
    "   ",
    "not json at all",
    '{"answer": "x", "n": 1',  # truncated
    '{"answer": "x", "n": 1} trailing words',
    '{"answer": "x"}',  # missing required
    '{"answer": "x", "n": "one"}',  # wrong type
    '{"answer": "x", "n": 1, "extra": true}',  # additional property
    "null",
    '[{"answer": "x", "n": 1}]',  # array instead of object
    "{'answer': 'x', 'n': 1}",  # single quotes
    '```json\n{"answer": "x"}\n```',  # fenced but invalid
]


def spec(**kw):
    base = dict(
        id="m", provider="p", wire="sim", price_in_ucu=1_000_000, price_out_ucu=4_000_000, context_tokens=1000
    )
    base.update(kw)
    return ModelSpec(**base)


def test_cost_known_answers_with_ceil_rule():
    s = spec()
    # 1250 in at 1.0 cu/M + 80 out at 4.0 cu/M = 1250 + 320 ucu (the example span of the contracts)
    assert cost_ucu(Usage(input_tokens=1250, output_tokens=80), s) == 1570
    small = spec(price_in_ucu=150_000, price_out_ucu=600_000)
    # 500*0.15 + 80*0.60 = 75 + 48 = 123 ucu exactly
    assert cost_ucu(Usage(input_tokens=500, output_tokens=80), small) == 123
    # 3 tokens at 0.15 cu/M = 0.45 ucu -> ceil -> 1
    assert cost_ucu(Usage(input_tokens=3), small) == 1
    assert cost_ucu(Usage(), small) == 0
    # cache terms default to the input price
    assert cost_ucu(Usage(input_tokens=100, cache_read_tokens=400), small) == 75
    cached = spec(
        price_in_ucu=150_000,
        price_out_ucu=600_000,
        price_cache_read_ucu=15_000,
        price_cache_write_ucu=187_500,
    )
    # 100*0.15 + 400*0.015 + 8*0.1875 + 10*0.6 = 15 + 6 + 1.5 + 6 = 28.5 -> 29
    assert (
        cost_ucu(
            Usage(input_tokens=100, cache_read_tokens=400, cache_write_tokens=8, output_tokens=10), cached
        )
        == 29
    )
    assert cost_ucu(Usage(input_tokens=10**6), spec(priced=False)) == 0


def test_estimator_rounds_up_bytes_over_four():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abc") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens("é") == 1  # 2 bytes


def test_registry_and_spec_validation():
    r = ModelRegistry([spec(id="a"), spec(id="b", provider="q")])
    assert r.targets() == ["p/a", "q/b"]
    assert r.get("q/b").price_cache_read_ucu == 1_000_000
    with pytest.raises(ValueError):
        r.add(spec(id="a"))
    with pytest.raises(KeyError):
        r.get("nope")
    with pytest.raises(ValueError, match="unknown dialect"):
        ModelSpec(id="x", provider="y", wire="openai_chat", dialect="klingon")
    assert ModelSpec(id="x", provider="y", wire="openai_chat").dialect == "openai"


def test_provider_error_taxonomy_is_closed():
    assert ProviderError("timeout").retryable
    assert ProviderError("rate_limited", retry_after_ms=1000).retry_after_ms == 1000
    assert not ProviderError("bad_request").retryable
    assert not ProviderError("auth").retryable
    with pytest.raises(ValueError):
        ProviderError("weird")
    e = ProviderError.from_dict(ProviderError("server_error", "x", status=503).to_dict())
    assert (e.kind, e.status, e.retryable) == ("server_error", 503, True)


def test_structured_output_valid_and_fenced():
    assert parse_structured('{"answer": "x", "n": 1}', SCHEMA) == {"answer": "x", "n": 1}
    assert parse_structured('```json\n{"answer": "x", "n": 2}\n```', SCHEMA)["n"] == 2


@pytest.mark.parametrize("text", MALFORMED_FIXTURES)
def test_structured_output_malformed_fixtures_raise_typed_error(text):
    with pytest.raises(StructuredOutputError) as ei:
        parse_structured(text, SCHEMA)
    assert str(ei.value)
    json.dumps(str(ei.value))  # the message is plain text that can go back to the model
