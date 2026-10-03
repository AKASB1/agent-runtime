"""OpenAI-compatible chat-completions adapter with dialect profiles (``openai``, ``deepseek``).

What differs between services lives in a :class:`Dialect`: usage field names, whether reasoning
text is returned and must be passed back inside a tool loop, the structured-output request field,
and extra request fields. The adapter raises only ``ProviderError``; credentials come only from its
configuration object and never appear in an error message.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from agent_runtime.canon import canonical
from agent_runtime.models.accounting import estimate_tokens
from agent_runtime.models.types import Message, ModelRequest, ModelResponse, ProviderError, ToolCall, Usage


@dataclass(frozen=True)
class Dialect:
    name: str
    #: usage: "openai" = prompt_tokens + prompt_tokens_details.cached_tokens;
    #: "deepseek" = prompt_cache_hit_tokens + prompt_cache_miss_tokens
    usage_style: str
    #: reasoning text field in responses (None if the service returns none)
    reasoning_field: str | None
    #: pass reasoning back on assistant messages when the request carries tools
    pass_back_reasoning_with_tools: bool
    #: structured output: "json_schema" or "json_object"
    response_format: str
    extra_body: dict[str, Any] = field(default_factory=dict)


DIALECTS: dict[str, Dialect] = {
    "openai": Dialect("openai", "openai", None, False, "json_schema"),
    "deepseek": Dialect("deepseek", "deepseek", "reasoning_content", True, "json_object"),
}

_FINISH = {"stop": "stop", "tool_calls": "tool_calls", "length": "length", "content_filter": "stop"}


def get_dialect(name: str) -> Dialect:
    try:
        return DIALECTS[name]
    except KeyError:
        raise ValueError(f"unknown dialect {name!r} (known: {', '.join(DIALECTS)})") from None


@dataclass
class OpenAIChatConfig:
    base_url: str
    api_key: str = field(default="", repr=False)
    dialect: str = "openai"
    extra_body: dict[str, Any] = field(default_factory=dict)
    trust_env: bool = False

    def __repr__(self) -> str:  # never show the key
        return f"OpenAIChatConfig(base_url={self.base_url!r}, dialect={self.dialect!r})"


def build_body(
    request: ModelRequest, model_id: str, dialect: Dialect, extra: dict[str, Any] | None = None
) -> dict:
    with_tools = bool(request.tools)
    msgs: list[dict[str, Any]] = []
    for m in request.messages:
        msgs.append(_wire_message(m, dialect, with_tools))
    body: dict[str, Any] = {"model": model_id, "messages": msgs, "max_tokens": request.max_output_tokens}
    if with_tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
            }
            for t in request.tools
        ]
    if request.response_schema is not None:
        if dialect.response_format == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": request.response_schema},
            }
        else:
            body["response_format"] = {"type": "json_object"}
    body.update(dialect.extra_body)
    if extra:
        body.update(extra)
    return body


def _wire_message(m: Message, dialect: Dialect, with_tools: bool) -> dict[str, Any]:
    if m.role == "tool":
        return {"role": "tool", "tool_call_id": m.tool_call_id or "", "content": m.content}
    if m.role == "assistant":
        d: dict[str, Any] = {
            "role": "assistant",
            "content": m.content if (m.content or not m.tool_calls) else None,
        }
        if m.tool_calls:
            d["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in m.tool_calls
            ]
        if (
            dialect.reasoning_field
            and dialect.pass_back_reasoning_with_tools
            and with_tools
            and m.reasoning is not None
        ):
            d[dialect.reasoning_field] = m.reasoning
        return d
    return {"role": m.role, "content": m.content}


def parse_usage(usage: dict[str, Any] | None, dialect: Dialect) -> Usage | None:
    if not isinstance(usage, dict):
        return None
    out = int(usage.get("completion_tokens") or 0)
    prompt = int(usage.get("prompt_tokens") or 0)
    if dialect.usage_style == "deepseek":
        details = usage.get("prompt_tokens_details") or {}
        hit = usage.get("prompt_cache_hit_tokens", details.get("prompt_cache_hit_tokens"))
        miss = usage.get("prompt_cache_miss_tokens", details.get("prompt_cache_miss_tokens"))
        hit = int(hit or 0)
        miss = int(miss) if miss is not None else max(prompt - hit, 0)
        return Usage(input_tokens=miss, output_tokens=out, cache_read_tokens=hit)
    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens") or 0)
    return Usage(input_tokens=max(prompt - cached, 0), output_tokens=out, cache_read_tokens=cached)


def parse_response(data: Any, request: ModelRequest, dialect: Dialect) -> ModelResponse:
    try:
        choice = data["choices"][0]
        msg = choice["message"]
    except (KeyError, IndexError, TypeError):
        raise ProviderError("server_error", "malformed response body: no choices[0].message") from None
    reason = choice.get("finish_reason") or "stop"
    if reason == "insufficient_system_resource":
        raise ProviderError("server_error", "provider reported insufficient_system_resource")
    calls: list[ToolCall] = []
    for c in msg.get("tool_calls") or []:
        fn = c.get("function") or {}
        raw = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            if not isinstance(args, dict):
                args = {"_invalid_json": raw}
        except (json.JSONDecodeError, TypeError, ValueError):
            args = {"_invalid_json": raw}
        calls.append(ToolCall(id=str(c.get("id") or ""), name=str(fn.get("name") or ""), arguments=args))
    text = msg.get("content") or ""
    usage = parse_usage(data.get("usage"), dialect)
    if usage is None:
        usage = Usage(
            input_tokens=estimate_tokens(
                canonical([m.model_dump(exclude_none=True) for m in request.messages])
            ),
            output_tokens=max(
                1, estimate_tokens(text + (canonical([c.model_dump() for c in calls]) if calls else ""))
            ),
            estimated=True,
        )
    reasoning = msg.get(dialect.reasoning_field) if dialect.reasoning_field else None
    return ModelResponse(
        text=text,
        tool_calls=calls,
        finish_reason=_FINISH.get(reason, "stop") if not calls else "tool_calls",
        usage=usage,
        reasoning=reasoning,
    )


def map_status(status: int, body_text: str, headers: httpx.Headers) -> ProviderError:
    snippet = body_text[:200]
    if status == 429:
        retry_ms = None
        if "retry-after-ms" in headers:
            try:
                retry_ms = int(float(headers["retry-after-ms"]))
            except ValueError:
                retry_ms = None
        elif "retry-after" in headers:
            try:
                retry_ms = int(float(headers["retry-after"]) * 1000)
            except ValueError:
                retry_ms = None
        return ProviderError("rate_limited", "HTTP 429", status=429, retry_after_ms=retry_ms)
    if status in (401, 402, 403):
        return ProviderError("auth", f"HTTP {status}", status=status)
    if status == 408:
        return ProviderError("timeout", "HTTP 408", status=status)
    if status >= 500:
        return ProviderError("server_error", f"HTTP {status}", status=status)
    lowered = snippet.lower()
    if "context_length" in lowered or "maximum context length" in lowered or "context length" in lowered:
        return ProviderError("context_length", f"HTTP {status}: context length exceeded", status=status)
    return ProviderError("bad_request", f"HTTP {status}", status=status)


class OpenAIChatAdapter:
    """``POST {base_url}/chat/completions`` with ``httpx``; ``transport`` lets tests run in process."""

    def __init__(
        self, config: OpenAIChatConfig, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.config = config
        self.dialect = get_dialect(config.dialect)
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url.rstrip("/"),
                transport=self._transport,
                trust_env=self.config.trust_env,
                follow_redirects=False,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def complete(self, request: ModelRequest, spec: Any) -> ModelResponse:
        body = build_body(request, spec.id, self.dialect, self.config.extra_body)
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = "Bearer " + self.config.api_key
        try:
            r = await self._get_client().post(
                "/chat/completions", json=body, headers=headers, timeout=request.timeout_ms / 1000.0
            )
        except httpx.TimeoutException:
            raise ProviderError("timeout", "request timed out") from None
        except httpx.HTTPError as e:
            raise ProviderError("server_error", f"transport error: {type(e).__name__}") from None
        if r.status_code != 200:
            raise map_status(r.status_code, r.text, r.headers)
        try:
            data = r.json()
        except (json.JSONDecodeError, ValueError):
            raise ProviderError("server_error", "malformed response body: not JSON") from None
        resp = parse_response(data, request, self.dialect)
        return resp.model_copy(update={"model": spec.id, "provider": spec.provider})
