"""Stand-in provider: a small ASGI app that follows the documented chat-completions wire format of
the ``openai`` and ``deepseek`` dialects. It is a test double (and the demo's provider), not a model.

It records every request body it sees (never the Authorization header), and can be told, through a
queue of directives, to answer with an error status, a malformed body, a delay, a scripted message,
cached-input counts, reasoning text, or no usage block. Without directives it echoes the last user
message. It also serves a few plain-text pages for the ``http_get`` tool tests.

Run: ``python -m agent_runtime.providers.standin --port 18510 [--dialect deepseek]``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import threading
from collections import deque
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, RedirectResponse, Response
from starlette.routing import Route


class StandIn:
    def __init__(self, dialect: str = "openai", expected_key: str | None = None) -> None:
        if dialect not in ("openai", "deepseek"):
            raise ValueError(f"unknown dialect {dialect!r}")
        self.dialect = dialect
        self.expected_key = expected_key
        self.queue: deque[dict[str, Any]] = deque()
        self.seen: list[dict[str, Any]] = []
        self.app = Starlette(
            routes=[
                Route("/v1/chat/completions", self.chat, methods=["POST"]),
                Route("/v1/models", self.models, methods=["GET"]),
                Route("/_control/enqueue", self.enqueue_http, methods=["POST"]),
                Route("/_control/seen", self.seen_http, methods=["GET"]),
                Route("/_control/reset", self.reset_http, methods=["POST"]),
                Route("/text/{name}", self.text, methods=["GET"]),
                Route("/redirect", self.redirect, methods=["GET"]),
                Route("/big", self.big, methods=["GET"]),
            ]
        )

    # -- control -------------------------------------------------------------------------------
    def enqueue(self, *directives: dict[str, Any]) -> None:
        self.queue.extend(directives)

    def reset(self) -> None:
        self.queue.clear()
        self.seen.clear()

    async def enqueue_http(self, request: Request) -> Response:
        data = await request.json()
        self.enqueue(*(data if isinstance(data, list) else [data]))
        return JSONResponse({"queued": len(self.queue)})

    async def seen_http(self, request: Request) -> Response:
        return JSONResponse(self.seen)

    async def reset_http(self, request: Request) -> Response:
        self.reset()
        return JSONResponse({"ok": True})

    # -- pages for http_get ----------------------------------------------------------------------
    async def text(self, request: Request) -> Response:
        return PlainTextResponse(f"page {request.path_params['name']}\n")

    async def redirect(self, request: Request) -> Response:
        return RedirectResponse("/text/target", status_code=302)

    async def big(self, request: Request) -> Response:
        return PlainTextResponse("x" * 300_000)

    async def models(self, request: Request) -> Response:
        return JSONResponse({"object": "list", "data": [{"id": "echo", "object": "model"}]})

    # -- chat completions ----------------------------------------------------------------------
    async def chat(self, request: Request) -> Response:
        auth = request.headers.get("authorization", "")
        if self.expected_key is not None and auth != f"Bearer {self.expected_key}":
            return JSONResponse({"error": {"message": "invalid api key", "type": "auth"}}, status_code=401)
        raw = await request.body()
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            return JSONResponse({"error": {"message": "invalid JSON body"}}, status_code=400)
        self.seen.append(body)
        d = self.queue.popleft() if self.queue else {}
        if d.get("delay_ms"):
            await asyncio.sleep(d["delay_ms"] / 1000.0)
        if "status" in d:
            headers = {}
            if d.get("retry_after_ms") is not None:
                headers["retry-after-ms"] = str(d["retry_after_ms"])
            msg = d.get("message", f"stand-in error {d['status']}")
            return JSONResponse({"error": {"message": msg}}, status_code=d["status"], headers=headers)
        if d.get("malformed") == "not_json":
            return Response(b"<html>oops", media_type="text/html")
        if d.get("malformed") == "no_choices":
            return JSONResponse({"id": "x", "object": "chat.completion"})
        message: dict[str, Any] = {"role": "assistant"}
        if "tool_calls" in d:
            message["content"] = None
            message["tool_calls"] = [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {
                        "name": c["name"],
                        "arguments": c["arguments"]
                        if isinstance(c["arguments"], str)
                        else json.dumps(c["arguments"]),
                    },
                }
                for c in d["tool_calls"]
            ]
            finish = "tool_calls"
        else:
            message["content"] = d.get("content", self._echo(body))
            finish = d.get("finish_reason", "stop")
        if self.dialect == "deepseek" and d.get("reasoning") is not None:
            message["reasoning_content"] = d["reasoning"]
        prompt = d.get("prompt_tokens", 10 + len(raw) // 40)
        completion = d.get("completion_tokens", 5)
        out: dict[str, Any] = {
            "id": f"chatcmpl-{len(self.seen)}",
            "object": "chat.completion",
            "model": body.get("model", ""),
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        }
        if d.get("usage") != "none":
            cached = int(d.get("cached", 0))
            if self.dialect == "openai":
                usage: dict[str, Any] = {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": prompt + completion,
                }
                if d.get("usage") == "cached" or cached:
                    usage["prompt_tokens_details"] = {"cached_tokens": cached}
            else:
                usage = {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": prompt + completion,
                    "prompt_cache_hit_tokens": cached,
                    "prompt_cache_miss_tokens": prompt - cached,
                }
            out["usage"] = usage
        return JSONResponse(out)

    @staticmethod
    def _echo(body: dict[str, Any]) -> str:
        if body.get("response_format"):
            return json.dumps({"answer": "stand-in"})
        for m in reversed(body.get("messages", [])):
            if m.get("role") == "user":
                return f"echo: {m.get('content', '')}"
        return "echo"


class ThreadedServer:
    """Run an ASGI app with uvicorn on 127.0.0.1 in a background thread (port 0 = any free port)."""

    def __init__(self, app: Any, port: int = 0) -> None:
        import uvicorn

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", port))
        self.port = self._sock.getsockname()[1]
        config = uvicorn.Config(app, log_level="warning", lifespan="off")
        self.server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        loop = (
            asyncio.SelectorEventLoop() if hasattr(asyncio, "SelectorEventLoop") else asyncio.new_event_loop()
        )
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.server.serve(sockets=[self._sock]))
        loop.close()

    def __enter__(self) -> ThreadedServer:
        self._thread.start()
        import time

        deadline = time.monotonic() + 10
        while not self.server.started:
            if time.monotonic() > deadline:
                raise RuntimeError("stand-in server did not start")
            time.sleep(0.01)
        return self

    def __exit__(self, *exc: Any) -> None:
        self.server.should_exit = True
        self._thread.join(timeout=10)
        self._sock.close()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    p = argparse.ArgumentParser(description="stand-in chat-completions provider (test double)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18510)
    p.add_argument("--dialect", default="openai", choices=["openai", "deepseek"])
    args = p.parse_args(argv)
    uvicorn.run(StandIn(args.dialect).app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
