"""FastAPI application. Every error has the body ``{"error": {"code", "message", "details"}}``.

Error codes: ``NOT_FOUND`` (404), ``BAD_REQUEST`` (400), ``UNKNOWN_WORKFLOW`` (400), ``VALIDATION_ERROR``
(422), ``CEILING_EXCEEDED`` (403), ``VERSION_CONFLICT`` (409), ``RUN_NOT_FINISHED`` (409),
``NOT_REPLAYABLE`` (409), ``INTERNAL`` (500).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from agent_runtime import __version__
from agent_runtime.api.service import ApiError, Service, Settings
from agent_runtime.store.base import NotFound, VersionConflict

API_ERROR_CODES = (
    "NOT_FOUND",
    "BAD_REQUEST",
    "UNKNOWN_WORKFLOW",
    "VALIDATION_ERROR",
    "CEILING_EXCEEDED",
    "VERSION_CONFLICT",
    "RUN_NOT_FINISHED",
    "NOT_REPLAYABLE",
    "INTERNAL",
)


class RunBody(BaseModel):
    workflow: str
    input: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


class StateBody(BaseModel):
    expected_version: int
    state: dict[str, Any] = Field(default_factory=dict)


class ForkBody(BaseModel):
    at_seq: int = Field(ge=0)


def error(status: int, code: str, message: str, details: dict | None = None) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message, "details": details or {}}}, status_code=status
    )


def create_app(settings: Settings | None = None, service: Service | None = None) -> FastAPI:
    app = FastAPI(
        title="agent-runtime",
        version=__version__,
        description="A small, deterministic, replayable agent runtime (harness level). Models are simulated or scripted unless an OpenAI-compatible endpoint is configured.",
    )
    svc = service or Service(settings or Settings())
    app.state.service = svc

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return error(exc.status, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        return error(
            422,
            "VALIDATION_ERROR",
            str(first.get("msg", "invalid request")),
            {"loc": [str(x) for x in first.get("loc", [])]},
        )

    @app.exception_handler(NotFound)
    async def _not_found(request: Request, exc: NotFound) -> JSONResponse:
        return error(404, "NOT_FOUND", f"not found: {exc.args[0] if exc.args else ''}")

    @app.exception_handler(VersionConflict)
    async def _conflict(request: Request, exc: VersionConflict) -> JSONResponse:
        return error(409, "VERSION_CONFLICT", str(exc), {"expected": exc.expected, "actual": exc.actual})

    @app.exception_handler(Exception)
    async def _internal(request: Request, exc: Exception) -> JSONResponse:
        return error(500, "INTERNAL", type(exc).__name__)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "models": [
                {
                    "target": s.target,
                    "provider": s.provider,
                    "model": s.id,
                    "wire": s.wire,
                    "dialect": s.dialect,
                    "price_in_ucu_per_mtok": s.price_in_ucu,
                    "price_out_ucu_per_mtok": s.price_out_ucu,
                    "price_cache_read_ucu_per_mtok": s.price_cache_read_ucu,
                    "priced": s.priced,
                    "context_tokens": s.context_tokens,
                    "capabilities": s.capabilities,
                }
                for s in svc.registry().specs()
            ]
        }

    @app.get("/v1/tools")
    async def tools() -> dict[str, Any]:
        return {"tools": svc.tool_catalog()}

    @app.get("/v1/usage")
    async def usage() -> dict[str, Any]:
        totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cost_ucu": 0}
        for u in svc.usage.values():
            for k in totals:
                totals[k] += u[k]
        return {
            "by_model": {k: dict(v) for k, v in sorted(svc.usage.items())},
            "total": totals,
            "cost_unit": "ucu (simulated cost units)",
        }

    @app.post("/v1/sessions", status_code=201)
    async def create_session() -> dict[str, Any]:
        sid = svc.store.create_session({})
        return {"id": sid, "events": [], "state": {}, "version": 0}

    @app.get("/v1/sessions/{sid}")
    async def get_session(sid: str) -> dict[str, Any]:
        events = svc.store.events(sid)
        state, version = svc.store.get_state(sid)
        return {
            "id": sid,
            "events": events,
            "state": state,
            "version": version,
            "runs": [r["id"] for r in svc.store.list_runs(sid)],
        }

    @app.put("/v1/sessions/{sid}/state")
    async def put_state(sid: str, body: StateBody) -> dict[str, Any]:
        version = svc.store.put_state(sid, body.state, body.expected_version)
        return {"id": sid, "state": body.state, "version": version}

    @app.post("/v1/sessions/{sid}/fork", status_code=201)
    async def fork(sid: str, body: ForkBody) -> dict[str, Any]:
        n = len(svc.store.events(sid))
        if body.at_seq > n:
            raise ApiError(422, "VALIDATION_ERROR", f"at_seq {body.at_seq} is beyond the log ({n} events)")
        new = svc.store.fork(sid, body.at_seq)
        state, version = svc.store.get_state(new)
        return {
            "id": new,
            "forked_from": sid,
            "at_seq": body.at_seq,
            "events": svc.store.events(new),
            "state": state,
            "version": version,
        }

    @app.get("/v1/sessions/{sid}/context")
    async def context(sid: str, policy: str = Query("none"), window: int = Query(8192)) -> dict[str, Any]:
        return svc.context(sid, policy, window)

    async def _run(session_id: str | None, body: RunBody, wait: bool) -> JSONResponse:
        run_id = await svc.start_run(session_id, body.model_dump())
        if wait:
            await svc.wait(run_id)
            return JSONResponse(svc.get_run(run_id), status_code=200)
        return JSONResponse({"id": run_id, "status": "running"}, status_code=202)

    @app.post("/v1/runs", status_code=202)
    async def post_run(body: RunBody = Body(...), wait: bool = Query(False)) -> JSONResponse:
        return await _run(None, body, wait)

    @app.post("/v1/sessions/{sid}/runs", status_code=202)
    async def post_session_run(
        sid: str, body: RunBody = Body(...), wait: bool = Query(False)
    ) -> JSONResponse:
        return await _run(sid, body, wait)

    @app.get("/v1/runs/{rid}")
    async def get_run(rid: str) -> dict[str, Any]:
        return svc.get_run(rid)

    @app.get("/v1/runs/{rid}/trace", response_class=PlainTextResponse)
    async def get_trace(rid: str) -> PlainTextResponse:
        return PlainTextResponse(svc.trace(rid), media_type="application/x-ndjson")

    @app.post("/v1/runs/{rid}/cancel")
    async def cancel(rid: str) -> dict[str, Any]:
        return svc.cancel(rid)

    @app.post("/v1/runs/{rid}/replay")
    async def replay(rid: str) -> dict[str, Any]:
        return await svc.replay(rid)

    return app


def write_openapi(path: str = "docs/openapi.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(openapi_text(), encoding="utf-8", newline="\n")


def openapi_text() -> str:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(Settings(db=str(Path(tmp) / "o.db"), workspace_root=str(Path(tmp) / "ws")))
        text = json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"
        app.state.service.store.close()
    return text
