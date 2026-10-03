"""The scaffold's original test, ported to the typed model interface (DECISIONS: the model now
receives a ``ModelRequest`` and the tool registry is typed)."""

import asyncio

from pydantic import BaseModel

from agent_runtime.models import ModelResponse
from agent_runtime.runtime import run_once
from agent_runtime.tools import Tool, ToolRegistry


class FakeModel:
    async def complete(self, request, spec):
        return ModelResponse(text=request.messages[-1].content.upper())


class EchoIn(BaseModel):
    text: str


class EchoOut(BaseModel):
    text: str


def test_model_and_tool():
    assert asyncio.run(run_once(FakeModel(), "hello")).text == "HELLO"

    async def echo(args: EchoIn) -> EchoOut:
        return EchoOut(text=args.text)

    registry = ToolRegistry()
    registry.register(Tool(name="echo", input_model=EchoIn, output_model=EchoOut, handler=echo))
    assert asyncio.run(registry.invoke("echo", {"text": "ok"})) == {"text": "ok"}
