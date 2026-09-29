import asyncio
import unittest
from agent_runtime.models import ModelResponse
from agent_runtime.runtime import run_once
from agent_runtime.tools import Tool, ToolRegistry

class FakeModel:
    async def complete(self, messages):
        return ModelResponse(messages[-1].content.upper())

class RuntimeTests(unittest.TestCase):
    def test_model_and_tool(self):
        self.assertEqual(asyncio.run(run_once(FakeModel(), "hello")).text, "HELLO")
        async def echo(args):
            return args["text"]
        registry = ToolRegistry()
        registry.register(Tool("echo", echo))
        self.assertEqual(asyncio.run(registry.invoke("echo", {"text": "ok"})), "ok")
