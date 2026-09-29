"""Run with PYTHONPATH=src python examples/local_run.py."""
import asyncio
from agent_runtime.models import ModelResponse
from agent_runtime.runtime import run_once

class EchoModel:
    async def complete(self, messages):
        return ModelResponse(messages[-1].content)

print(asyncio.run(run_once(EchoModel(), "hello")).text)
