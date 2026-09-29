"""Model adapter and a single-call runtime."""
from typing import Protocol
from agent_runtime.models import Message, ModelResponse

class ModelAdapter(Protocol):
    async def complete(self, messages: list[Message]) -> ModelResponse: ...

async def run_once(model: ModelAdapter, prompt: str) -> ModelResponse:
    if not prompt:
        raise ValueError("prompt must not be empty")
    return await model.complete([Message("user", prompt)])
