"""Model adapters: scripted, simulated, and OpenAI-compatible (dialects ``openai`` and ``deepseek``)."""

from agent_runtime.providers.openai_chat import DIALECTS, OpenAIChatAdapter, OpenAIChatConfig
from agent_runtime.providers.scripted import ScriptedModel

__all__ = ["DIALECTS", "OpenAIChatAdapter", "OpenAIChatConfig", "ScriptedModel"]
