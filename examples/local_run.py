"""The scaffold's example on the typed interface: one call to a scripted model.

Run with ``PYTHONPATH=src python examples/local_run.py`` (or after ``pip install -e .``).
"""

import asyncio

from agent_runtime.providers.scripted import ScriptedModel
from agent_runtime.runtime import run_once

model = ScriptedModel([lambda request: {"text": request.messages[-1].content}])
print(asyncio.run(run_once(model, "hello")).text)
