# LLM Agent Runtime

A small runtime for tool-using LLM applications. The goal is to make model calls, tools, state, retries, tracing, and evaluation explicit instead of hiding them behind one large agent abstraction.

**Status:** implementation scaffold.

## Scope

- provider-neutral model interface
- tool registry with typed inputs and outputs
- structured output validation
- explicit workflow / state-machine execution
- retrieval adapters
- short-term and persistent memory
- retry, timeout, and cancellation handling
- tracing, latency, token usage, and cost accounting
- offline evaluation hooks

## Proposed stack

Python 3.12 · FastAPI · Pydantic · asyncio · PostgreSQL · Redis · OpenTelemetry

## Runtime model

```text
Request
  │
  ▼
Workflow / State
  │
  ├──► Model Adapter
  ├──► Tool Registry
  ├──► Retrieval
  └──► Memory
        │
        ▼
Trace + Usage + Evaluation
```

## Repository layout

```text
src/agent_runtime/
  models/
  tools/
  workflow/
  retrieval/
  memory/
  runtime/
  telemetry/
  evals/
tests/
examples/
```

See [IMPLEMENTATION.md](IMPLEMENTATION.md).

## Reference projects

- [openai/openai-agents-python](https://github.com/openai/openai-agents-python) — lightweight agent primitives, tracing, and handoffs
- [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) — stateful graph execution
- [microsoft/autogen](https://github.com/microsoft/autogen) — multi-agent runtime patterns and tool orchestration

## License

MIT

## Available now

Importable runtime primitives cover model calls, a named async tool registry, sequential steps, in-memory session messages, and exact-match evaluation. Run `PYTHONPATH=src python -m unittest discover -s tests` or `PYTHONPATH=src python examples/local_run.py`. Provider adapters and external storage are planned.
