# Harness layers

This project is a small agent runtime at the level of a harness: the layer between a model and its tools that owns the agent loop, the model's context, the tools and their permissions, the sessions, and the accounting. It is not an agent framework, not a coding agent, and no claim is made about how well any model works in it.

| Harness layer | In this repository (modules) | Status |
|---|---|---|
| Agent loop | `runtime/engine.py` (Run), `workflow/workflows.py` (react, react_parallel, plan_execute, plan_replan, agent, attempt loop; `delegate_plan` for E6), `planning/` (validator, list scheduler) | built |
| Model adapters | `providers/scripted.py`, `sim/model.py` (simulated), `providers/openai_chat.py` (dialects openai, deepseek; tested against a local stand-in only) | built; Anthropic Messages: Tier 2, not built |
| Context manager | `context/derive.py` (derivation, none/window/elide/summarize, hysteresis), compaction events in `store/`, fork | built |
| Tool runtime | `tools/` (typed registry), `mcp/` (MCP over stdio on the mcp SDK), `workspace/` (files, exec without shell, HTTP GET) | built |
| Skills | `agents/skills.py` (a directory of `SKILL.md` files; the catalog in the system prompt; `load_skill`; catalog and body tokens counted per run) | built (Tier 2 item) |
| Subagents | `agents/` (catalog, delegate), `Run._run_child` (fresh context, allowlist, budget share, depth limit, nested spans) | built |
| Session and state | `store/` (SQLite and memory; append-only log, versioned state, restart persistence) | built |
| Permissions | `permissions/policy.py` (sandbox and approval modes, path, command, and host policy, approver), `workspace/paths.py` | built; policy inside the process, not isolation by the operating system |
| Retry and recovery | engine: retry with jitter, fallback, repair, budgets, cancellation | built; resume, circuit breaker, hedging: Tier 2, not built |
| Observability | `telemetry/trace.py` (JSON Lines spans with GenAI attribute names, `validate_trace`), `runtime/replay.py` | built; OpenTelemetry export: Tier 2, not built |
| Agent API | `api/` (FastAPI), `cli.py` | built |

## What the permission layer is and is not

It decides, before every tool call, whether the call may run: a sandbox mode (`read_only` or `workspace_write`; there is no mode without a sandbox), an approval mode (`ask`, `auto`, `never`), an allowlist of command prefixes, a deny list of executable names, and an allowlist of hosts. It is **policy inside the process and not isolation by the operating system**: `run_command` can do whatever its command can do under the account that runs the service, the deny list guards against accidents and is not a security boundary, and a symbolic link swapped in between the path check and the use defeats the check.

## Taken from the harness reference, as ideas

The public design documents of DeepSeek's open-source agent harness (`deepseek-ai/deepseek-harness`, MIT) were read for ideas; no code was copied, run, or read from a local copy.

- "Model-visible means logged": the model's input is derived from an append-only session log, and requests are reconstructable from it (here a test rebuilds every request of a batch from the stored log).
- An explicit step (one model request plus the tools it calls) inside a turn.
- A tool pipeline with a guard stage before execution; approval as a fail-closed one-shot answer (no approver means deny).
- Sandbox mode and approval policy as two independent settings.
- Compaction recorded as a log event with the covered range and token counts.
- Subagents behind one interface, the child with a fresh context.

## Where this project differs

- The layers are plain modules with typed interfaces, not plugins on an event bus; there are no waterfalls or hooks.
- Compaction is one event, not a start/summary/end triple, and runs in one process.
- No streaming; traces are JSON Lines and replay compares digests span by span.
- The sandbox is in-process policy only (no bwrap/Landlock/Seatbelt/ACL backends).

## Absent on purpose

A plugin framework, a hook system, a web interface, session format migrations, OS-level sandbox backends, agent teams, browser or computer use. Retrieval and persistent memory are a Tier 2 item that was not built (the scaffold's `Retriever` protocol and in-memory `MemoryStore` stay as they were; README `## TODO`).
