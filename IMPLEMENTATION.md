# Implementation plan and status

## Design rule

Keep the runtime deterministic where possible. Model generation is one step inside an explicit execution graph; tool calls, permission decisions, compactions, and state transitions are inspectable and replayable. What the model sees is always derived from a logged session.

## Core interfaces

- **Model adapter**: `complete(ModelRequest, ModelSpec) -> ModelResponse`, raising only `ProviderError` (a closed taxonomy). Requests carry messages, tool specs, an optional response schema, limits, and `meta`; responses carry text, tool calls, a finish reason, and four-way usage (uncached input, output, cache read, cache write) with an `estimated` flag.
- **Tool**: a name, Pydantic input and output models (or a declared JSON Schema for MCP tools), a timeout, a side effect (`none`, `read`, `write`), and a capability (`none`, `fs_read`, `fs_write`, `exec`, `net`) that the permission layer reads.
- **Workflow**: an async function over the run engine (`call_model`, `call_structured`, `call_tool`) with typed configuration; plans are step graphs run by a list scheduler.
- **Trace**: one per run, one span per model call, tool call, routing decision, permission decision, compaction, subagent, and validation; replay re-executes a run from it.

Formats and rules: [docs/contracts.md](docs/contracts.md). Layers: [docs/harness.md](docs/harness.md).

## Delivery order (Tier 1 = M1-M12)

- [x] M1 Foundations: toolchain, clock and virtual-time loop, seeded streams, contracts
- [x] M2 Models and providers: types and error taxonomy, registry, accounting, scripted model, OpenAI-compatible adapter (dialects `openai`, `deepseek`), stand-in server, structured output with repair, `scripts/live_smoke.py`
- [x] M3 Tools and MCP: typed registry, MCP stdio client (SDK 2.2.0) with PID-owned lifecycle, native-vs-MCP equivalence
- [x] M4 Workflow, state, trace, replay: engine (sequential and parallel, budgets, cancellation), SQLite store, session log and derivation, trace and `validate_trace`, replay with divergence report
- [x] M5 Simulator and runner: world, tasks, verifier, simulated model, faults, evaluation runner and statistics
- [x] M6 Decision layer: retry, fallback, failure table; routers `fixed`, `random_mix`, `cascade`, `threshold`, offline oracle; planner modes, plan validator, list scheduler; informativeness check
- [x] M7 Context manager: policies `none`, `window`, `elide`, `summarize`, compaction with hysteresis, fork
- [x] M8 Workspace tools and permissions: resolver, decision table, approver, permission spans, env allowlist, caps, process-tree kill
- [x] M9 Subagents: `delegate`, catalog, allowlist, shared meter and budget share, depth limit, nested spans, replay and cancellation
- [x] M10 Worked example: six problems, MCP solver and verifier, `optimize` workflow, E4a
- [x] M11 API and operations: FastAPI service, ceilings, fork, context endpoint, OpenAPI drift test, JSON logs, CLI, demo
- [x] M12 Evaluation and delivery: reviews, full evaluation (E1-E3, E4a, and the Tier 2 studies E4b, E5, E6), figures, documentation, CI file

Tier 2, built in this run: [x] context study E5, [x] verifier study E4b, [x] skills (catalog and `load_skill`), [x] delegation study E6 (`delegate_plan`). Not run: live validation E7 (no endpoint configured). Not built (see the README's `## TODO`): durable resume, approvals and steering over the API, Anthropic Messages adapter, circuit breaker and hedging, tool routing, wire conformance against the serving gateway of `llm-serving-control`, OpenTelemetry export, PostgreSQL parity, retrieval and persistent memory.

## Decision problem

- **Quality** is defined twice: at run time it is the verifier's acceptance of the final answer plus the validations the runtime saw (no invalid output left); in evaluation it is the checker's `success` (the final answer equals the gold answer and the world saw no duplicate or wrong write). A router only ever sees the first.
- **Routing**: per call, choose a target among the candidates (targets with the needed capabilities whose window holds the input's irreducible part) under a budget of cost, latency (deadline), and model calls; the cascade escalates a rejected or failed attempt to the next model unless the attempt cannot finish before the deadline at the model's median latency. Routers are compared with random mixing at the same mean cost (E1).
- **Retry and fallback**: the failure table of the contracts (E2 measures it under fault profiles).
- **Planning**: react, react_parallel, plan_execute, plan_replan; a replan sees the completed results and the failure and never repeats a completed step (E3). `delegate_plan` runs the independent branches of a plan as child agents (E6).
- **Scheduling**: list scheduling of independent plan steps under a concurrency limit k, by longest remaining path (Graham's bound tested).

The accounting that feeds the evaluation also feeds these decisions (spent cost, elapsed time, calls), so every rule is compared with simple baselines that may win.

## Evaluation and acceptance

See [docs/evaluation.md](docs/evaluation.md): frozen protocol (tuning seeds 1-10, evaluation seeds 101-130), paired differences with Student-t intervals, win/tie/loss with fixed tie bands, the informativeness check, and the results. Every result is simulated.

## Cross-project contracts

- The conventions (time, randomness, units, determinism, frozen configurations, disjoint seeds, common random numbers, paired differences, simple baselines) follow version 1 of the contracts of `llm-serving-control` and `gpu-cluster-scheduler`. No code is shared with them; this repository defines its own contracts in [docs/contracts.md](docs/contracts.md).
- The runtime's OpenAI-compatible adapter is what a client of the OpenAI-compatible gateway of `llm-serving-control` would use; this is **not tested** (the wire-conformance run against that gateway is Tier 2 and did not run).

## Status checkpoint

All of Tier 1 and the Tier 2 items E4b, E5, skills, and E6 are implemented, tested, and evaluated in simulation (see the README for the test, benchmark, and demo commands and the results). Models are simulated or scripted; the OpenAI-compatible adapter was tested only against the local stand-in.
