# Architecture

`agent-runtime` is the layer between a model and its tools: it owns the agent loop, the model's context, the tools and the permissions they run under, the sessions, and the accounting. Everything is a plain module with a typed interface; there is no plugin framework and no hook system. The binding formats and rules are in [contracts.md](contracts.md); the harness view is in [harness.md](harness.md).

## Modules and boundaries

```text
src/agent_runtime/
  clock.py          Clock (system, scaled, virtual) and the virtual-time event loop
  rand.py           seeded streams (FNV-1a + SplitMix64), Box-Muller
  canon.py          canonical JSON and SHA-256 digests
  models/           ModelRequest/ModelResponse/Usage/ProviderError, registry, cost (ceil rule), structured output
  providers/        scripted model, OpenAI-compatible adapter (dialects openai, deepseek), stand-in server (test double)
  tools/            typed tool registry, tool results
  mcp/              MCP stdio client on the mcp SDK (PID-owned lifecycle), test server
  workspace/        path resolution and the workspace tools (fs_*, run_command, http_get)
  permissions/      the decision table (pure) and the approval settlement
  context/          derive_messages and the policies none/window/elide/summarize, token counters
  store/            Store interface; in-memory and SQLite implementations
  telemetry/        trace writer and validate_trace
  routing/          routers fixed, random_mix, cascade, threshold
  planning/         plan schema, plan validator, list scheduler
  runtime/          the run engine (model and tool pipelines, budgets, cancellation), replay, run specs
  workflow/         chat, react, react_parallel, plan_execute, plan_replan, agent, delegate_plan (E6); the attempt loop
  agents/           subagent catalog and the delegate tool; skills (SKILL.md catalog and load_skill)
  sim/              simulated world, task generator, simulated model and provider
  optimization/     the worked example: solver, verifier, MCP server, optimize workflow, E4a, the E4b mutants
  evals/            evaluation runner, statistics, experiments E1-E3, E5 (context_study), E6 (delegation_study), report, commands
  api/              FastAPI service
  cli.py            python -m agent_runtime ...
  retrieval/, memory/   scaffold protocols (Tier 2, not built further)
```

Domain code (models, the engine, the derivation and the policies, the permission decision, routers, planners, the simulator, the statistics) imports no web framework and no database driver and reads time only from a `Clock`; a test fails if any module except `clock.py` (and two test doubles) calls the wall clock.

## The run loop

A `Run` executes one workflow with one trace, one meter, one budget, and one cancellation flag. The workflows are ordinary async functions over three engine calls:

1. `call_model(role, ...)`: budget check -> candidate targets (capabilities, and the irreducible part of the input must fit the window) -> `route` span with the router's decision -> derivation of the input from the log for the chosen target (compaction first if due) -> provider attempts with retry, backoff with jitter, and fallback per the failure table (one `model` span per attempt, the request and response payloads with digests) -> billing (`cost_ucu` with the ceil rule) -> the answer is appended to the log.
2. `call_structured(role, schema)`: `call_model` plus schema validation; a failure goes back to the model as a logged user message (repair budget 2), then the router names the next model, then the step fails with `invalid_output`.
3. `call_tool(call, origin)`: budget check -> lookup in the run's (or child's) allowlist -> argument validation -> permission decision (`permission` span for capabilities other than `none`) -> execution under the run's concurrency limit with a timeout and read-only retries -> output validation -> `tool` span.

`react` and `react_parallel` loop over model turns; `plan_execute` and `plan_replan` make one planning call, validate the plan, and list-schedule its steps (whenever fewer than k steps run, start the ready step with the longest remaining path; ties by step id), logging each executed step as an assistant tool call and its result; a replan sees the completed results and the failure and never repeats a completed step. All task workflows run inside the attempt loop: a final answer goes to the run-time verifier, and a rejected or failed attempt runs again in a fresh conversation only when the router escalates (the cascade).

## The session log and the derivation

A session is an append-only log of `message` and `compaction` events plus a versioned JSON state. Every message that reaches a model is logged before the call; the input of a call is `derive_messages(log, policy, budget)`, a pure function. Compaction is triggered with hysteresis (above `compact_at` of the budget, down to `compact_to`), is itself a logged event, and costs one cache miss; between two compactions each input extends the one before. The model span records the log position and the policy, so every request can be rebuilt from the stored log alone (a test does this for 100+ runs). Forks, replays, and the context endpoint use the same function.

## The tool pipeline and the permission layer

Tools are typed (Pydantic models or the JSON Schema an MCP server declares), carry a side effect and a capability, and return results, never exceptions. The permission layer runs after argument validation and before execution; its decision is a pure function of the policy, the capability, and the arguments after path resolution, settled by the approval mode. Path resolution rejects hostile syntax before touching the disk and then requires the real path to lie under the workspace's real path.

**Limits, stated plainly.** The permission layer is policy inside the process, not isolation by the operating system. `run_command` can do whatever its command can do under the account that runs the service. The deny list of executable names guards against accidents, not attacks. Path checks happen at the time of the call: a link that another process swaps in between the check and the use defeats them.

## Subagents

`delegate(agent, task)` runs a child inside the same run: a fresh conversation (the spec's system prompt and the task only), a tool allowlist, a target, `max_steps`, and a budget share bounded by what remains of the parent's budget. Its spans nest under a `subagent` span in the parent's trace; its calls are billed to the parent's run; only its capped final text, status, and step count return. Replay and cancellation pass through children.

## Failure model

Provider errors map to a closed taxonomy and are handled by the failure table (contracts section 3): retry with full jitter for transient errors, fallback after the retries or on `auth`, `context_length` only to larger windows, `bad_request` fails the step. Tool errors are results that the model or planner sees. Budgets and cancellation stop new calls and end the run with its partial trace closed. A replay re-executes the workflow against the recorded calls and raises `ReplayDivergence` at the first differing span.
