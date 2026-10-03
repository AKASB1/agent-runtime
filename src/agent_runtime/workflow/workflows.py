"""The agent loop: ``chat`` and the planner modes ``react``, ``react_parallel``, ``plan_execute``,
``plan_replan``, each inside the attempt loop (verifier and cascade escalation).

- ``react``: each turn the model issues one tool call or the final answer; the runtime runs it and
  appends the result (several calls in one turn run one after another).
- ``react_parallel``: each turn the model issues all calls whose dependencies are done, and they run
  together under the concurrency limit.
- ``plan_execute``: one planning call returns the whole plan, the scheduler runs it, one call writes
  the answer. Dependents of a failed step are skipped.
- ``plan_replan``: as ``plan_execute``; a failed step stops the schedule and triggers a replanning
  call (budget ``max_replans``) that sees the completed results and the failure.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from agent_runtime.canon import canonical
from agent_runtime.models.structured import StructuredOutputError
from agent_runtime.models.types import Message, ToolCall
from agent_runtime.planning.plan import PLAN_SCHEMA, PlanStep, run_dag, validate_plan
from agent_runtime.routing.routers import AttemptOutcome
from agent_runtime.runtime.engine import LAST_MODEL_SPAN, Run, StepFailed, workflow

INVALID_CODES = ("INVALID_ARGUMENTS", "UNKNOWN_TOOL")


def system_prompt(run: Run, inp: dict[str, Any]) -> str:
    """The system prompt, with the skills catalog (one line per skill) when the run has skills."""
    text = inp.get("system") or run.env.system_prompt
    if run.env.skills is not None:
        catalog = run.env.skills.catalog()
        if catalog:
            run.meter.skill_catalog_tokens += run.env.skills.catalog_tokens()
            text = text + "\n\n" + catalog
    return text


def start_conversation(run: Run, inp: dict[str, Any]) -> None:
    """System prompt (once per conversation) and the task as the newest user message."""
    if not run.events():
        run.append(Message(role="system", content=system_prompt(run, inp)))
    run.append(Message(role="user", content=inp.get("task") or inp.get("message") or ""))


@workflow("chat")
async def chat(run: Run, inp: dict[str, Any]) -> str:
    if not run.events():
        run.append(Message(role="system", content=inp.get("system") or run.env.system_prompt))
    run.append(Message(role="user", content=inp["message"]))
    async with run.step("answer"):
        resp = await run.call_model("answer")
    return resp.text


async def attempt_loop(run: Run, inp: dict[str, Any], body) -> str:  # noqa: ANN001
    """Run ``body`` as attempt 1, 2, ...: verify a final answer; a rejected or failed attempt runs
    again (fresh conversation) when the router escalates."""
    attempt = 1
    while True:
        conv = (
            "main" if attempt == 1 else f"{run.run_id}/a{attempt}"
        )  # later attempts: a fresh conversation of this run
        status, reason, text = "succeeded", None, ""
        calls_before = run.meter.model_calls
        try:
            async with run.step(f"attempt{attempt}", conv=conv, attempt=attempt):
                text = await body(run, inp)
                accepted = verify(run, text, attempt)
        except StepFailed as e:
            status, reason, accepted = "failed", e.reason, False
        run.history.append(
            AttemptOutcome(
                attempt, run.last_target or "", status, accepted, run.meter.model_calls - calls_before
            )
        )
        if accepted:
            run.accepted = True
            return text
        if run.router.escalate(
            replace(run.route_context("escalate", run.env.registry.targets()), attempt=attempt)
        ):
            run.meter.escalations += 1
            attempt += 1
            continue
        if status == "failed":
            raise StepFailed(reason)
        run.accepted = False
        run.extra["failure_reason"] = "verifier_rejected"
        return text


def verify(run: Run, text: str, attempt: int) -> bool:
    v = run.env.verifier
    if v is None:
        return True
    f = run.frame
    span = run.trace.start("validate", "verifier", f.span, {"agent_runtime.step.path": f.path})
    ok = bool(v(text, attempt))
    run.trace.end(span, "ok", **{"agent_runtime.verifier.accepted": ok})
    return ok


async def react_body(run: Run, inp: dict[str, Any], *, parallel: bool) -> str:
    start_conversation(run, inp)
    turn = 0
    invalid_streak = 0
    exclude: set[str] = set()
    while True:
        turn += 1
        async with run.step(f"turn{turn}"):
            resp = await run.call_model(
                "act", tools=run.tool_specs(), exclude=exclude, meta={"parallel": True} if parallel else None
            )
            if not resp.tool_calls:
                return resp.text
            origin = LAST_MODEL_SPAN.get()
            results = await run.run_tool_calls(resp.tool_calls, origin, parallel=parallel)
        if any(r.error_code in INVALID_CODES for r in results):
            # the validation error went back as the tool result: a repair of this step
            run.meter.repairs += 1
            invalid_streak += 1
            if invalid_streak > run.config.repair_budget:
                exclude.add(run.last_target or "")
                invalid_streak = 0
                cands, _ = run._candidates(run.events(), {"tools"}, run.config.max_output_tokens, exclude)
                if not cands or run.router.choose(run.route_context("act", cands)) is None:
                    raise StepFailed("invalid_output", "repair budget used up and no next model")
        else:
            invalid_streak = 0
            exclude = set()


@workflow("react")
async def react(run: Run, inp: dict[str, Any]) -> str:
    return await attempt_loop(run, inp, lambda r, i: react_body(r, i, parallel=False))


@workflow("react_parallel")
async def react_parallel(run: Run, inp: dict[str, Any]) -> str:
    return await attempt_loop(run, inp, lambda r, i: react_body(r, i, parallel=True))


# -- plans ---------------------------------------------------------------------------------------


async def make_plan(run: Run, role: str, n: int, completed: set[str]) -> tuple[list[PlanStep], str]:
    allow = list(run.frame.tools_allow) if run.frame.tools_allow is not None else None
    holder: dict[str, list[PlanStep]] = {}

    def check(value: Any) -> None:
        holder["steps"] = validate_plan(value, run.env.tools, allow=allow, completed=completed)

    try:
        await run.call_structured(role, PLAN_SCHEMA, validate=check, max_output_tokens=1024)
    except StepFailed as e:
        if e.reason == "invalid_output":
            raise StepFailed("plan_invalid", str(e)) from None
        raise
    steps = holder["steps"]
    f = run.frame
    span = run.trace.start(
        "plan",
        f"plan{n}",
        f.span,
        {
            "agent_runtime.step.path": f.path,
            "agent_runtime.plan.step_ids": [f"p{n}#{s.id}" for s in steps],
            "agent_runtime.plan.steps": [
                {"id": s.id, "tool": s.tool, "depends_on": list(s.depends_on)} for s in steps
            ],
        },
    )
    run.trace.end(span, "ok")
    return steps, span.span_id


async def execute_plan(
    run: Run,
    steps: list[PlanStep],
    n: int,
    plan_span: str,
    *,
    stop_on_failure: bool,
    completed: dict[str, dict[str, Any]],
    done_ids: set[str],
) -> list[str]:
    """Run the plan's steps; returns the ids of failed steps. Completed (tool, args) pairs are never
    run again (their earlier result is reused), so a write step never runs twice."""
    failed: list[str] = []

    async def run_step(step: PlanStep) -> bool:
        sig = canonical([step.tool, step.args])
        if sig in completed:
            done_ids.add(step.id)
            return True
        call = ToolCall(id=f"p{n}#{step.id}", name=step.tool, arguments=step.args)
        async with run.step(f"exec_{step.id}"):
            res = await run.call_tool(call, plan_span)
        run.append(Message(role="assistant", tool_calls=[call]))
        run.append(Message(role="tool", tool_call_id=call.id, content=res.content()))
        if res.ok:
            completed[sig] = {"step": step.id}
            done_ids.add(step.id)
            return True
        failed.append(step.id)
        return False

    k = max(1, run.config.concurrency)
    await run_dag(steps, k, run_step, stop_on_failure=stop_on_failure, done=done_ids)
    return failed


async def plan_body(run: Run, inp: dict[str, Any], *, replan: bool) -> str:
    start_conversation(run, inp)
    completed: dict[str, dict[str, Any]] = {}
    done_ids: set[str] = set()
    n = 1
    async with run.step("plan"):
        steps, span = await make_plan(run, "plan", n, set())
    replans = 0
    while True:
        async with run.step(f"execute{n}"):
            failed = await execute_plan(
                run, steps, n, span, stop_on_failure=replan, completed=completed, done_ids=done_ids
            )
        if failed and replan and replans < run.config.max_replans:
            replans += 1
            run.meter.replans += 1
            run.append(
                Message(
                    role="user",
                    content=(
                        f"Step(s) {', '.join(failed)} failed. Completed steps: {', '.join(sorted(done_ids)) or 'none'}. "
                        "Return a plan for the remaining steps only."
                    ),
                )
            )
            n += 1
            async with run.step(f"replan{replans}"):
                steps, span = await make_plan(run, "replan", n, set(done_ids))
            continue
        break
    async with run.step("answer"):
        resp = await run.call_model("answer")
    return resp.text


@workflow("plan_execute")
async def plan_execute(run: Run, inp: dict[str, Any]) -> str:
    return await attempt_loop(run, inp, lambda r, i: plan_body(r, i, replan=False))


@workflow("plan_replan")
async def plan_replan(run: Run, inp: dict[str, Any]) -> str:
    return await attempt_loop(run, inp, lambda r, i: plan_body(r, i, replan=True))


__all__ = ["StructuredOutputError", "attempt_loop", "start_conversation"]


@workflow("agent")
async def agent(run: Run, inp: dict[str, Any]) -> str:
    """``react`` over the run's registry (workspace tools, MCP tools, ``delegate``) with the context
    manager and the permission layer; what differs from ``react`` is the environment."""
    return await attempt_loop(run, inp, lambda r, i: react_body(r, i, parallel=False))


# -- delegation (E6) -----------------------------------------------------------------------------


async def delegate_body(run: Run, inp: dict[str, Any]) -> str:
    """One planning call; the plan's branches (components without its tail) run as child agents in
    parallel; the parent runs the tail steps and writes the answer from the children's texts."""
    from agent_runtime.agents import AgentSpec
    from agent_runtime.planning.plan import plan_branches, plan_tail

    start_conversation(run, inp)
    async with run.step("plan"):
        steps, span = await make_plan(run, "plan", 1, set())
    groups = plan_branches(steps)
    child = inp.get("child_target") or {"target": "sim-a/small", "fallback": ["sim-a/medium"]}
    calls = []
    for n, group in enumerate(groups, 1):
        bsteps = [s for s in steps if s.id in group]
        name = f"branch{n}"
        run.env.agents[name] = AgentSpec(
            name=name,
            tools=sorted({s.tool for s in bsteps}),
            target=child["target"],
            fallback=child.get("fallback", []),
            max_steps=2 * len(bsteps) + 2,
            system_prompt="You are a helper agent. Run the listed tool calls and report their results.",
        )
        task = "Run these tool calls and report their results: " + "; ".join(
            f"{s.id} {s.tool} {canonical(s.args)}" for s in bsteps
        )
        calls.append(ToolCall(id=f"d{n}", name="delegate", arguments={"agent": name, "task": task}))
    async with run.step("delegate") as st:
        run.append(Message(role="assistant", tool_calls=calls))
        await run.run_tool_calls(calls, st, parallel=True)
    tail = set(plan_tail(steps))
    done = {s.id for s in steps if s.id not in tail}
    if tail:
        async with run.step("execute1"):
            await execute_plan(
                run,
                [s for s in steps if s.id in tail],
                1,
                span,
                stop_on_failure=False,
                completed={},
                done_ids=done,
            )
    async with run.step("answer"):
        resp = await run.call_model("answer")
    return resp.text


@workflow("delegate_plan")
async def delegate_plan(run: Run, inp: dict[str, Any]) -> str:
    return await attempt_loop(run, inp, delegate_body)
