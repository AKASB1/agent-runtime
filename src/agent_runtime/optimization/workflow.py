"""The ``optimize`` workflow, built from the runtime's public API:

``formulate`` (a model call with the formulation as structured output) -> ``solve`` -> ``verify``;
a rejection sends the diagnostics back for a new formulation (at most 2 repairs); then ``report`` (a
model call that states the objective and the values of the card's variables) -> ``check_report``; a
mismatch sends one re-ask. The solver and the verifier are tools of an MCP server (``opt.*``).
"""

from __future__ import annotations

import json
from typing import Any

from agent_runtime.canon import canonical
from agent_runtime.models.types import Message, ToolCall
from agent_runtime.optimization.core import FORMULATION_SCHEMA, REPORT_SCHEMA
from agent_runtime.runtime.engine import Run, StepFailed, workflow

SYSTEM = (
    "You are an optimization modelling assistant. Write a linear or mixed-integer formulation as JSON with "
    "'sense', 'variables' (name, lb, ub, type), 'objective' (terms, constant), and 'constraints' (name, terms, "
    "sense, rhs). Use the card's variable names for the decision variables."
)
MAX_VERIFIER_REPAIRS = 2


def card_prompt(card: dict[str, Any]) -> str:
    return (
        f"Problem {card['id']}: {card['statement']}\n"
        f"Data: {canonical(card['data'])}\n"
        f"Decision variables (use these names): {canonical(card['variables'])}\n"
        "Return the formulation as one JSON object."
    )


def _codes(diags: list[dict[str, str]]) -> list[str]:
    return [d["code"] for d in diags]


@workflow("optimize")
async def optimize(run: Run, inp: dict[str, Any]) -> str:
    card = inp["card"]
    run.append(Message(role="system", content=SYSTEM))
    run.append(Message(role="user", content=card_prompt(card)))
    codes: list[list[str]] = []
    run.extra["diagnostics"] = codes
    repairs = 0
    n = 0
    while True:
        n += 1
        async with run.step(f"formulate{n}"):
            form, _ = await run.call_structured(
                "formulate", FORMULATION_SCHEMA, error_prefix="V1_PARSE", max_output_tokens=2048
            )
        async with run.step(f"solve{n}") as st:
            sres = await run.call_tool(
                ToolCall(id=f"solve{n}", name="opt.solve", arguments={"formulation": form}), st
            )
        if not sres.ok:
            raise StepFailed("step_failed", f"solver tool failed: {sres.error_message}")
        async with run.step(f"verify{n}") as st:
            vres = await run.call_tool(
                ToolCall(
                    id=f"verify{n}",
                    name="opt.verify",
                    arguments={"card": card, "formulation": form, "solution": sres.result},
                ),
                st,
            )
        if not vres.ok:
            raise StepFailed("step_failed", f"verifier tool failed: {vres.error_message}")
        diags = vres.result["diagnostics"]
        codes.append(_codes(diags))
        if vres.result["accepted"]:
            break
        if repairs >= MAX_VERIFIER_REPAIRS:
            run.extra["candidates"] = n
            raise StepFailed("verifier_rejected", "; ".join(_codes(diags)))
        repairs += 1
        run.append(
            Message(
                role="user",
                content="The verifier rejected the formulation: "
                + json.dumps(diags)
                + " Return a corrected formulation.",
            )
        )
    run.extra["candidates"] = n
    solution = sres.result
    run.append(
        Message(
            role="user",
            content=(
                f"The solver returned: {canonical(solution)}. Report the objective and the values of the card's variables "
                'as JSON {"objective": number, "values": {name: number}}.'
            ),
        )
    )
    for k in (1, 2):
        async with run.step(f"report{k}"):
            rep, _ = await run.call_structured("report", REPORT_SCHEMA)
        async with run.step(f"check{k}") as st:
            cres = await run.call_tool(
                ToolCall(
                    id=f"check{k}",
                    name="opt.check_report",
                    arguments={"card": card, "report": rep, "solution": solution},
                ),
                st,
            )
        if not cres.ok:
            raise StepFailed("step_failed", f"check_report failed: {cres.error_message}")
        codes.append(_codes(cres.result["diagnostics"]))
        if cres.result["accepted"]:
            run.extra["objective"] = solution["objective"]
            return canonical(rep)
        if k == 1:
            run.append(
                Message(
                    role="user",
                    content="The report does not match the solver: "
                    + json.dumps(cres.result["diagnostics"])
                    + " Report again.",
                )
            )
    raise StepFailed("verifier_rejected", "the report does not match the solver")
