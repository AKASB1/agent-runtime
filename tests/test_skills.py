"""Tier 2 item 4: skills (catalog in the system prompt, load_skill, accounting)."""

import pathlib

import pytest

from agent_runtime.agents.skills import BODY_CAP_TOKENS, SkillRegistry, parse_skill
from helpers import ScriptedEnv, kinds, run_scripted, u

ROOT = pathlib.Path(__file__).resolve().parents[1]
SKILLS = SkillRegistry.load(ROOT / "examples" / "skills")


def test_the_example_skills_load_and_the_catalog_has_one_line_each():
    assert sorted(SKILLS.skills) == ["optimization-modelling", "world-orders"]
    cat = SKILLS.catalog()
    assert cat.count("\n- ") == 2 and "world-orders: Answer questions" in cat
    assert "get_order" not in cat  # bodies and tool lists are not in the catalog
    assert SKILLS.skills["world-orders"].tools[0] == "get_order"
    with pytest.raises(ValueError):
        parse_skill("no front matter")


def env_with_skills(script):
    return ScriptedEnv(
        {"p": script},
        tools=lambda: [
            __import__("agent_runtime.mcp.twin", fromlist=["x"]).native_lookup_tool(),
            SKILLS.tool(),
        ],
        skills=SKILLS,
    )


def test_nothing_is_loaded_without_a_call():
    senv = env_with_skills([{"text": "done", "usage": u()}])
    res = run_scripted(senv, "react", {"task": "t"})
    req = senv.models["p"].requests[0]
    assert "Skills you can load" in req.messages[0].content
    assert all("Look up the order with get_order" not in m.content for m in req.messages)
    assert res.meter.skill_catalog_tokens > 0 and res.meter.skill_body_tokens == 0


def test_a_loaded_body_is_a_tool_span_and_is_counted():
    senv = env_with_skills(
        [
            {
                "tool_calls": [{"id": "s1", "name": "load_skill", "arguments": {"name": "world-orders"}}],
                "usage": u(),
            },
            {
                "tool_calls": [{"id": "s2", "name": "load_skill", "arguments": {"name": "no-such-skill"}}],
                "usage": u(),
            },
            {"text": "done", "usage": u()},
        ]
    )
    res = run_scripted(senv, "react", {"task": "t"})
    tools = kinds(res, "tool")
    assert (
        tools[0]["name"] == "load_skill"
        and "Look up the order with get_order" in tools[0]["result"]["result"]["body"]
    )
    assert tools[1]["result"]["error"]["code"] == "NOT_FOUND"
    assert res.meter.skill_body_tokens > 0
    assert "Look up the order" in senv.models["p"].requests[1].messages[-1].content
    assert len(tools[0]["result"]["result"]["body"]) <= BODY_CAP_TOKENS * 4 + 20
