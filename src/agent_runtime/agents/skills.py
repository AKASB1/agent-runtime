"""Skills (Tier 2 item 4): a skill is a directory with a ``SKILL.md`` whose front matter has ``name``,
``description``, and an optional ``tools`` list (the tools the skill relies on); the body is the
instructions. The system prompt carries only the catalog (one line per skill); the model loads a
skill with the tool ``load_skill(name)``, whose result is the body (capped). Nothing is loaded
without a call; the run's meter counts the tokens of the catalog and of the loaded bodies."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from agent_runtime.models.accounting import estimate_tokens
from agent_runtime.tools import Tool, ToolError

BODY_CAP_TOKENS = 2000
_FRONT = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    tools: tuple[str, ...] = field(default_factory=tuple)


def parse_skill(text: str) -> Skill:
    m = _FRONT.match(text.replace("\r\n", "\n"))
    if not m:
        raise ValueError("SKILL.md needs a front matter block between --- lines")
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    if "name" not in meta or "description" not in meta:
        raise ValueError("SKILL.md front matter needs name and description")
    tools = tuple(t.strip() for t in meta.get("tools", "").strip("[]").split(",") if t.strip())
    return Skill(meta["name"], meta["description"], m.group(2).strip(), tools)


class SkillRegistry:
    def __init__(self, skills: list[Skill] | None = None) -> None:
        self.skills: dict[str, Skill] = {}
        for s in skills or []:
            if s.name in self.skills:
                raise ValueError(f"duplicate skill {s.name!r}")
            self.skills[s.name] = s

    @classmethod
    def load(cls, directory: str | Path) -> SkillRegistry:
        found = []
        for p in sorted(Path(directory).glob("*/SKILL.md")):
            found.append(parse_skill(p.read_text(encoding="utf-8")))
        return cls(found)

    def catalog(self) -> str:
        lines = [f"- {s.name}: {s.description}" for s in sorted(self.skills.values(), key=lambda s: s.name)]
        return "Skills you can load with load_skill(name):\n" + "\n".join(lines) if lines else ""

    def catalog_tokens(self) -> int:
        return estimate_tokens(self.catalog())

    def tool(self) -> Tool:
        registry = self

        class LoadIn(BaseModel):
            model_config = ConfigDict(extra="forbid")
            name: str = Field(min_length=1)

        async def load(v: LoadIn) -> dict:
            s = registry.skills.get(v.name)
            if s is None:
                raise ToolError("NOT_FOUND", f"no skill named {v.name!r}")
            body = s.body
            b = body.encode("utf-8")
            if len(b) > BODY_CAP_TOKENS * 4:
                body = b[: BODY_CAP_TOKENS * 4].decode("utf-8", errors="ignore") + " [... cut]"
            return {"name": s.name, "tools": list(s.tools), "body": body}

        return Tool(
            name="load_skill",
            handler=load,
            input_model=LoadIn,
            description="Load the instructions of a skill from the catalog.",
            side_effect="read",
            capability="none",
            meta={"skill_loader": True},
        )
