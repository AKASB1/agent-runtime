"""The decision table (``docs/contracts.md``, Workspace tools and permissions). Rows are tried from
the top; the first that applies decides. ``decide`` is pure: path resolution happens before it, and
its result (``resolved``) is what the span records and what replay decides on again."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

DEFAULT_DENY_EXEC: tuple[str, ...] = (
    "cmd", "powershell", "pwsh", "bash", "sh", "wsl",
    "rm", "rmdir", "del", "rd", "format", "diskpart",
    "curl", "wget", "ssh", "scp", "ftp", "nc",
    "docker", "sudo", "reg", "taskkill", "shutdown", "schtasks",
)  # fmt: skip

#: rule ids (stable; they appear in spans and in PERMISSION_DENIED results)
RULES: tuple[str, ...] = (
    "path_outside_workspace",
    "deny_exec",
    "capability_allowed",
    "read_only_sandbox",
    "workspace_write",
    "allow_exec_prefix",
    "exec_needs_approval",
    "allow_hosts",
    "net_needs_approval",
)

Sandbox = Literal["read_only", "workspace_write"]
Approval = Literal["ask", "auto", "never"]


class PermissionPolicy(BaseModel):
    sandbox: Sandbox = "workspace_write"
    approval: Approval = "never"
    allow_exec: list[list[str]] = Field(default_factory=list)
    deny_exec: list[str] = Field(default_factory=lambda: list(DEFAULT_DENY_EXEC))
    allow_hosts: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class Decision:
    decision: Literal["allow", "ask", "deny"]
    rule: str
    #: a denial no approval can overturn never reaches the approver
    final: bool = False


def exe_name(argv0: str) -> str:
    """File name without directory and extension, lower case."""
    base = (
        argv0.replace("\\", "/").rsplit("/", 1)[-1].rstrip(". ")
    )  # Windows ignores trailing dots and spaces
    if "." in base:
        base = base.rsplit(".", 1)[0]
    return base.lower()


def _prefix_match(argv: list[str], prefix: list[str]) -> bool:
    if not prefix or len(argv) < len(prefix):
        return False
    return exe_name(argv[0]) == exe_name(prefix[0]) and argv[1 : len(prefix)] == prefix[1:]


def decide(policy: PermissionPolicy, capability: str, resolved: dict[str, Any]) -> Decision:
    """``resolved``: ``{"paths_ok": bool, "argv": [...], "host": "h:p"}`` (keys as the tool needs)."""
    ro = policy.sandbox == "read_only"
    if resolved.get("paths_ok") is False:
        return Decision("deny", "path_outside_workspace", final=True)
    if capability == "exec":
        argv = list(resolved.get("argv") or [])
        deny = {d.lower() for d in policy.deny_exec}
        if not argv or exe_name(argv[0]) in deny:
            return Decision("deny", "deny_exec", final=True)
    if capability in ("none", "fs_read"):
        return Decision("allow", "capability_allowed")
    if capability == "fs_write":
        return (
            Decision("deny", "read_only_sandbox", final=True) if ro else Decision("allow", "workspace_write")
        )
    if capability == "exec":
        argv = list(resolved.get("argv") or [])
        if any(_prefix_match(argv, p) for p in policy.allow_exec):
            return (
                Decision("deny", "read_only_sandbox", final=True)
                if ro
                else Decision("allow", "allow_exec_prefix")
            )
        return (
            Decision("deny", "read_only_sandbox", final=True)
            if ro
            else Decision("ask", "exec_needs_approval")
        )
    if capability == "net":
        host = resolved.get("host")
        if host in policy.allow_hosts:
            return (
                Decision("deny", "read_only_sandbox", final=True) if ro else Decision("allow", "allow_hosts")
            )
        return (
            Decision("deny", "read_only_sandbox", final=True) if ro else Decision("ask", "net_needs_approval")
        )
    return Decision("deny", "read_only_sandbox", final=True)


Approver = Callable[[str, dict[str, Any], str], Awaitable[bool] | bool]


async def settle(
    policy: PermissionPolicy, d: Decision, approver: Approver | None, tool: str, resolved: dict[str, Any]
) -> tuple[str, str | None]:
    """Return (final decision ``allow``/``deny``, who decided an ``ask``: approver|auto|never|none)."""
    if d.decision != "ask":
        return d.decision, None
    if policy.approval == "auto":
        return "allow", "auto"
    if policy.approval == "never":
        return "deny", "never"
    if approver is None:
        return "deny", "none"
    answer = approver(tool, resolved, d.rule)
    if not isinstance(answer, bool):
        answer = await answer
    return ("allow" if answer is True else "deny"), "approver"
