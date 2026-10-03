"""Permission layer: a pure decision function over (policy, capability, resolved arguments), settled
by the approval mode. It is policy inside the process, not isolation by the operating system."""

from agent_runtime.permissions.policy import (
    DEFAULT_DENY_EXEC,
    RULES,
    Decision,
    PermissionPolicy,
    decide,
    exe_name,
    settle,
)

__all__ = ["DEFAULT_DENY_EXEC", "RULES", "Decision", "PermissionPolicy", "decide", "exe_name", "settle"]
