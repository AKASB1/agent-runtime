"""Context manager: the session event log, the derivation of the model's input, and the policies
``none``, ``window``, ``elide``, ``summarize`` (compaction events with hysteresis)."""

from agent_runtime.context.derive import (
    POLICIES,
    STUB_PREFIX,
    SUMMARY_PREFIX,
    Compaction,
    Derived,
    EstimatorCounter,
    SimCounter,
    TokenCounter,
    apply_log,
    derive_messages,
    summary_request_messages,
)

__all__ = [
    "POLICIES",
    "STUB_PREFIX",
    "SUMMARY_PREFIX",
    "Compaction",
    "Derived",
    "EstimatorCounter",
    "SimCounter",
    "TokenCounter",
    "apply_log",
    "derive_messages",
    "summary_request_messages",
]
