"""Every error kind, run state, failure reason, API error code, permission rule, tool error code, and
verifier code appears in docs/contracts.md and in at least one test."""

import pathlib

import pytest

from agent_runtime.api.app import API_ERROR_CODES
from agent_runtime.models.types import ERROR_KINDS, FAILURE_REASONS, RUN_STATES
from agent_runtime.permissions import RULES
from agent_runtime.tools import TOOL_ERROR_CODES

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACTS = (ROOT / "docs" / "contracts.md").read_text(encoding="utf-8")
TESTS = "\n".join(
    p.read_text(encoding="utf-8")
    for p in (ROOT / "tests").glob("test_*.py")
    if p.name != "test_contracts_doc.py"
)
VERIFIER_CODES = [
    "V1_PARSE", "V1_DUPLICATE_NAME", "V1_UNDECLARED_VARIABLE", "V1_NONFINITE", "V1_BOUNDS", "V1_MISSING_CARD_VARIABLE",
    "V2_INFEASIBLE", "V2_UNBOUNDED", "V3_CONSTRAINT", "V3_OBJECTIVE", "V3_INTEGRALITY",
    "V4_FEASIBLE_WITNESS_REJECTED", "V4_INFEASIBLE_WITNESS_ACCEPTED", "V5_REPORT_MISMATCH",
]  # fmt: skip
ALL = [
    *ERROR_KINDS,
    *RUN_STATES,
    *FAILURE_REASONS,
    *API_ERROR_CODES,
    *RULES,
    *TOOL_ERROR_CODES,
    *VERIFIER_CODES,
]


@pytest.mark.parametrize("name", ALL)
def test_name_is_in_the_contracts_and_in_a_test(name):
    assert f"`{name}`" in CONTRACTS, f"{name} missing from docs/contracts.md"
    assert name in TESTS, f"no test mentions {name}"
