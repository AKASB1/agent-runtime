"""Structured output: parse a model's text against a JSON Schema.

The only normalisation is stripping one surrounding Markdown code fence (```json ... ```), because
models commonly add it; everything else that is not exactly one JSON value matching the schema is a
validation error whose message goes back to the model in the repair loop.
"""

from __future__ import annotations

import json
import re
from typing import Any

import jsonschema

_FENCE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n?```\s*$", re.DOTALL)


class StructuredOutputError(ValueError):
    pass


def parse_structured(text: str, schema: dict[str, Any]) -> Any:
    """Return the parsed value or raise :class:`StructuredOutputError` with a model-readable message."""
    if text is None or not text.strip():
        raise StructuredOutputError("empty output: expected one JSON value matching the schema")
    m = _FENCE.match(text)
    body = m.group(1) if m else text
    try:
        value = json.loads(body)
    except json.JSONDecodeError as e:
        raise StructuredOutputError(
            f"output is not valid JSON: {e.msg} at line {e.lineno} column {e.colno}"
        ) from None
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(value), key=lambda err: (list(err.absolute_path), err.message))
    if errors:
        first = errors[0]
        where = "/".join(str(p) for p in first.absolute_path) or "(root)"
        raise StructuredOutputError(f"schema violation at {where}: {first.message}")
    return value


def validate_args(value: Any, schema: dict[str, Any]) -> str | None:
    """Validate tool arguments against a JSON Schema; return an error message or None."""
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(value), key=lambda err: (list(err.absolute_path), err.message))
    if not errors:
        return None
    first = errors[0]
    where = "/".join(str(p) for p in first.absolute_path) or "(root)"
    return f"{where}: {first.message}"
