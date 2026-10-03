"""Cost of a call in micro-cost-units (ucu), integer arithmetic with the ``ceil`` rule."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_runtime.models.registry import ModelSpec
    from agent_runtime.models.types import Usage


def cost_ucu(usage: Usage, spec: ModelSpec) -> int:
    """``ceil((in*p_in + out*p_out + cr*p_cr + cw*p_cw) / 1e6)``; ``in`` is the uncached input."""
    if not spec.priced:
        return 0
    num = (
        usage.input_tokens * spec.price_in_ucu
        + usage.output_tokens * spec.price_out_ucu
        + usage.cache_read_tokens * int(spec.price_cache_read_ucu or 0)
        + usage.cache_write_tokens * int(spec.price_cache_write_ucu or 0)
    )
    return -(-num // 1_000_000)


def estimate_tokens(text: str) -> int:
    """The documented estimator: UTF-8 bytes divided by 4, rounded up."""
    return -(-len(text.encode("utf-8")) // 4)
