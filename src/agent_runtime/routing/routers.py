"""Routing decisions. ``choose(ctx) -> Decision(target, reason)``; ``observe(outcome)`` follows
each call; ``escalate(ctx)`` answers whether a task that was rejected or failed gets another
attempt (only the cascade says yes)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_runtime.rand import stream


@dataclass(frozen=True)
class AttemptOutcome:
    """What happened to a finished attempt of this task (only what the runtime knows)."""

    attempt: int
    target: str
    status: str  # succeeded | failed | ...
    accepted: bool  # verifier acceptance and validations, never the checker's success
    model_calls: int


@dataclass(frozen=True)
class RouteContext:
    step_kind: str  # the role of the call: act, plan, replan, answer, formulate, summarize, ...
    candidates: tuple[str, ...]
    spent_cost_ucu: int
    elapsed_ms: int
    calls: int
    attempt: int
    history: tuple[AttemptOutcome, ...]
    hint: float | None
    task_id: str
    deadline_ms: int | None
    #: public facts about the candidates (price, window, median latency); never outcomes
    specs: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    target: str
    reason: str


class Router(Protocol):
    name: str

    def choose(self, ctx: RouteContext) -> Decision | None: ...

    def observe(self, outcome: Any) -> None: ...

    def escalate(self, ctx: RouteContext) -> bool: ...


class _Base:
    name = "base"

    def observe(self, outcome: Any) -> None:
        return None

    def escalate(self, ctx: RouteContext) -> bool:
        return False

    @staticmethod
    def _first_candidate(prefs: list[str], ctx: RouteContext, why: str) -> Decision | None:
        """The first preference that is offered; None when none is (the step then fails)."""
        for t in prefs:
            if t in ctx.candidates:
                return Decision(t, why if t == prefs[0] else f"{why}; {prefs[0]} not offered, fallback {t}")
        return None


class FixedRouter(_Base):
    name = "fixed"

    def __init__(self, target: str, fallback: list[str] | None = None) -> None:
        self.target = target
        self.fallback = list(fallback or [])

    def choose(self, ctx: RouteContext) -> Decision | None:
        return self._first_candidate([self.target, *self.fallback], ctx, f"fixed({self.target})")


class RandomMixRouter(_Base):
    """Per task (and attempt) one draw with fixed probabilities from stream ``route/<task_id>``."""

    name = "random_mix"

    def __init__(self, probs: dict[str, float], seed: int, fallback: list[str] | None = None) -> None:
        total = sum(probs.values())
        if total <= 0:
            raise ValueError("probabilities must sum to a positive number")
        # an explicit order (by target) so that the draw does not depend on how the dict was written
        self.probs = [(t, probs[t] / total) for t in sorted(probs)]
        self.seed = seed
        self.fallback = list(fallback or [])
        self._cache: dict[tuple[str, int], str] = {}

    def choose(self, ctx: RouteContext) -> Decision | None:
        key = (ctx.task_id, ctx.attempt)
        if key not in self._cache:
            u = stream(self.seed, f"route/{ctx.task_id}/{ctx.attempt}").random()
            acc = 0.0
            pick = self.probs[-1][0]
            for t, p in self.probs:
                acc += p
                if u < acc:
                    pick = t
                    break
            self._cache[key] = pick
        pick = self._cache[key]
        return self._first_candidate([pick, *self.fallback], ctx, f"random_mix draw -> {pick}")


class CascadeRouter(_Base):
    """Attempt ``a`` runs on ``chain[a-1]``; a rejected or failed attempt escalates to the next model
    unless the attempt cannot finish before the deadline at the model's median latency."""

    name = "cascade"

    def __init__(self, chain: list[str], *, budget_aware: bool = True, est_tokens: int = 100) -> None:
        if not chain:
            raise ValueError("empty chain")
        self.chain = list(chain)
        self.budget_aware = budget_aware
        self.est_tokens = est_tokens

    def choose(self, ctx: RouteContext) -> Decision | None:
        idx = min(ctx.attempt - 1, len(self.chain) - 1)
        return self._first_candidate(
            [self.chain[idx], *self.chain[idx + 1 :]], ctx, f"cascade step {idx + 1}"
        )

    def escalate(self, ctx: RouteContext) -> bool:
        if ctx.attempt >= len(self.chain):
            return False
        if not self.budget_aware or ctx.deadline_ms is None:
            return True
        nxt = self.chain[ctx.attempt]
        spec = ctx.specs.get(nxt, {})
        median_call = spec.get("latency_base_ms", 0) + spec.get("latency_per_token_ms", 0) * self.est_tokens
        calls = ctx.history[-1].model_calls if ctx.history else 1
        return ctx.elapsed_ms + max(calls, 1) * median_call <= ctx.deadline_ms


class ThresholdRouter(_Base):
    """``hint < t1`` -> small, ``hint < t2`` -> medium, otherwise large."""

    name = "threshold"

    def __init__(self, t1: float, t2: float, models: list[str], fallback: list[str] | None = None) -> None:
        if len(models) != 3:
            raise ValueError("threshold needs three models (small, medium, large)")
        self.t1, self.t2 = t1, t2
        self.models = list(models)
        self.fallback = list(fallback or [])

    def choose(self, ctx: RouteContext) -> Decision | None:
        h = ctx.hint if ctx.hint is not None else self.t2
        m = self.models[0] if h < self.t1 else self.models[1] if h < self.t2 else self.models[2]
        return self._first_candidate(
            [m, *self.fallback], ctx, f"threshold hint={h:.3f} t1={self.t1} t2={self.t2}"
        )


def make_router(cfg: dict[str, Any] | None, *, seed: int, default_target: str) -> Router:
    """Build a router from configuration ``{"type": ..., ...}``."""
    cfg = dict(cfg or {})
    kind = cfg.pop("type", "fixed")
    if kind == "fixed":
        return FixedRouter(cfg.get("target", default_target), cfg.get("fallback"))
    if kind == "random_mix":
        return RandomMixRouter(cfg["probs"], seed, cfg.get("fallback"))
    if kind == "cascade":
        return CascadeRouter(cfg["chain"], budget_aware=cfg.get("budget_aware", True))
    if kind == "threshold":
        return ThresholdRouter(cfg["t1"], cfg["t2"], cfg["models"], cfg.get("fallback"))
    raise ValueError(f"unknown router type {kind!r}")
