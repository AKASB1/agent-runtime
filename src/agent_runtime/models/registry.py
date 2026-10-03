"""Model registry: targets (provider, model) with prices, windows, and capabilities."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Wire = Literal["sim", "scripted", "openai_chat", "anthropic_messages"]
DIALECTS: tuple[str, ...] = ("openai", "deepseek")


class ModelSpec(BaseModel):
    id: str
    provider: str
    wire: Wire
    dialect: str | None = None
    price_in_ucu: int = 0  # micro-cu per million tokens
    price_out_ucu: int = 0
    price_cache_read_ucu: int | None = None  # defaults to price_in_ucu
    price_cache_write_ucu: int | None = None  # defaults to price_in_ucu
    context_tokens: int = 8192
    capabilities: list[str] = Field(default_factory=lambda: ["tools", "json_schema"])
    priced: bool = True
    #: median latency model of a simulated target (base + per output token), for budget-aware stops
    latency_base_ms: int = 0
    latency_per_token_ms: int = 0

    @model_validator(mode="after")
    def _defaults(self) -> ModelSpec:
        if self.price_cache_read_ucu is None:
            self.price_cache_read_ucu = self.price_in_ucu
        if self.price_cache_write_ucu is None:
            self.price_cache_write_ucu = self.price_in_ucu
        if self.wire == "openai_chat":
            if self.dialect is None:
                self.dialect = "openai"
            if self.dialect not in DIALECTS:
                raise ValueError(f"unknown dialect {self.dialect!r} (known: {', '.join(DIALECTS)})")
        return self

    @property
    def target(self) -> str:
        return f"{self.provider}/{self.id}"


class ModelRegistry:
    """Targets by ``provider/model``; iteration order is registration order (explicit)."""

    def __init__(self, specs: list[ModelSpec] | None = None) -> None:
        self._specs: dict[str, ModelSpec] = {}
        for s in specs or []:
            self.add(s)

    def add(self, spec: ModelSpec) -> None:
        if spec.target in self._specs:
            raise ValueError(f"duplicate target {spec.target}")
        self._specs[spec.target] = spec

    def get(self, target: str) -> ModelSpec:
        try:
            return self._specs[target]
        except KeyError:
            raise KeyError(f"unknown target {target!r}") from None

    def __contains__(self, target: object) -> bool:
        return target in self._specs

    def targets(self) -> list[str]:
        return list(self._specs)

    def specs(self) -> list[ModelSpec]:
        return list(self._specs.values())
