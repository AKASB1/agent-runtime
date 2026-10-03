"""Seeded random streams: one stream per component, named so that results do not depend on
interleaving. ``random.Random(splitmix64(seed XOR fnv1a64(name)))``; only ``random()`` and
``getrandbits()`` are used (their output is stable across CPython versions)."""

from __future__ import annotations

import math
import random

MASK64 = (1 << 64) - 1


def fnv1a64(name: str) -> int:
    h = 0xCBF29CE484222325
    for b in name.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & MASK64
    return h


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def stream(seed: int, name: str) -> random.Random:
    """The stream of component ``name`` under ``seed``."""
    return random.Random(splitmix64((seed & MASK64) ^ fnv1a64(name)))


def normal(rng: random.Random) -> float:
    """Standard normal draw by Box-Muller from two ``random()`` values."""
    u1 = rng.random()
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)


def lognormal(rng: random.Random, sigma: float) -> float:
    """Lognormal factor with median 1."""
    return math.exp(sigma * normal(rng))


def below(rng: random.Random, n: int) -> int:
    """Uniform integer in [0, n) from ``random()`` (no ``randrange``/``choice``)."""
    if n <= 0:
        raise ValueError("n must be positive")
    return min(n - 1, int(rng.random() * n))
