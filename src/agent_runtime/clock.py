"""Clocks and the virtual-time event loop.

Domain code reads time only through a :class:`Clock` (integer milliseconds since the start of
the run or the process) and waits only through it. This module is the only place in the package
that touches the wall clock.

The virtual-time loop is an ``asyncio.SelectorEventLoop`` whose ``time()`` returns the virtual
time and whose selector never blocks: when the loop would wait for the next timer, the selector
advances the virtual time to that timer instead. Timers that are due at the same virtual time run
in the order (time, sequence number), so a run never depends on scheduling luck. The loop supports
neither real sockets beyond its self-pipe nor subprocesses.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import selectors
import time
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, Protocol, TypeVar

T = TypeVar("T")


class Clock(Protocol):
    """Integer milliseconds; waiting goes through the clock."""

    virtual: bool

    def now_ms(self) -> int: ...

    async def sleep(self, ms: int) -> None: ...

    async def wait_for(self, aw: Awaitable[T], timeout_ms: int | None) -> T: ...


class SystemClock:
    """Monotonic wall clock, optionally scaled (``scale`` virtual ms per real ms)."""

    virtual = False

    def __init__(self, scale: float = 1.0) -> None:
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.scale = scale
        self._t0 = time.monotonic()

    def now_ms(self) -> int:
        return int((time.monotonic() - self._t0) * 1000.0 * self.scale)

    def _seconds(self, ms: int) -> float:
        return max(0.0, ms / 1000.0 / self.scale)

    async def sleep(self, ms: int) -> None:
        await asyncio.sleep(self._seconds(ms))

    async def wait_for(self, aw: Awaitable[T], timeout_ms: int | None) -> T:
        if timeout_ms is None:
            return await aw
        return await asyncio.wait_for(aw, self._seconds(timeout_ms))


def wall_ms() -> float:
    """Wall-clock milliseconds for ``wall_`` measurements only (never for domain decisions)."""
    return time.perf_counter() * 1000.0


class _SeqTimerHandle(asyncio.TimerHandle):
    """Timer ordered by (when, sequence) so that equal-time timers fire in creation order."""

    __slots__ = ("_seq",)

    def __init__(self, when, callback, args, loop, context, seq):  # noqa: ANN001
        super().__init__(when, callback, args, loop, context)
        self._seq = seq

    def __lt__(self, other):  # noqa: ANN001
        if self._when == other._when:
            return self._seq < getattr(other, "_seq", 0)
        return self._when < other._when

    def __le__(self, other):  # noqa: ANN001
        return self < other or self is other

    def __gt__(self, other):  # noqa: ANN001
        return other < self

    def __ge__(self, other):  # noqa: ANN001
        return other < self or self is other


class _VirtualSelector(selectors.BaseSelector):
    """Wraps the real selector; instead of blocking it advances the loop's virtual time."""

    def __init__(self) -> None:
        self._real = selectors.DefaultSelector()
        self.loop: VirtualTimeLoop | None = None

    def register(self, fileobj, events, data=None):  # noqa: ANN001
        return self._real.register(fileobj, events, data)

    def unregister(self, fileobj):  # noqa: ANN001
        return self._real.unregister(fileobj)

    def modify(self, fileobj, events, data=None):  # noqa: ANN001
        return self._real.modify(fileobj, events, data)

    def get_map(self):
        return self._real.get_map()

    def get_key(self, fileobj):  # noqa: ANN001
        return self._real.get_key(fileobj)

    def close(self) -> None:
        self._real.close()

    def select(self, timeout=None):  # noqa: ANN001
        ready = self._real.select(0)
        if ready or timeout == 0:
            return ready
        loop = self.loop
        assert loop is not None
        scheduled = loop._scheduled  # noqa: SLF001
        if not scheduled:
            raise RuntimeError("virtual-time loop has nothing scheduled and would block forever")
        head = scheduled[0]
        target = round(head._when * 1000.0)  # noqa: SLF001
        if target > loop.clock._now:  # noqa: SLF001
            loop.clock._now = target  # noqa: SLF001
        return []


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    """Event loop on virtual time; see the module docstring."""

    def __init__(self) -> None:
        selector = _VirtualSelector()
        super().__init__(selector)
        selector.loop = self
        self._timer_seq = itertools.count()
        # the wall clock's resolution (15.6 ms on Windows) must not pop virtual timers early
        self._clock_resolution = 1e-7
        self.clock = VirtualClock(self)

    def time(self) -> float:
        return self.clock._now / 1000.0  # noqa: SLF001

    def call_at(self, when, callback, *args, context=None):  # noqa: ANN001
        if when is None:
            raise TypeError("when cannot be None")
        self._check_closed()
        timer = _SeqTimerHandle(when, callback, args, self, context, next(self._timer_seq))
        heapq.heappush(self._scheduled, timer)
        timer._scheduled = True  # noqa: SLF001
        return timer


class VirtualClock:
    """The clock of a :class:`VirtualTimeLoop`; ``sleep`` schedules on exact integer milliseconds."""

    virtual = True

    def __init__(self, loop: VirtualTimeLoop) -> None:
        self._loop = loop
        self._now = 0

    def now_ms(self) -> int:
        return self._now

    async def sleep(self, ms: int) -> None:
        ms = int(ms)
        if ms <= 0:
            return  # no yield: a zero-length wait never reorders tasks (replay relies on it)
        fut = self._loop.create_future()
        handle = self._loop.call_at((self._now + ms) / 1000.0, _resolve, fut)
        try:
            await fut
        finally:
            handle.cancel()

    async def wait_for(self, aw: Awaitable[T], timeout_ms: int | None) -> T:
        if timeout_ms is None:
            return await aw
        return await asyncio.wait_for(aw, timeout_ms / 1000.0)


def _resolve(fut: asyncio.Future) -> None:
    if not fut.done():
        fut.set_result(None)


def run_virtual(main: Callable[[VirtualClock], Coroutine[Any, Any, T]]) -> T:
    """Run ``main(clock)`` to completion on a fresh virtual-time loop."""
    loop = VirtualTimeLoop()
    try:
        return loop.run_until_complete(main(loop.clock))
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            loop.close()
