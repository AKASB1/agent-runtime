import asyncio
import pathlib
import re

from agent_runtime.clock import SystemClock, run_virtual
from agent_runtime.rand import below, fnv1a64, lognormal, normal, splitmix64, stream


def test_fnv1a64_known_answers():
    assert fnv1a64("") == 0xCBF29CE484222325
    assert fnv1a64("a") == 0xAF63DC4C8601EC8C
    assert fnv1a64("foobar") == 0x85944171F73967E8


def test_splitmix64_known_answers():
    assert splitmix64(0) == 0xE220A8397B1DCDAF
    # the standard SplitMix64 sequence from state 0: state advances by the golden gamma
    assert splitmix64(0x9E3779B97F4A7C15) == 0x6E789E6AA1B965F4


def test_streams_are_reproducible_and_independent():
    a1 = [stream(7, "sim/a").random() for _ in range(3)]
    a2 = [stream(7, "sim/a").random() for _ in range(3)]
    assert a1 == a2
    a = stream(7, "sim/a")
    b = stream(7, "sim/b")
    xs = [a.random() for _ in range(200)]
    ys = [b.random() for _ in range(200)]
    assert xs != ys
    # drawing from b does not change a: interleaving independence
    a_again = stream(7, "sim/a")
    _ = [stream(7, "sim/b").random() for _ in range(50)]
    assert [a_again.random() for _ in range(200)] == xs
    # different seeds give different streams
    assert stream(8, "sim/a").random() != stream(7, "sim/a").random()


def test_normal_and_lognormal_moments():
    rng = stream(1, "moments")
    zs = [normal(rng) for _ in range(20000)]
    mean = sum(zs) / len(zs)
    var = sum((z - mean) ** 2 for z in zs) / len(zs)
    assert abs(mean) < 0.03 and abs(var - 1.0) < 0.05
    ls = sorted(lognormal(rng, 0.2) for _ in range(20001))
    assert abs(ls[10000] - 1.0) < 0.02
    assert all(0 <= below(rng, 5) < 5 for _ in range(1000))


def test_thousand_virtual_timers_fire_in_order():
    fired: list[tuple[int, int]] = []

    async def main(clock):
        async def sleeper(i: int, ms: int):
            await clock.sleep(ms)
            fired.append((clock.now_ms(), i))

        rng = stream(3, "timers")
        delays = [below(rng, 5000) + 1 for _ in range(1000)]
        await asyncio.gather(*(sleeper(i, d) for i, d in enumerate(delays)))
        return delays

    delays = run_virtual(main)
    times = [t for t, _ in fired]
    assert times == sorted(times)
    assert sorted(times) == sorted(delays)
    # each sleeper woke exactly at its own virtual time
    assert all(t == delays[i] for t, i in fired)


def test_equal_time_wakeups_run_in_sequence_order():
    order: list[int] = []

    async def main(clock):
        async def sleeper(i: int):
            await clock.sleep(100)
            order.append(i)

        tasks = [asyncio.ensure_future(sleeper(i)) for i in range(50)]
        await asyncio.gather(*tasks)
        return clock.now_ms()

    assert run_virtual(main) == 100
    assert order == list(range(50))


def test_virtual_wait_for_times_out_on_virtual_time():
    async def main(clock):
        async def slow():
            await clock.sleep(10_000)
            return "late"

        try:
            await clock.wait_for(slow(), 2500)
        except TimeoutError:
            return clock.now_ms()
        return -1

    assert run_virtual(main) == 2500


def test_system_clock_is_monotonic_and_scaled():
    c = SystemClock(scale=50)
    a = c.now_ms()
    asyncio.run(c.sleep(500))  # 10 ms of real time
    assert c.now_ms() >= a + 400


def test_domain_code_never_reads_the_wall_clock():
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "agent_runtime"
    pattern = re.compile(r"\btime\.(time|monotonic|perf_counter|sleep)\(|datetime\.now\(|asyncio\.sleep\(")
    allowed = {"clock.py", "standin.py", "testserver.py"}
    offenders = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if path.name in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        for m in pattern.finditer(text):
            offenders.append(f"{rel}: {m.group(0)}")
    assert offenders == []
