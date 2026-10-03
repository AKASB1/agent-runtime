"""Task generator ``gen(seed, n, mix, shape)``: tasks from templates (a lookup, a join, an
aggregate over items, and any of them followed by ``send_report`` for 20% of the tasks), each with
a gold plan (a DAG of tool calls with literal arguments), a gold answer, a tier, and a difficulty
hint. Tier by number of steps: easy 1-2, medium 3-5, hard 6-8 (``wide`` tasks have 4-10 steps; 9-10
count as hard). The tasks are far simpler than real agent tasks; they exercise the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_runtime.rand import below, normal, stream
from agent_runtime.sim.world import COUNTRIES, World, calc_value, convert_amount, make_world

MIXES: dict[str, tuple[float, float, float]] = {"mixed": (0.50, 0.35, 0.15), "hard_heavy": (0.15, 0.35, 0.50)}
SHAPES = ("chain", "wide")
TIERS = ("easy", "medium", "hard")


@dataclass(frozen=True)
class GoldStep:
    id: str
    tool: str
    args: dict[str, Any]
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class Task:
    id: str
    text: str
    tier: str
    steps: tuple[GoldStep, ...]
    answer: str
    hint: float
    report: bool
    shape: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def tier_index(self) -> int:
        return TIERS.index(self.tier) + 1

    def step(self, sid: str) -> GoldStep:
        for s in self.steps:
            if s.id == sid:
                return s
        raise KeyError(sid)

    def sinks(self) -> list[str]:
        needed = {d for s in self.steps for d in s.depends_on}
        return [s.id for s in self.steps if s.id not in needed]


def tier_of(n_steps: int) -> str:
    return "easy" if n_steps <= 2 else "medium" if n_steps <= 5 else "hard"


def _order_with(world: World, rng: Any, min_items: int) -> dict[str, Any]:
    while True:
        o = world.orders[f"O{below(rng, 2000) + 1:04d}"]
        if len(o["items"]) >= min_items:
            return o


def _customer_with(world: World, rng: Any, min_orders: int, min_items: int = 1) -> str:
    while True:
        c = f"C{below(rng, 200) + 1:03d}"
        oids = world.orders_by_customer[c][:10]
        if len(oids) >= min_orders and all(
            len(world.orders[o]["items"]) >= min_items for o in oids[:min_orders]
        ):
            return c


def _fmt(amount: float, cur: str) -> str:
    return f"{amount:.2f} {cur}"


class _Builder:
    def __init__(self) -> None:
        self.steps: list[GoldStep] = []

    def add(self, tool: str, args: dict[str, Any], deps: list[str]) -> str:
        sid = f"s{len(self.steps) + 1}"
        self.steps.append(GoldStep(sid, tool, args, tuple(deps)))
        return sid


def _order_value(
    world: World, rng: Any, b: _Builder, m: int, convert: bool, chain: bool
) -> tuple[str, str, str, str]:
    """get_order -> get_product x m -> calc(sum) [-> convert]; returns (answer, customer, last step, text)."""
    o = _order_with(world, rng, m)
    s_order = b.add("get_order", {"order_id": o["id"]}, [])
    prev = s_order
    prod_steps = []
    values = []
    for it in o["items"][:m]:
        p = world.products[it["product_id"]]
        sid = b.add("get_product", {"product_id": p["id"]}, [prev if chain else s_order])
        prod_steps.append(sid)
        prev = sid
        values.append(round(p["price"] * it["qty"], 2))
    total = calc_value("sum", values)
    s_calc = b.add("calc", {"op": "sum", "values": values}, prod_steps)
    last, answer = s_calc, _fmt(total, "USD")
    if convert:
        amount = convert_amount(total, "USD", o["currency"])
        last = b.add(
            "convert", {"amount": total, "from_currency": "USD", "to_currency": o["currency"]}, [s_calc]
        )
        answer = _fmt(amount, o["currency"])
    text = f"What is the value of the first {m} item(s) of order {o['id']} in {o['currency'] if convert else 'USD'}?"
    return answer, o["customer_id"], last, text


def _multi_order(world: World, rng: Any, b: _Builder, w: int, convert: bool) -> tuple[str, str, str, str]:
    """(get_order -> get_product of its first item) x w, independent branches -> calc [-> convert]."""
    c = _customer_with(world, rng, w)
    prods, values = [], []
    oids = world.orders_by_customer[c][:w]
    for oid in oids:
        o = world.orders[oid]
        s_o = b.add("get_order", {"order_id": oid}, [])
        it = o["items"][0]
        p = world.products[it["product_id"]]
        prods.append(b.add("get_product", {"product_id": p["id"]}, [s_o]))
        values.append(round(p["price"] * it["qty"], 2))
    total = calc_value("sum", values)
    s_calc = b.add("calc", {"op": "sum", "values": values}, prods)
    cur = world.customers[c]["currency"]
    last, answer = s_calc, _fmt(total, "USD")
    if convert:
        last = b.add("convert", {"amount": total, "from_currency": "USD", "to_currency": cur}, [s_calc])
        answer = _fmt(convert_amount(total, "USD", cur), cur)
    return (
        answer,
        c,
        last,
        f"What is the value of the first item of each of the orders {', '.join(oids)} of customer {c}?",
    )


def _core_chain(world: World, rng: Any, b: _Builder, k: int) -> tuple[str, str, str, str]:
    if k == 1:
        c = f"C{below(rng, 200) + 1:03d}"
        sid = b.add("get_customer", {"customer_id": c}, [])
        return world.customers[c]["country"], c, sid, f"Which country is customer {c} in?"
    if k == 2:
        o = _order_with(world, rng, 1)
        s1 = b.add("get_order", {"order_id": o["id"]}, [])
        p = world.products[o["items"][0]["product_id"]]
        s2 = b.add("get_product", {"product_id": p["id"]}, [s1])
        return (
            _fmt(p["price"], p["currency"]),
            o["customer_id"],
            s2,
            f"What is the unit price of the first product of order {o['id']}?",
        )
    if k == 3:
        return _order_value(world, rng, b, 1, False, True)
    return _order_value(world, rng, b, k - 3, True, True)


def _core_wide(world: World, rng: Any, b: _Builder, k: int) -> tuple[str, str, str, str]:
    table = {
        4: ("o", 2, False),
        5: ("m", 2, False),
        6: ("m", 2, True),
        7: ("m", 3, False),
        8: ("m", 3, True),
        9: ("m", 4, False),
        10: ("m", 4, True),
    }
    kind, w, conv = table[k]
    if kind == "o":
        return _order_value(world, rng, b, w, conv, False)
    return _multi_order(world, rng, b, w, conv)


def gen(
    seed: int,
    n: int,
    mix: str = "mixed",
    shape: str = "chain",
    *,
    hint_sigma: float = 0.6,
    world_seed: int = 1,
) -> list[Task]:
    if mix not in MIXES:
        raise ValueError(f"unknown mix {mix!r}")
    if shape not in SHAPES:
        raise ValueError(f"unknown shape {shape!r}")
    world = make_world(world_seed)
    rng = stream(seed, f"gen/{mix}/{shape}")
    p_easy, p_med, _ = MIXES[mix]
    tasks = []
    for i in range(n):
        u = rng.random()
        drawn = "easy" if u < p_easy else "medium" if u < p_easy + p_med else "hard"
        report = rng.random() < 0.2
        if shape == "chain":
            k = {"easy": 1 + below(rng, 2), "medium": 3 + below(rng, 3), "hard": 6 + below(rng, 3)}[drawn]
            if report and k == 1:
                k = 2
            core = k - 1 if report else k
            b = _Builder()
            answer, cust, last, text = _core_chain(world, rng, b, core)
        else:
            k = {"easy": 4, "medium": 5, "hard": 6 + below(rng, 5)}[drawn]
            core = k - 1 if report else k
            core = max(4, min(core, 10))  # a report task has at least 5 steps (4 + the report)
            b = _Builder()
            answer, cust, last, text = _core_wide(world, rng, b, core)
        tid = f"t{i:04d}"
        if report:
            b.add("send_report", {"customer_id": cust, "subject": f"report {tid}", "value": answer}, [last])
            text += f" Then send the result to customer {cust} as report '{tid}'."
        steps = tuple(b.steps)
        tier = tier_of(len(steps))
        hint = TIERS.index(tier) + 1 + hint_sigma * normal(stream(seed, f"task/{tid}/hint"))
        tasks.append(Task(tid, text, tier, steps, answer, round(hint, 6), report, shape))
    return tasks


def perturb(answer: str) -> str:
    """A wrong final answer: a perturbed copy of the gold answer."""
    parts = answer.split(" ")
    try:
        value = float(parts[0])
        return " ".join([f"{value * 1.1 + 0.01:.2f}", *parts[1:]])
    except ValueError:
        codes = [c for c, _ in COUNTRIES]
        return codes[(codes.index(answer) + 1) % len(codes)] if answer in codes else answer + "?"
