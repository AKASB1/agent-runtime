"""The seeded world (``world_seed`` 1: 200 customers, 600 products, 2000 orders, a fixed currency
table) and its tools. Tool results are pure functions of the world and the arguments; virtual
latencies are fixed per tool times a lognormal factor (median 1, sigma 0.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, ConfigDict

from agent_runtime.rand import below, lognormal, stream
from agent_runtime.tools import Tool, ToolError

COUNTRIES = [("DE", "EUR"), ("FR", "EUR"), ("GB", "GBP"), ("US", "USD"), ("JP", "JPY"), ("CN", "CNY")]
RATES_TO_USD = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "JPY": 0.0067, "CNY": 0.138}
CATEGORIES = ["tools", "toys", "books", "garden", "kitchen", "office"]
TOOL_LATENCY_MS = {
    "get_customer": 300,
    "list_orders": 350,
    "get_order": 250,
    "get_product": 250,
    "convert": 100,
    "calc": 20,
    "send_report": 400,
}
READ_TOOLS = ("get_customer", "list_orders", "get_order", "get_product", "convert", "calc")


@dataclass(frozen=True)
class World:
    customers: dict[str, dict[str, Any]]
    products: dict[str, dict[str, Any]]
    orders: dict[str, dict[str, Any]]
    orders_by_customer: dict[str, list[str]]


@lru_cache(maxsize=4)
def make_world(world_seed: int = 1) -> World:
    rng = stream(world_seed, "world")
    customers = {}
    for i in range(1, 201):
        cid = f"C{i:03d}"
        country, cur = COUNTRIES[below(rng, len(COUNTRIES))]
        customers[cid] = {
            "id": cid,
            "name": f"Customer {i}",
            "country": country,
            "currency": cur,
            "segment": ["retail", "business"][below(rng, 2)],
        }
    products = {}
    for i in range(1, 601):
        pid = f"P{i:03d}"
        cur = "USD"  # all list prices are in USD; orders are converted to the customer's currency
        price = round(2.0 + rng.random() * 198.0, 2)
        products[pid] = {
            "id": pid,
            "name": f"Product {i}",
            "price": price,
            "currency": cur,
            "category": CATEGORIES[below(rng, len(CATEGORIES))],
        }
    orders = {}
    by_cust: dict[str, list[str]] = {c: [] for c in customers}
    for i in range(1, 2001):
        oid = f"O{i:04d}"
        cid = f"C{below(rng, 200) + 1:03d}"
        n_items = 1 + below(rng, 6)
        items = []
        used = set()
        for _ in range(n_items):
            p = f"P{below(rng, 600) + 1:03d}"
            if p in used:
                continue
            used.add(p)
            items.append({"product_id": p, "qty": 1 + below(rng, 5)})
        orders[oid] = {"id": oid, "customer_id": cid, "items": items, "currency": customers[cid]["currency"]}
        by_cust[cid].append(oid)
    return World(customers, products, orders, by_cust)


# -- tool models (strict: validation always catches an invalid call) -----------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CustomerIn(_Strict):
    customer_id: str


class CustomerOut(BaseModel):
    id: str
    name: str
    country: str
    currency: str
    segment: str


class ListOrdersIn(_Strict):
    customer_id: str


class ListOrdersOut(BaseModel):
    customer_id: str
    order_ids: list[str]


class OrderIn(_Strict):
    order_id: str


class Item(BaseModel):
    product_id: str
    qty: int


class OrderOut(BaseModel):
    id: str
    customer_id: str
    items: list[Item]
    currency: str


class ProductIn(_Strict):
    product_id: str


class ProductOut(BaseModel):
    id: str
    name: str
    price: float
    currency: str
    category: str


class ConvertIn(_Strict):
    amount: float
    from_currency: str
    to_currency: str


class AmountOut(BaseModel):
    amount: float


class CalcIn(_Strict):
    op: str  # sum | count | max
    values: list[float]


class ValueOut(BaseModel):
    value: float


class ReportIn(_Strict):
    customer_id: str
    subject: str
    value: str


class ReportOut(BaseModel):
    report_id: str


def convert_amount(amount: float, frm: str, to: str) -> float:
    return round(amount * RATES_TO_USD[frm] / RATES_TO_USD[to], 2)


def calc_value(op: str, values: list[float]) -> float:
    if op == "sum":
        return round(sum(values), 2)
    if op == "count":
        return float(len(values))
    if op == "max":
        return max(values) if values else 0.0
    raise ToolError("NOT_FOUND", f"no operation {op!r}")


def pure_call(world: World, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """The result of a read tool (raises ToolError NOT_FOUND)."""
    if tool == "get_customer":
        c = world.customers.get(args["customer_id"])
        if c is None:
            raise ToolError("NOT_FOUND", f"no customer {args['customer_id']}")
        return dict(c)
    if tool == "list_orders":
        if args["customer_id"] not in world.customers:
            raise ToolError("NOT_FOUND", f"no customer {args['customer_id']}")
        return {
            "customer_id": args["customer_id"],
            "order_ids": world.orders_by_customer[args["customer_id"]][:10],
        }
    if tool == "get_order":
        o = world.orders.get(args["order_id"])
        if o is None:
            raise ToolError("NOT_FOUND", f"no order {args['order_id']}")
        return {**o, "items": [dict(i) for i in o["items"]]}
    if tool == "get_product":
        p = world.products.get(args["product_id"])
        if p is None:
            raise ToolError("NOT_FOUND", f"no product {args['product_id']}")
        return dict(p)
    if tool == "convert":
        for k in ("from_currency", "to_currency"):
            if args[k] not in RATES_TO_USD:
                raise ToolError("NOT_FOUND", f"no currency {args[k]}")
        return {"amount": convert_amount(args["amount"], args["from_currency"], args["to_currency"])}
    if tool == "calc":
        return {"value": calc_value(args["op"], list(args["values"]))}
    raise ToolError("UNKNOWN_TOOL", tool)


TOOL_MODELS = {
    "get_customer": (CustomerIn, CustomerOut, "Look up a customer by id (C001..)."),
    "list_orders": (ListOrdersIn, ListOrdersOut, "List the order ids of a customer (at most 10)."),
    "get_order": (OrderIn, OrderOut, "Look up an order by id (O0001..)."),
    "get_product": (ProductIn, ProductOut, "Look up a product by id (P001..)."),
    "convert": (ConvertIn, AmountOut, "Convert an amount between currencies."),
    "calc": (CalcIn, ValueOut, "Compute sum, count, or max of a list of numbers."),
    "send_report": (ReportIn, ReportOut, "Send a report to a customer (a write; never repeat it)."),
}


@dataclass
class WorldSession:
    """The world as one run sees it: latencies on the run's clock, transient faults, the write log."""

    world: World
    clock: Any
    seed: int
    run_id: str
    fail_rate: float = 0.0
    sigma: float = 0.2
    writes: list[dict[str, Any]] = field(default_factory=list)
    calls: int = 0

    def tools(self) -> list[Tool]:
        out = []
        for name, (tin, tout, desc) in TOOL_MODELS.items():
            out.append(
                Tool(
                    name=name,
                    handler=self._handler(name),
                    input_model=tin,
                    output_model=tout,
                    description=desc,
                    side_effect="write" if name == "send_report" else "read",
                    capability="none",
                    timeout_ms=10_000,
                )
            )
        return out

    def _handler(self, name: str):
        async def handler(value: BaseModel) -> dict[str, Any]:
            self.calls += 1
            n = self.calls
            rng = stream(self.seed, f"tool/{self.run_id}/{n}")
            await self.clock.sleep(int(round(TOOL_LATENCY_MS[name] * lognormal(rng, self.sigma))))
            if self.fail_rate > 0 and name in READ_TOOLS:
                if stream(self.seed, f"toolfault/{self.run_id}/{n}").random() < self.fail_rate:
                    raise ToolError("UNAVAILABLE", "transient failure")
            args = value.model_dump()
            if name == "send_report":
                if args["customer_id"] not in self.world.customers:
                    raise ToolError("NOT_FOUND", f"no customer {args['customer_id']}")
                self.writes.append(args)
                return {"report_id": f"R{len(self.writes)}"}
            return pure_call(self.world, name, args)

        return handler
