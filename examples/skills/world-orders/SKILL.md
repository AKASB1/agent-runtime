---
name: world-orders
description: Answer questions about orders, products, and their value with the simulated world's tools.
tools: [get_order, get_product, calc, convert, send_report]
---
1. Look up the order with get_order and read its items (product ids and quantities).
2. Look up each product you need with get_product; list prices are in USD.
3. Add up price times quantity with calc (op "sum"); convert the total with convert only when the question asks for another currency.
4. Send a report with send_report only when the task asks for it, exactly once, with the final value.
5. Answer with the number and the currency, for example "123.45 EUR".
