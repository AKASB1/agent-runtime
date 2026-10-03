---
name: optimization-modelling
description: Write a linear or mixed-integer formulation in the JSON format of the optimize workflow.
tools: [opt.solve, opt.verify, opt.check_report]
---
Write one JSON object with "sense" ("min" or "max"), "variables" (name, lb, ub, type), "objective" (terms, constant), and "constraints" (name, terms, sense, rhs).
Use the card's variable names for every decision variable; add auxiliary variables only when needed.
Write each limit of the statement as its own constraint and check its direction (<= for capacities, >= for requirements).
Use type "binary" for yes/no decisions and "integer" for counts; keep "continuous" otherwise.
When the verifier rejects a formulation, fix the constraint its diagnostic names and change nothing else.
