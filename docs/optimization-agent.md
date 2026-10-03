# The worked example: an optimization-modelling agent

A model formulates, a solver solves, a verifier checks. The workflow (`optimize`) is an ordinary workflow of this runtime, built from its public API; the solver and the verifier are tools of an MCP server (`python -m agent_runtime.optimization.server`). **The model is scripted**: each problem has a fixed sequence of candidate formulations in `examples/optimization/scripts/` (data). The example shows the runtime and the verifier at work and says nothing about any model's modelling ability.

## Flow

```text
formulate (model, structured output = formulation JSON)
   -> opt.solve (SciPy milp / HiGHS)
   -> opt.verify (checks V1-V4)
        rejected: the diagnostics go back as a logged user message, new formulation (at most 2 repairs)
   -> report (model, structured output {objective, values})
   -> opt.check_report (check V5)
        mismatch: one re-ask
```

A malformed formulation (not JSON, wrong shape) is caught by the structured-output validation of the run engine and repaired there; its message carries the code `V1_PARSE`.

## Problem cards

`examples/optimization/problems/<id>.json`: `id`, `kind`, `statement` (at most 150 words of English), `data`, `variables` (the names the formulation must use for its decision variables; auxiliary variables may be added), `witnesses` (feasible and infeasible points over the card's variables, each with a note), and `tolerances` (feasibility 1e-6, integrality 1e-6, objective 1e-6 absolute and relative, report 1e-4). The reference formulations in `examples/optimization/reference/` are used only by the tests and by E4; a test checks that the sources of the workflow, the server, and the verifier do not refer to the reference folder.

| Problem | Kind | Scripted candidates (in order) | Expected diagnostic |
|---|---|---|---|
| `product_mix` | LP | missing constraint (labour hours); reversed inequality (machine hours >= 100); correct. Reports: misreported objective; correct | `V4_INFEASIBLE_WITNESS_ACCEPTED`; `V4_FEASIBLE_WITNESS_REJECTED`; -. Report: `V5_REPORT_MISMATCH` |
| `diet` | LP | coefficient off by a factor of 10 (egg protein 130); malformed JSON (truncated); correct | `V4_INFEASIBLE_WITNESS_ACCEPTED`; `V1_PARSE`; - |
| `transportation` | LP | undeclared variable (`x_s3_d1`); reversed inequality (demand <=); correct | `V1_UNDECLARED_VARIABLE`; `V4_INFEASIBLE_WITNESS_ACCEPTED`; - |
| `knapsack` | ILP | integrality dropped (continuous 0..1); correct | `V4_INFEASIBLE_WITNESS_ACCEPTED` (the witness "half of item 2"); - |
| `facility` | MILP | unbounded formulation (max, no capacity or opening constraints); **missing constraint (capacity of f1) that is not binding at the optimum and that no infeasible witness violates**; correct | `V2_UNBOUNDED`; **none: accepted (blind spot)**; never reached |
| `assignment` | ILP | renamed variables (`x12` instead of `x_w1_t2`); correct. Reports: misreported value; correct | `V1_MISSING_CARD_VARIABLE`; -. Report: `V5_REPORT_MISMATCH` |

## What the verifier checks

- **V1** the formulation parses; names are unique; every term names a declared variable; no coefficient is NaN or infinite; bounds are consistent; the card's variables are declared (`V1_PARSE`, `V1_DUPLICATE_NAME`, `V1_UNDECLARED_VARIABLE`, `V1_NONFINITE`, `V1_BOUNDS`, `V1_MISSING_CARD_VARIABLE`).
- **V2** the solver status is optimal (`V2_INFEASIBLE`, `V2_UNBOUNDED`, `V2_SOLVER_ERROR`).
- **V3** the returned solution satisfies every bound, integrality condition, and constraint of the formulation when recomputed without the solver, and its objective equals the reported one (`V3_BOUND`, `V3_INTEGRALITY`, `V3_CONSTRAINT`, `V3_OBJECTIVE`).
- **V4** witnesses: a feasible witness must be feasible for the formulation (the card's variables fixed within the formulation's own bounds, auxiliary variables solved for) and must not beat the optimum; an infeasible witness must be infeasible (`V4_FEASIBLE_WITNESS_REJECTED`, `V4_WITNESS_BEATS_OPTIMUM`, `V4_INFEASIBLE_WITNESS_ACCEPTED`).
- **V5** the report repeats the solver's objective and the values of the card's variables (`V5_REPORT_MISMATCH`).

## What the verifier does not check

- That the formulation means what the statement says beyond what the witnesses test.
- A missing constraint that is not binding at the optimum and that no infeasible witness violates. **Blind spot in this example:** the second candidate of `facility` drops the capacity constraint of depot f1; f1's capacity (100) exceeds the total demand (60), so the optimum does not change and no infeasible witness violates that constraint alone. The verifier accepts it. The accepted objective equals the reference here only because the constraint is slack for this data; with a total demand above 100 the same formulation could be wrong (when the optimum would ship more than 100 units from f1) and still pass every check that does not have a witness for it.
- An objective coefficient that no witness reveals (a test shows a changed coefficient that the witnesses do not detect).
- The correctness of the data on the card.
- The uniqueness of the optimum.
- Anything about the solver beyond its tolerances.

## Results (E4a)

The table of the run on commit `5240bef` is in [`results/e4/table.csv`](../results/e4/table.csv) and in the README. All six runs end `succeeded` with an accepted objective equal to the reference (product_mix 1800, diet 2.34545, transportation 350, knapsack 29, facility 275, assignment 9), after 2 or 3 candidates and 3 to 5 model calls. Every wrong candidate except the blind spot gets the diagnostic its script expects; the reversed inequality of `product_mix` also accepts an infeasible witness, so it gets `V4_FEASIBLE_WITNESS_REJECTED` and `V4_INFEASIBLE_WITNESS_ACCEPTED`. Both misreports get `V5_REPORT_MISMATCH`. The verifier study (E4b, Tier 2) is in [evaluation.md](evaluation.md) and [`results/tables.md`](../results/tables.md). Model costs are simulated: the scripted target is priced like the simulated `medium` model and its token counts are estimates (UTF-8 bytes / 4), so the costs are labelled estimated.

## A note on public benchmarks

The public benchmarks for natural-language optimization modelling (NL4Opt, MAMO, IndustryOR, and others) have documented annotation errors (see the audits cited in "Constructing Industrial-Scale Optimization Modeling Benchmark", arXiv 2602.10450). They are not used as ground truth here, and no error rates are quoted.
