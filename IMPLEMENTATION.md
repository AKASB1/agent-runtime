# Implementation plan

## Design rule

Keep the runtime deterministic where possible. Model generation is one step inside an explicit execution graph; tool calls and state transitions should be inspectable and replayable.

## Core interfaces

### Model adapter

Normalize messages, tool definitions, structured output, streaming, token usage, and provider errors.

### Tool

Each tool has a name, Pydantic input model, output model, timeout, and side-effect flag.

### Workflow

A workflow owns the state object and transitions between model, tool, retrieval, and terminal nodes.

### Trace

Record one trace per run and one span per model/tool/retrieval call.

## Delivery order

1. one provider adapter
2. typed tool registry
3. sequential workflow executor
4. structured output validation
5. trace store
6. retrieval adapter
7. persistent memory
8. evaluation runner
9. multi-provider routing and failure fallback

## Scaffold checkpoint

The current implementation contains protocol and in-memory primitives only. The listed provider, persistent-memory, and trace milestones remain planned.
