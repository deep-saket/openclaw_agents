# Node Reference

## Graph Order

`START -> relevance_intent -> (irrelevant_response | pre_plan_intent -> execution_path_intent/memory path -> react/tool loop -> plan_proposal -> reflect -> relevant_response) -> END`

## Node Contracts

### 1) `relevance_intent`

- Type: `CollectionIntentNode`
- Input: `user_input`, conversation context from memory
- Output keys: `relevance_intent`, `intent`, optional `response`
- Route map:
  - `relevant` -> `pre_plan_intent`
  - `irrelevant|empty` -> `irrelevant_response`
- Guardrails:
  - context-aware override for identity responses during active collections sessions

### 2) `irrelevant_response`

- Type: `CollectionResponseNode` (static path)
- Input: static `response_map` output from relevance gate
- Output: final out-of-scope `response`
- Route: terminal

### 3) `pre_plan_intent`

- Type: `CollectionIntentNode`
- Input: `user_input`
- Output: `pre_plan_intent`
- Route map:
  - `plan` -> `plan_proposal`
  - `decide` -> `execution_path_intent`

### 4) `execution_path_intent`

- Type: `CollectionIntentNode`
- Output: `execution_path_intent`
- Route map:
  - `need_memory` -> `memory_retrieve`
  - `need_tool` -> `react`

### 5) `memory_retrieve`

- Type: `MemoryRetrieveNode`
- Output: memory context/retrieval metadata
- Route: `post_memory_plan_intent`

### 6) `post_memory_plan_intent`

- Type: `CollectionIntentNode`
- Output: `post_memory_plan_intent`
- Route map:
  - `plan` -> `plan_proposal`
  - `react` -> `react`

### 7) `react`

- Type: `ReactNode`
- Output: `decision`, `steps`, potential tool call
- Route map:
  - `act` -> `tool_execution`
  - `respond|end` -> `plan_proposal`

### 8) `tool_execution`

- Type: `ToolExecutionNode`
- Output: `observation.tool_phase`
- Route: back to `react`

### 9) `plan_proposal`

- Type: `PlanProposalNode`
- Output:
  - `plan_proposal`
  - `conversation_plan`
  - `response_target`
  - `route`
- Route map:
  - `continue` -> `reflect`
- Critical logic:
  - plan tree update, marker reconciliation, node progression
  - strict verification marker gate (`verify_identity` requires `verify_dob` + `verify_mobile` success for required fields)

### 10) `reflect`

- Type: `CollectionReflectNode`
- Output: `reflection_feedback`, completion route
- Route map:
  - `retry_plan_proposal` -> `plan_proposal`
  - `complete` -> `relevant_response`

### 11) `relevant_response`

- Type: `CollectionResponseNode`
- Output: final `response`, normalized `response_target`
- Guardrails:
  - no internal reasoning/tool jargon to customer
  - verification step cannot jump to dues if identity is incomplete

## Wrapper-Added Runtime Keys

All node executions are wrapped and annotated with:

- `node_history`
- `previous_node`
- `next_node`
- `conversation_phase`

This wrapper-level instrumentation powers UI-level debug visibility.
