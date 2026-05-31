# Node Reference

## Graph Order

`START -> relevance_intent -> (irrelevant_response | entity_extract -> negotiation_classification -> pre_plan_intent -> execution_path_intent/memory path -> verification_react or react -> tool_execution loop -> post_verification_intent/plan_proposal_state -> plan_proposal_graph -> plan_proposal_directive -> reflect -> relevant_response) -> END`

## Node Contracts

### 1) `relevance_intent`

- Type: `CollectionIntentNode`
- Input: `user_input`, conversation context from memory
- Output keys: `relevance_intent`, `intent`, optional `response`
- Route map:
  - `relevant` -> `entity_extract`
  - `irrelevant|empty` -> `irrelevant_response`
- Guardrails:
  - context-aware override for identity responses during active collections sessions

### 2) `irrelevant_response`

- Type: `CollectionResponseNode` (static path)
- Input: static `response_map` output from relevance gate
- Output: final out-of-scope `response`
- Route: terminal

### 3) `entity_extract`

- Type: `CollectionEntityExtractNode`
- Input: `user_input`, recent conversation, verification requirements
- Output keys:
  - `extracted_entities`
  - `extracted_entities_turn`
  - `extracted_entity_descriptions`
  - `verification_entities`
  - `customer_payment_capacity`
  - `customer_payment_capacity_pct`
- Route: `negotiation_classification`

### 4) `negotiation_classification`

- Type: `NegotiationClassificationNode`
- Input:
  - `user_input`
  - `recent_conversation`
  - `memory.state`
  - `extracted_entities`
  - `customer_profile_summary`
  - `payment_history_summary`
  - `offer_history_summary`
  - verification state
- Output keys:
  - `negotiation_classification`
  - `conversation_mode`
  - `negotiation_stage`
  - `customer_payment_posture`
  - `discount_stage`
  - `customer_payment_willingness`
  - `hardship_context`
  - `discount_requested`
  - `discount_offered`
  - `discount_accepted`
  - `discount_rejected`
  - `counter_offer_present`
  - `response_mode`
  - `active_dialogue_owner`
- Route: `pre_plan_intent`
- Ownership:
  - owns persistent negotiation cognition state
  - owns posture, discount stage, and customer willingness scoring
  - preserves discount-request / counter-offer / accept / reject lifecycle state across turns
  - does not choose tools or final user-facing responses

### 5) `pre_plan_intent`

- Type: `CollectionIntentNode`
- Input: `user_input` plus negotiation/verification/planning context
- Output: `pre_plan_intent`
- Route map:
  - `plan` -> `plan_proposal_state`
  - `decide` -> `execution_path_intent`

### 6) `execution_path_intent`

- Type: `CollectionIntentNode`
- Output: `execution_path_intent`
- Route map:
  - `need_memory` -> `memory_retrieve`
  - `verification_react` -> `verification_react`
  - `need_tool` / `react` -> `react`
- Guardrails:
  - if verification is incomplete and current turn supplies missing verification evidence, route to `verification_react`

### 7) `memory_retrieve`

- Type: `MemoryRetrieveNode`
- Output: memory context/retrieval metadata
- Route: `post_memory_plan_intent`

### 8) `post_memory_plan_intent`

- Type: `CollectionIntentNode`
- Output: `post_memory_plan_intent`
- Route map:
  - `plan` -> `plan_proposal_state`
  - `verification_react` -> `verification_react`
  - `react` -> `react`

### 9) `verification_react`

- Type: `VerificationReactNode`
- Output:
  - `decision`
  - `steps`
  - verification progression keys:
    - `verified_dob`
    - `verified_mobile`
    - `verification_verified_fields`
    - `verification_missing_fields`
    - `identity_verified`
- Route map:
  - `act` -> `tool_execution`
  - `respond|end` -> `post_verification_intent`
- Ownership:
  - this node is the owner of verification progression state
  - verification state is recomputed from `observations`
  - only verification tools are available here

### 10) `post_verification_intent`

- Type: `CollectionIntentNode`
- Output: `post_verification_intent`
- Route map:
  - `plan` -> `plan_proposal_state`
  - `react` -> `react`

### 11) `react`

- Type: `CollectionReactNode`
- Output: `decision`, `steps`, potential non-verification tool call
- Route map:
  - `act` -> `tool_execution`
  - `respond|end` -> `plan_proposal_state`
- Constraints:
  - non-verification tool catalog only
  - consumes negotiation continuity context from graph state

### 12) `tool_execution`

- Type: `ToolExecutionNode`
- Output:
  - `observations`
  - `observation`
- Route map:
  - verification-tool results -> `verification_react`
  - other tool results -> `react`
- Contract:
  - appends normalized observation entries with `tool_name`, `input`, and `output`

### 13) `plan_proposal_state`

- Type: `PlanProposalStateNode`
- Output:
  - `plan_prepared_memory_state`
  - `plan_signals`
  - `plan_mode`
  - `plan_origin`
  - `effective_identity_verified`
  - `latest_observation`
  - `observed_tool`
  - `observed_tool_output`
  - `existing_conversation_plan`
  - `plan_state_prompt`
  - `plan_state_system_prompt`
  - `plan_state_llm_response`
  - `plan_state_llm_error`
- Route map:
  - `continue` -> `plan_proposal_graph`
- Ownership:
  - overlays verification and negotiation state into a prepared planning snapshot
  - classifies plan signals and effective planning mode
  - does not mutate plan tree or build final response directives

### 14) `plan_proposal_graph`

- Type: `PlanProposalGraphNode`
- Output:
  - `conversation_plan`
  - `plan_tree_context`
  - `plan_graph_debug`
  - `response_target`
  - `route`
- Route map:
  - `continue` -> `plan_proposal_directive`
- Critical logic:
  - plan tree update, marker reconciliation, node progression
  - `identity_verified` is the authoritative verification gate for leaving `verify_identity`
  - preserves historical completed/skipped/blocked nodes while revising only current and future path

### 15) `plan_proposal_directive`

- Type: `PlanProposalDirectiveNode`
- Output:
  - `plan_proposal`
  - `response_target`
  - `route`
  - optional `decision`
  - optional `handoff_payload`
  - optional `additional_targets`
  - optional `memory_helper_trigger`
  - `prompt`
  - `system_prompt`
  - `llm_response`
  - `llm_error`
- Route map:
  - `continue` -> `reflect`
- Critical logic:
  - builds final proposal and `response_directive`
  - preserves hardship negotiation continuity in the proposal contract
  - may route to `discount_planning_agent` or emit loop/termination payloads
  - routes to `discount_planning_agent` for verified hardship/cannot-pay, discount or settlement requests, partial-payment proposals, counter-offers, and active `discount_stage` values `requested` / `counter_offer`
  - specialist handoff payload includes `customer_payment_capacity`, `customer_payment_capacity_pct`, `customer_payment_posture`, `discount_stage`, and hardship reason when available
  - consumes but does not mutate negotiation-owned state such as `discount_stage`

### 16) `reflect`

- Type: `CollectionReflectNode`
- Output: `reflection_feedback`, completion route
- Route map:
  - `retry_plan_proposal` -> `plan_proposal_state`
  - `complete` -> `relevant_response`

### 17) `relevant_response`

- Type: `CollectionResponseNode`
- Output: final `response`, normalized `response_target`
- Guardrails:
  - no internal reasoning/tool jargon to customer
  - verification step cannot jump to dues if identity is incomplete
  - response tone should respect `response_mode`
  - active hardship negotiation should respect `active_dialogue_owner`

## Wrapper-Added Runtime Keys

All node executions are wrapped and annotated with:

- `node_history`
- `previous_node`
- `next_node`
- `conversation_phase`

This wrapper-level instrumentation powers UI-level debug visibility.
