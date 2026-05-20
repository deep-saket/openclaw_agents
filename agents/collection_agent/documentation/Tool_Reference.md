# Tool Reference

This reference explains how each tool should be called and how output should be interpreted inside the graph.

## Tool Rules

- Inputs should match schema exactly.
- Tool failures must not silently progress customer-owned plan steps.
- Tool output is normalized into `observations` as `{tool_name, input, output}`.
- `verification_react` may only choose verification tools.
- `react` may only choose non-verification tools.

## Verification Tool Catalog

### `verify_dob`

- Purpose: verify borrower DOB against the active challenge record.
- Inputs: `case_id?`, `customer_id?`, `dob`
- Outputs: `status` (`verified|failed|locked`), `field=dob`, `failed_attempts`
- Important:
  - `verify_identity` stays active until `identity_verified=true`
  - successful output contributes to React-owned verification progression state

### `verify_mobile`

- Purpose: verify borrower registered mobile number against the active challenge record.
- Inputs: `case_id?`, `customer_id?`, `phone`
- Outputs: `status` (`verified|failed|locked`), `field=phone`, `failed_attempts`
- Important:
  - successful output contributes to React-owned verification progression state
  - `locked` is treated as incomplete for planning

## Non-Verification Tool Catalog

### `loan_policy_lookup`

- Purpose: load policy constraints for waiver, restructure, and promise workflows.
- Inputs: `case_id?`, `loan_id?`
- Outputs: policy object

### `offer_eligibility`

- Purpose: evaluate concession or arrangement eligibility.
- Inputs: `case_id`, `hardship_flag?`, `requested_waiver_pct?`
- Outputs: `allowed`, `offer_type`, `approved_waiver_pct`

### `plan_propose`

- Purpose: propose or revise repayment plan options under hardship or negotiation context.
- Inputs: `case_id`, `hardship_reason?`, `revision_index?`, `max_installment_amount?`
- Outputs: `plan_id`, `monthly_amount`, `first_due_date`, `status`

### `payment_link_create`

- Purpose: issue pay-now link.
- Inputs: `case_id`, `amount`, `channel?`
- Outputs: `payment_reference_id`, `payment_url`, `expires_at`

### `promise_capture`

- Purpose: record promise-to-pay details.
- Inputs: `case_id`, `promised_date`, `promised_amount`
- Outputs: `promise_id`, `status`

### `human_escalation`

- Purpose: escalate dispute, fraud, legal, or vulnerability requests to humans.
- Inputs: `case_id`, `reason`
- Outputs: `escalation_id`, `queue`, `priority`

## Internal Helper Tools

These are used for extraction or verification support and are not selected from the runtime React tool catalogs.

### `entity_extract`

- Purpose: extract generic entities from raw input text for session-state updates.
- Inputs: `text`
- Outputs: `entities`, `entity_keys`

### `verification_entity_extract`

- Purpose: extract verification-relevant entities from raw input.
- Inputs: `text`, `required_fields`, `include_name?`
- Outputs: `entities`, `detected_fields`, `missing_fields`

### `verification_memory_verify`

- Purpose: verify extracted verification entities against expected challenge values cached in memory.
- Inputs: `entities`, `expected_challenge`, `required_fields`
- Outputs: `status`, `matched`, `missing_fields`, `mismatched_fields`
