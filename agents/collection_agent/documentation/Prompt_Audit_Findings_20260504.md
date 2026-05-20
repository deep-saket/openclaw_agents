# Prompt Audit Findings (2026-05-04)

Scope: Collections agent conversation robustness audit across random/off-topic/adversarial/hardship/verification flows, with architecture held constant unless critical failures require fixes.

Audit artifacts:

- `agents/collection_agent/runtime/evals/prompt_audit_20260503T193122Z.json`
- Runtime stack traces from collection UI server process

---

## Executive Summary

What worked:

- Identity progression guard works for partial verification:
  - `"My name is Aditi Sharma"` no longer advances to dues explanation.
  - Plan remained on `verify_identity` with marker `pending`.
- Response node asks missing verification fields instead of falsely confirming completion.

Critical blockers found (not prompt-only):

1. Structured-output error handling crashes intent flow.
2. Planner has no resilience when LLM decision fails.
3. These lead to HTTP 500 and stop conversation progression after a few turns.

Because of these blockers, prompt quality beyond early turns could not be fully stress-tested in live multi-scenario runs.

---

## Observed Scenario Outcomes

## 1) `happy_path_full_verification_then_pay`

- Start and first two turns worked.
- Turn 1 response: requested full verification bundle.
- Turn 2 response: requested missing fields only.
- Turn 3 onward: HTTP 500 due to runtime exception (not prompt text quality).

## 2) `off_topic_and_random`

- Could not run: start failed due same upstream runtime failure.

## 3) `prompt_injection_policy_bypass_attempt`

- Could not run: start failed due same upstream runtime failure.

## 4) `hardship_branch_negotiation`

- Could not run: start failed due same upstream runtime failure.

## 5) `trust_challenge_and_escalation_style`

- Could not run: start failed due same upstream runtime failure.

---

## Root-Cause Runtime Failures (Dire Need)

### A) Structured output exception construction is broken

Error observed:

- `StructuredOutputError.__init__() takes 1 positional argument but 2 were given`

Impact:

- Any LLM structured parse failure in intent node becomes fatal and throws 500.

Where:

- `agents/collection_agent/llm_structured.py`

### B) Planner has hard fail when LLM unavailable

Error observed:

- `CollectionPlanner LLM decision unavailable; deterministic planner fallback is disabled.`

Impact:

- React node path can terminate with 500 instead of graceful fallback response.

Where:

- `agents/collection_agent/planner.py`

These are reliability architecture issues and must be fixed before deeper prompt tuning can be trusted.

---

## Node-by-Node Risk Assessment

| Node | What can go right | What can go wrong | Prompt change needed | Architecture change needed |
| --- | --- | --- | --- | --- |
| `relevance_intent` | keeps off-topic out | LLM parse failure crashes flow | yes | yes (error resilience) |
| `pre_plan_intent` | fast route to plan vs decide | over-routes to plan for social greetings | yes | no |
| `execution_path_intent` | chooses memory/tool route | may skip needed tool when context weak | yes | no |
| `memory_retrieve` | adds continuity | stale/insufficient memory not flagged | light | optional |
| `post_memory_plan_intent` | adapts after memory | may bounce to react unnecessarily | yes | no |
| `react` | picks action/tool | hard-fails if planner LLM unavailable | no | yes |
| `tool_execution` | structured tool outputs | malformed args can fail repeated turns | small | optional |
| `plan_proposal_directive` | proposal/direction packaging | can over-eagerly request full verification repeatedly if upstream planning context is wrong | yes | no |
| `reflect` | quality gate | weak criteria can accept under-specified response | yes | no |
| `relevant_response` | user-safe packaging | tone can feel robotic / too rigid | yes | no |

---

## Prompt Changes Proposed (Primary)

These changes keep current graph architecture mostly unchanged.

### 1) `intent.relevance_system_prompt`

Add explicit rules:

- Treat borrower trust/safety queries ("is this scam", "prove this call") as in-scope.
- Treat mixed-language/Hinglish collection replies as in-scope in active session.
- Mark irrelevant only when no debt-collection relationship is present.

Proposed addition:

- `intent=relevant for trust-verification queries about the call itself (e.g., scam concerns, callback verification requests) in an active collections session.`
- `In active session, short multilingual replies that continue identity/payment discussion remain relevant.`

### 2) `intent.pre_plan_system_prompt`

Current binary (`plan|decide`) is too coarse for courtesy turns.

Prompt adjustment:

- Keep labels unchanged, but define:
  - `plan` for direct customer-facing response without tool need.
  - `decide` only when a tool/memory/planner branch is genuinely needed.

Expected effect:

- Reduces premature deep routing for simple conversational acknowledgements.

### 3) `intent.execution_path_system_prompt`

Add stronger criteria:

- `need_tool` when verification completion or payment action can be executed now.
- `need_memory` when prior commitments/follow-up dates are referenced.
- avoid `need_tool` when evidence is partial and only clarifying question is required.

### 4) `plan_proposal_directive` prompt (current plan proposal render path)

Add strict planning instructions:

- If current plan node is `verify_identity` and `identity_verified=false`, do not advance to `explain_dues`.
- Ask only missing fields based on `verification_collected` and `required_fields`.
- Do not ask for already-provided fields repeatedly.
- For scam/trust concerns, provide verification-safe script before requesting sensitive info.

### 5) `response.system_prompt`

Add tone + compliance constraints:

- Never claim identity is confirmed unless `identity_verified=true`.
- For verification in progress, ask for missing fields only.
- If customer challenges trust, provide safe verification path (official callback/channel) before proceeding.
- Keep questions short and one actionable next step.

### 6) `reflect.system_prompt`

Strengthen completeness criteria:

- In verification stage, response is incomplete if it does not request missing required verification data.
- In payment stage, response is incomplete if no explicit next action is offered.

---

## Dire-Need Architecture Changes (Minimal)

These are reliability fixes, not topology redesign.

1. Fix `StructuredOutputError` class initialization so exceptions carry message safely.
2. Add graceful fallback behavior when structured LLM output fails in intent nodes:
   - either deterministic fallback by toggle, or soft-fail response path (no 500).
3. Add graceful fallback in `CollectionPlanner` when LLM unavailable:
   - return conservative `respond_directly` message instead of raising fatal exception.
4. Add node-level error shield in UI runtime:
   - convert internal exceptions into structured error response to keep demo session alive.

Without these, prompt tuning impact cannot be reliably validated end-to-end.

---

## Suggested Prompt-Only Rollout Order

1. `response.system_prompt` and plan proposal instruction hardening.
2. intent router prompt refinements (`pre_plan`, `execution_path`).
3. relevance refinements for trust/scam and mixed-language replies.
4. reflect completeness criteria upgrade.

---

## Suggested Validation Set (after reliability fixes)

- Happy path with full verification -> dues -> pay link.
- Partial verification with mixed order fields.
- Wrong verification repeated until lock.
- Off-topic random queries mid-session.
- Prompt injection attempts for policy bypass.
- Hardship request with revised EMI branch.
- Scam/trust concern escalation.
- Human escalation request.
