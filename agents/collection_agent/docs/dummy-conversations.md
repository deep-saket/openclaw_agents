# Dummy Conversations (Showcase Script)

Use these conversations for demo day.

## Conversation A: Strict Collections -> Payment Link

1. Customer (SMS): `Hello, this is about COLL-1001`
2. Agent: Opens collections flow and requests verification details before sensitive dues disclosure.
3. Customer: `I am willing to make a payment now for COLL-1001 amount=6000`
4. Agent: Runs `payment_link_create` in strict mode once verification is complete.
5. Agent response: payment link shared.

What this demonstrates:
- script-first collections behavior
- verification before dues/payment details
- deterministic payment-link path

## Conversation B: Hardship + Negotiation Continuity

1. Customer (SMS): `I need assistance. I lost my job and cannot pay this month. case_id=COLL-1002`
2. Agent: switches to hardship negotiation mode and runs `offer_eligibility`.
3. Plan node: injects `plan_propose` and returns initial plan.
4. Customer: `This does not work. Can you keep it under 1200?`
5. Plan node loop: revises and runs `plan_propose` again.
6. Customer: `Yes that works for me`
7. Agent: captures promise (`promise_capture`).

What this demonstrates:
- multi-step hardship negotiation
- persistent hardship context across turns
- no reset into generic pay-now-or-arrangement menu

## Conversation C: Off-topic Guardrail

1. Customer: `Who won the super bowl last year?`
2. Agent: rejects and redirects to debt/payment domain.

What this demonstrates:
- compliance-safe scope boundaries

## Notes for Presenter

- Use same `--session-id` for all turns in one conversation so memory carries over.
- Show runtime files under `runtime/` after each flow to prove persistence.
