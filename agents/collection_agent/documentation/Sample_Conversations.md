# Sample Conversations

## Conversation A: Verification incomplete, dues disclosure blocked

1. Agent: "Hello Aditi, this is Alex from the bank's collections team. I am calling regarding your loan account dues. Before I share details, please confirm your date of birth and registered phone number."
2. Customer: "Sorry for the delay, how can you help me?"
3. Agent: "Before I share details, please confirm your date of birth (YYYY-MM-DD) and your registered phone number."
4. Customer: "My name is Aditi Sharma"
5. Agent: "Thank you for confirming your name. Please also share your date of birth (YYYY-MM-DD) and your registered phone number."

Expected graph behavior:

- current node remains `verify_identity`
- marker for `verify_identity` remains `pending`
- no dues explanation until `verify_dob` / `verify_mobile` complete verification

## Conversation B: Verification complete and payment intent

1. Customer: "DOB 1991-08-19 and my phone is 919900001001"
2. Agent: "Thank you, verification is complete. Your total due is INR 13350. Would you like to pay now?"
3. Customer: "Yes, send payment link on SMS"
4. Agent: "Done. I can generate the link now."

Expected tool path:

- `verify_dob` -> `verified`
- `verify_mobile` -> `verified`
- `payment_link_create`

## Conversation C: Hardship branch

1. Customer: "I cannot pay full amount this week, can you reduce EMI?"
2. Agent: routes into hardship negotiation mode
3. Agent: evaluates eligibility and proposes plan
4. Customer: accepts plan and asks for follow-up date
5. Agent: captures promise

Expected tool path:

- `offer_eligibility`
- `plan_propose`
- `promise_capture`

## Conversation D: Discount planning flow

1. Customer: "I lost my job and cannot pay the full EMI. Can you offer any settlement?"
2. Agent: routes into hardship negotiation mode and specialist discount planning
3. Specialist handoff payload includes hardship reason, posture, and payment capacity context
4. Specialist recommendation returns to collection agent
5. Agent presents the returned option to the customer

Expected state evolution:

- `customer_payment_posture=cannot_pay`
- `discount_stage=requested -> planning -> offered`
- `discount_requested=true`
- `discount_offered=true`
- `response_target=discount_planning_agent` on the specialist hop

## Conversation E: Partial payment flow

1. Customer: "I can pay 2000 today if you can settle the rest."
2. Entity extraction captures `customer_payment_capacity=2000`
3. Negotiation classification sets `customer_payment_posture=partial_now`
4. Planner routes to `discount_planning_agent`

Expected handoff payload:

- `customer_payment_capacity=2000`
- `customer_payment_capacity_pct=null`
- `customer_payment_posture=partial_now`
- `discount_stage=requested`

## Conversation F: Posture transition flow

1. Turn 1 customer: "I lost my job." -> `customer_payment_posture=cannot_pay`
2. Turn 5 customer: "I can pay 3000 today." -> `customer_payment_posture=partial_now`
3. Turn 9 customer: "Send me the payment link." -> `customer_payment_posture=pay_now`

Expected persistence:

- `customer_payment_posture_history` records each transition in order
- `customer_payment_willingness` rises across the turns
- final conversation can move into promise capture / payment link flow without losing historical hardship evidence

## Conversation G: Out-of-scope

1. Customer: "What is today’s weather?"
2. Agent: "This request is outside collections scope..."

Expected graph behavior:

- `relevance_intent` -> `irrelevant`
- `irrelevant_response`
- terminal
