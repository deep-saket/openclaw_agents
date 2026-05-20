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

## Conversation D: Out-of-scope

1. Customer: "What is today’s weather?"
2. Agent: "This request is outside collections scope..."

Expected graph behavior:

- `relevance_intent` -> `irrelevant`
- `irrelevant_response`
- terminal
