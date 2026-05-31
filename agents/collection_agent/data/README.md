# Collection Agent Data Dictionary

This folder contains static dummy fixtures used by `agents/collection_agent`.

All files are local demo data only. No production banking records are included.

## Files

- `cases.json`
- `customers.json`
- `policies.json`
- `customer_profile.json`
- `payment_history.json`
- `offer_history.json`
- `assistance_programs.json`

## Entity Relationships

- `cases[].customer_id` joins with `customers[].customer_id`
- `cases[].loan_id` joins with `policies[].loan_id`
- `customer_profile[].customer_id` joins with `customers[].customer_id`
- `payment_history[].customer_id` joins with `customers[].customer_id`
- `offer_history[].case_id` joins with `cases[].case_id`
- `assistance_programs[].eligible_products[*]` joins with `cases[].product`

## `cases.json`

One row per delinquency case.

Keys:

- `case_id`: unique case identifier (`COLL-xxxx`)
- `customer_id`: foreign key to customer profile
- `loan_id`: foreign key to policy row
- `portfolio_id`: business portfolio bucket
- `product`: product type (`personal_loan`, `business_loan`, `home_loan`)
- `dpd`: days past due (integer)
- `emi_amount`: monthly EMI amount
- `overdue_amount`: overdue principal/EMI amount currently due
- `late_fee`: accumulated late fee
- `status`: case status (`open` expected in demo)
- `risk_band`: simple risk segment (`low`, `medium`, `high`)
- `assigned_agent`: current owner / dialer assignment
- `last_contact_date`: most recent contact date
- `contact_attempts`: count of prior attempts
- `promise_to_pay_date`: latest promised repayment date if present
- `promise_to_pay_amount`: latest promised repayment amount if present

Interpretation:

- `total_due` is usually treated as `overdue_amount + late_fee`
- higher `dpd` and `risk_band=high` should typically be prioritized first

## `customers.json`

One row per customer contact and verification profile.

Keys:

- `customer_id`: unique customer identifier
- `name`: display name
- `phone`: phone contact
- `email`: email contact
- `preferred_language`: preferred spoken language
- `preferred_channel`: preferred contact channel
- `vulnerable_customer`: coarse vulnerability flag for guardrails
- `challenge`: object containing verification challenge values
- `challenge.dob`: date-of-birth string
- `challenge.last4_pan`: masked PAN suffix
- `challenge.zip`: postal code

Interpretation:

- `challenge` values are the source-of-truth for iterative verification checks
- `verify_dob` compares provided DOB against `challenge.dob`
- `verify_mobile` compares provided phone against `challenge.phone`

## `policies.json`

One row per loan policy profile.

Keys:

- `loan_id`: unique loan identifier
- `product`: product type
- `max_promise_days`: maximum policy window for promise-to-pay
- `waiver_allowed`: whether waiver is allowed
- `max_waiver_pct`: upper waiver percent limit
- `restructure_allowed`: whether restructure flow is allowed
- `notes`: free text policy guidance used in explanations
- `allow_partial_payment`: whether partial payment can be discussed
- `min_partial_payment_pct`: minimum partial payment threshold
- `allow_counter_offer`: whether customer counter-offers are permitted
- `max_counter_offer_rounds`: maximum counter-offer rounds allowed
- `requires_hardship_for_discount`: whether concessions require hardship evidence

Interpretation:

- `offer_eligibility` and `loan_policy_lookup` rely on these constraints
- policy is authoritative for concession-related decisions

## `customer_profile.json`

One row per customer behavioural profile.

Keys:

- `customer_id`
- `employment_type`
- `customer_segment`
- `risk_score`
- `tenure_years`
- `previous_hardship_count`
- `previous_settlement_count`
- `previous_broken_promises`
- `preferred_language`
- `preferred_channel`
- `vulnerable_customer`

## `payment_history.json`

One row per customer payment history summary.

Keys:

- `customer_id`
- `payments[]`
- `payments[].date`
- `payments[].amount`

## `offer_history.json`

One row per case-level concession / settlement history.

Keys:

- `case_id`
- `offers[]`
- `offers[].offer_pct`
- `offers[].status`

## `assistance_programs.json`

One row per hardship or concession program.

Keys:

- `program_id`
- `eligible_products[]`
- `hardship_reasons[]`
- `max_discount_pct`
- `max_installments`
- `requires_manager_approval`

## Active collection context

`CollectionContextBuilder` loads these datasets before the graph starts and builds:

```json
{
  "customer": {},
  "case": {},
  "policy": {},
  "customer_profile": {},
  "payment_history": {},
  "offer_history": {},
  "assistance_programs": []
}
```

That aggregate is injected into graph state as `active_collection_context` and summarized into memory-friendly prompt context fields.

## Editing Guidance

- preserve join keys (`case_id`, `customer_id`, `loan_id`)
- keep numeric fields numeric (no quotes around amounts/DPD)
- avoid duplicate IDs
