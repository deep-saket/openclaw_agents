# Collection Agent Data Dictionary

This folder contains static dummy fixtures used by `agents/collection_agent`.

All files are local demo data only. No production banking records are included.

## Files

- `cases.json`
- `customers.json`
- `policies.json`

## Entity Relationships

- `cases[].customer_id` joins with `customers[].customer_id`
- `cases[].loan_id` joins with `policies[].loan_id`

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

Interpretation:

- `offer_eligibility` and `loan_policy_lookup` rely on these constraints
- policy is authoritative for concession-related decisions

## Editing Guidance

- preserve join keys (`case_id`, `customer_id`, `loan_id`)
- keep numeric fields numeric (no quotes around amounts/DPD)
- avoid duplicate IDs
