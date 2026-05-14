from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from src.tools.base import BaseTool

from agents.collection_agent.tools.common import utc_now
from agents.collection_agent.tools.data_store import CollectionDataStore
from agents.collection_agent.tools.schemas import CustomerVerifyInput, CustomerVerifyOutput


@dataclass(slots=True)
class CustomerVerifyTool(BaseTool[CustomerVerifyInput, CustomerVerifyOutput]):
    store: CollectionDataStore
    name: str = "customer_verify"
    description: str = "Verify customer challenge answers using local fixture data."
    input_schema = CustomerVerifyInput
    output_schema = CustomerVerifyOutput

    def execute(self, input: CustomerVerifyInput) -> CustomerVerifyOutput:
        case_row = self.store.get_case(case_id=input.case_id, customer_id=input.customer_id)
        if case_row is None:
            raise ValueError("Unknown case/customer for verification.")
        customer_id = str(case_row.get("customer_id"))
        customer = self.store.get_customer(customer_id)
        if customer is None:
            raise ValueError(f"Missing fixture customer: {customer_id}")

        challenge = dict(customer.get("challenge", {}))
        required_fields = ["dob", "phone"]
        verification_rows = self.store.load_runtime("verification_attempts.json")
        failed_attempts = sum(
            1
            for row in verification_rows
            if row.get("customer_id") == customer_id and str(row.get("status")) == "failed"
        )

        expected_dob = str(challenge.get("dob", "")).strip().lower()
        expected_phone = str(challenge.get("phone", customer.get("phone", ""))).strip().lower()
        provided = {key: str(value).strip().lower() for key, value in input.challenge_answers.items()}
        matched = (
            provided.get("dob", "") == expected_dob
            and provided.get("phone", "") == expected_phone
        )
        if failed_attempts >= 3:
            status = "locked"
        elif matched:
            status = "verified"
        else:
            status = "failed"

        verification_rows.append(
            {
                "attempt_id": f"VER-{uuid4().hex[:10].upper()}",
                "customer_id": customer_id,
                "case_id": case_row.get("case_id"),
                "status": status,
                "created_at": utc_now().isoformat(),
            }
        )
        self.store.save_runtime("verification_attempts.json", verification_rows)

        next_failed_attempts = failed_attempts + (1 if status == "failed" else 0)
        return CustomerVerifyOutput(
            customer_id=customer_id,
            status=status,
            failed_attempts=next_failed_attempts,
            required_fields=required_fields,
        )
