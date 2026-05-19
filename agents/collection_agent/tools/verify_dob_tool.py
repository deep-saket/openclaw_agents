from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from src.tools.base import BaseTool

from agents.collection_agent.tools.common import utc_now
from agents.collection_agent.tools.data_store import CollectionDataStore
from agents.collection_agent.tools.schemas import VerifyDOBInput, VerifyDOBOutput


@dataclass(slots=True)
class VerifyDOBTool(BaseTool[VerifyDOBInput, VerifyDOBOutput]):
    store: CollectionDataStore
    name: str = "verify_dob"
    description: str = "Verify customer date of birth against fixture challenge data."
    input_schema = VerifyDOBInput
    output_schema = VerifyDOBOutput

    def execute(self, input: VerifyDOBInput) -> VerifyDOBOutput:
        case_row = self.store.get_case(case_id=input.case_id, customer_id=input.customer_id)
        if case_row is None:
            raise ValueError("Unknown case/customer for DOB verification.")
        customer_id = str(case_row.get("customer_id"))
        customer = self.store.get_customer(customer_id)
        if customer is None:
            raise ValueError(f"Missing fixture customer: {customer_id}")

        challenge = dict(customer.get("challenge", {}))
        expected_dob = str(challenge.get("dob", "")).strip().lower()
        provided_dob = str(input.dob).strip().lower()

        verification_rows = self.store.load_runtime("verification_attempts.json")
        failed_attempts = sum(
            1
            for row in verification_rows
            if row.get("customer_id") == customer_id
            and row.get("field") == "dob"
            and str(row.get("status")) == "failed"
        )

        if failed_attempts >= 3:
            status = "locked"
        elif provided_dob and provided_dob == expected_dob:
            status = "verified"
        else:
            status = "failed"

        verification_rows.append(
            {
                "attempt_id": f"VER-{uuid4().hex[:10].upper()}",
                "customer_id": customer_id,
                "case_id": case_row.get("case_id"),
                "field": "dob",
                "status": status,
                "created_at": utc_now().isoformat(),
            }
        )
        self.store.save_runtime("verification_attempts.json", verification_rows)

        next_failed_attempts = failed_attempts + (1 if status == "failed" else 0)
        return VerifyDOBOutput(
            customer_id=customer_id,
            status=status,
            field="dob",
            failed_attempts=next_failed_attempts,
        )

