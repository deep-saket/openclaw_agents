from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from src.tools.base import BaseTool

from agents.collection_agent.tools.schemas import PayByPhoneCollectInput, PayByPhoneCollectOutput
from agents.collection_agent.tools.data_store import CollectionDataStore


@dataclass(slots=True)
class PayByPhoneCollectTool(BaseTool[PayByPhoneCollectInput, PayByPhoneCollectOutput]):
    store: CollectionDataStore
    name: str = "pay_by_phone_collect"
    description: str = "Mock pay-by-phone collection flow for demo (non-PCI, simulated)."
    input_schema = PayByPhoneCollectInput
    output_schema = PayByPhoneCollectOutput

    def execute(self, input: PayByPhoneCollectInput) -> PayByPhoneCollectOutput:
        if not input.consent_confirmed:
            raise ValueError("Consent must be confirmed before simulated pay-by-phone collection.")
        payment_id = f"PBPH-{uuid4().hex[:10].upper()}"
        receipt = f"RCT-{uuid4().hex[:8].upper()}"
        collected = input.amount
        if input.simulate_status == "failed":
            collected = 0.0
        elif input.simulate_status == "partial":
            collected = round(input.amount * 0.5, 2)

        self.store.append_runtime(
            "phone_payments.json",
            {
                "payment_id": payment_id,
                "case_id": input.case_id,
                "requested_amount": input.amount,
                "collected_amount": collected,
                "status": input.simulate_status,
                "receipt_reference": receipt,
            },
        )
        return PayByPhoneCollectOutput(
            payment_id=payment_id,
            case_id=input.case_id,
            collected_amount=collected,
            status=input.simulate_status,
            receipt_reference=receipt,
        )
