from __future__ import annotations

from dataclasses import dataclass

from src.tools.base import BaseTool

from agents.collection_agent.tools.data_store import CollectionDataStore
from agents.collection_agent.tools.schemas import PaymentStatusCheckInput, PaymentStatusCheckOutput


@dataclass(slots=True)
class PaymentStatusCheckTool(BaseTool[PaymentStatusCheckInput, PaymentStatusCheckOutput]):
    store: CollectionDataStore
    name: str = "payment_status_check"
    description: str = "Check or simulate payment status from local runtime records."
    input_schema = PaymentStatusCheckInput
    output_schema = PaymentStatusCheckOutput

    def execute(self, input: PaymentStatusCheckInput) -> PaymentStatusCheckOutput:
        rows = self.store.load_runtime("payment_links.json")
        selected: dict | None = None
        for row in rows:
            if row.get("payment_reference_id") == input.payment_reference_id:
                selected = row
                break
        if selected is None:
            raise ValueError("Unknown payment_reference_id.")

        if input.simulate_status is not None:
            selected["status"] = input.simulate_status
            self.store.save_runtime("payment_links.json", rows)

        status = str(selected.get("status", "pending"))
        needs_additional_action = status != "success"
        return PaymentStatusCheckOutput(
            payment_reference_id=input.payment_reference_id,
            status=status,
            amount=float(selected.get("amount", 0.0)),
            needs_additional_action=needs_additional_action,
        )
