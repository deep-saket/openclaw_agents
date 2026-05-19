from __future__ import annotations

from dataclasses import dataclass

from src.tools.base import BaseTool

from agents.collection_agent.tools.data_store import CollectionDataStore
from agents.collection_agent.tools.schemas import DuesExplainBuildInput, DuesExplainBuildOutput


@dataclass(slots=True)
class DuesExplainBuildTool(BaseTool[DuesExplainBuildInput, DuesExplainBuildOutput]):
    store: CollectionDataStore
    name: str = "dues_explain_build"
    description: str = "Build a customer-safe dues explanation from fixture case data."
    input_schema = DuesExplainBuildInput
    output_schema = DuesExplainBuildOutput

    def execute(self, input: DuesExplainBuildInput) -> DuesExplainBuildOutput:
        case_row = self.store.get_case(case_id=input.case_id)
        if case_row is None:
            raise ValueError("Unknown case for dues explanation.")
        overdue = float(case_row.get("overdue_amount", 0.0))
        late_fee = float(case_row.get("late_fee", 0.0))
        total_due = round(overdue + late_fee, 2)
        explanation = (
            f"Your loan account has {int(case_row.get('dpd', 0))} days past due. "
            f"Overdue EMI amount is INR {overdue:.2f}, late fee is INR {late_fee:.2f}. "
            f"Total amount due now is INR {total_due:.2f}."
        )
        return DuesExplainBuildOutput(
            case_id=input.case_id,
            customer_id=str(case_row.get("customer_id")),
            total_due=total_due,
            explanation=explanation,
        )
