from __future__ import annotations

from dataclasses import dataclass

from src.tools.base import BaseTool

from agents.collection_agent.tools.common import as_case_record
from agents.collection_agent.tools.data_store import CollectionDataStore
from agents.collection_agent.tools.schemas import CaseFetchInput, CaseFetchOutput


@dataclass(slots=True)
class CaseFetchTool(BaseTool[CaseFetchInput, CaseFetchOutput]):
    store: CollectionDataStore
    name: str = "case_fetch"
    description: str = "Fetch delinquent cases from local JSON fixtures with simple filters."
    input_schema = CaseFetchInput
    output_schema = CaseFetchOutput

    def execute(self, input: CaseFetchInput) -> CaseFetchOutput:
        cases = self.store.load_cases()
        filtered: list[dict] = []
        for row in cases:
            if input.case_id and row.get("case_id") != input.case_id:
                continue
            if input.customer_id and row.get("customer_id") != input.customer_id:
                continue
            if input.portfolio_id and row.get("portfolio_id") != input.portfolio_id:
                continue
            if input.status and str(row.get("status", "")).lower() != input.status.lower():
                continue
            dpd = int(row.get("dpd", 0))
            if input.dpd_min is not None and dpd < input.dpd_min:
                continue
            if input.dpd_max is not None and dpd > input.dpd_max:
                continue
            filtered.append(row)
        filtered.sort(key=lambda row: (int(row.get("dpd", 0)), float(row.get("overdue_amount", 0.0))), reverse=True)
        records = [as_case_record(row) for row in filtered[: input.limit]]
        return CaseFetchOutput(total=len(records), cases=records)
