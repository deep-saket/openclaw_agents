from __future__ import annotations

from dataclasses import dataclass

from src.tools.base import BaseTool

from agents.collection_agent.tools.data_store import CollectionDataStore
from agents.collection_agent.tools.schemas import CasePrioritizeInput, CasePrioritizeOutput, PrioritizedCase


@dataclass(slots=True)
class CasePrioritizeTool(BaseTool[CasePrioritizeInput, CasePrioritizeOutput]):
    store: CollectionDataStore
    name: str = "case_prioritize"
    description: str = "Rank cases by delinquency and amount-at-risk."
    input_schema = CasePrioritizeInput
    output_schema = CasePrioritizeOutput

    def execute(self, input: CasePrioritizeInput) -> CasePrioritizeOutput:
        cases = self.store.load_cases()
        scored: list[PrioritizedCase] = []
        for row in cases:
            case_id = str(row.get("case_id"))
            if input.case_ids and case_id not in input.case_ids:
                continue
            if input.portfolio_id and row.get("portfolio_id") != input.portfolio_id:
                continue
            dpd = float(row.get("dpd", 0))
            overdue = float(row.get("overdue_amount", 0.0))
            risk_band = str(row.get("risk_band", "medium")).lower()
            risk_multiplier = {"low": 1.0, "medium": 1.2, "high": 1.5}.get(risk_band, 1.1)
            score = round((dpd * 1.8 + overdue / 1000.0) * risk_multiplier, 2)
            reason = f"DPD={int(dpd)}, overdue={overdue:.2f}, risk_band={risk_band}"
            scored.append(PrioritizedCase(case_id=case_id, priority_score=score, reason=reason))
        scored.sort(key=lambda row: row.priority_score, reverse=True)
        queue = scored[: input.top_k]
        return CasePrioritizeOutput(total=len(queue), queue=queue)
