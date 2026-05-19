from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from src.tools.base import BaseTool

from agents.collection_agent.tools.common import utc_now
from agents.collection_agent.tools.data_store import CollectionDataStore
from agents.collection_agent.tools.schemas import DispositionUpdateInput, DispositionUpdateOutput


@dataclass(slots=True)
class DispositionUpdateTool(BaseTool[DispositionUpdateInput, DispositionUpdateOutput]):
    store: CollectionDataStore
    name: str = "disposition_update"
    description: str = "Persist case disposition updates with audit ids."
    input_schema = DispositionUpdateInput
    output_schema = DispositionUpdateOutput

    def execute(self, input: DispositionUpdateInput) -> DispositionUpdateOutput:
        audit_id = f"AUD-{uuid4().hex[:10].upper()}"
        updated_at = utc_now()
        self.store.append_runtime(
            "dispositions.json",
            {
                "audit_id": audit_id,
                "case_id": input.case_id,
                "disposition_code": input.disposition_code,
                "notes": input.notes,
                "updated_at": updated_at.isoformat(),
            },
        )
        return DispositionUpdateOutput(
            case_id=input.case_id,
            disposition_code=input.disposition_code,
            audit_id=audit_id,
            updated_at=updated_at,
        )
