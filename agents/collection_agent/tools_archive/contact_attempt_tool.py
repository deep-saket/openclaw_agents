from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from src.tools.base import BaseTool

from agents.collection_agent.tools.common import utc_now
from agents.collection_agent.tools.data_store import CollectionDataStore
from agents.collection_agent.tools.schemas import ContactAttemptInput, ContactAttemptOutput


@dataclass(slots=True)
class ContactAttemptTool(BaseTool[ContactAttemptInput, ContactAttemptOutput]):
    store: CollectionDataStore
    name: str = "contact_attempt"
    description: str = "Record outbound contact attempts for a case."
    input_schema = ContactAttemptInput
    output_schema = ContactAttemptOutput

    def execute(self, input: ContactAttemptInput) -> ContactAttemptOutput:
        attempt_id = f"ATT-{uuid4().hex[:10].upper()}"
        created_at = utc_now()
        status = "reached" if input.reached else "not_reached"
        self.store.append_runtime(
            "contact_attempts.json",
            {
                "attempt_id": attempt_id,
                "case_id": input.case_id,
                "channel": input.channel,
                "template_id": input.template_id,
                "status": status,
                "notes": input.notes,
                "created_at": created_at.isoformat(),
            },
        )
        return ContactAttemptOutput(
            attempt_id=attempt_id,
            case_id=input.case_id,
            channel=input.channel,
            status=status,
            created_at=created_at,
        )
