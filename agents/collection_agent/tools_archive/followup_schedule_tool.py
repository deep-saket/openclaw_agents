from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from src.tools.base import BaseTool

from agents.collection_agent.tools.common import utc_now
from agents.collection_agent.tools.data_store import CollectionDataStore
from agents.collection_agent.tools.schemas import FollowupScheduleInput, FollowupScheduleOutput


@dataclass(slots=True)
class FollowupScheduleTool(BaseTool[FollowupScheduleInput, FollowupScheduleOutput]):
    store: CollectionDataStore
    name: str = "followup_schedule"
    description: str = "Schedule a follow-up task in local runtime storage."
    input_schema = FollowupScheduleInput
    output_schema = FollowupScheduleOutput

    def execute(self, input: FollowupScheduleInput) -> FollowupScheduleOutput:
        schedule_id = f"SCH-{uuid4().hex[:10].upper()}"
        payload = {
            "schedule_id": schedule_id,
            "case_id": input.case_id,
            "scheduled_for": input.scheduled_for,
            "preferred_channel": input.preferred_channel,
            "reason": input.reason,
            "created_at": utc_now().isoformat(),
            "status": "scheduled",
        }
        self.store.append_runtime("followups.json", payload)
        return FollowupScheduleOutput(
            schedule_id=schedule_id,
            case_id=input.case_id,
            scheduled_for=input.scheduled_for,
            preferred_channel=input.preferred_channel,
            reason=input.reason,
        )
