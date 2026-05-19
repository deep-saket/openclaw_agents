from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from src.tools.base import BaseTool

from agents.collection_agent.tools.schemas import ChannelSwitchInput, ChannelSwitchOutput
from agents.collection_agent.tools.data_store import CollectionDataStore


@dataclass(slots=True)
class ChannelSwitchTool(BaseTool[ChannelSwitchInput, ChannelSwitchOutput]):
    store: CollectionDataStore
    name: str = "channel_switch"
    description: str = "Simulate a channel switch while carrying prior conversation context."
    input_schema = ChannelSwitchInput
    output_schema = ChannelSwitchOutput

    def execute(self, input: ChannelSwitchInput) -> ChannelSwitchOutput:
        switch_id = f"CHSW-{uuid4().hex[:10].upper()}"
        attempts = self.store.load_runtime("contact_attempts.json")
        recent = [a for a in attempts if a.get("case_id") == input.case_id][-3:]
        summary = "No prior outreach events." if not recent else "; ".join(
            f"{row.get('channel')}:{row.get('status')}" for row in recent
        )
        self.store.append_runtime(
            "channel_switches.json",
            {
                "switch_id": switch_id,
                "case_id": input.case_id,
                "from_channel": input.from_channel,
                "to_channel": input.to_channel,
                "reason": input.reason,
                "carried_context_summary": summary,
            },
        )
        return ChannelSwitchOutput(
            switch_id=switch_id,
            case_id=input.case_id,
            from_channel=input.from_channel,
            to_channel=input.to_channel,
            reason=input.reason,
            carried_context_summary=summary,
        )
