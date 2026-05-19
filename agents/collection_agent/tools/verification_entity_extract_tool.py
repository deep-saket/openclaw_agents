from __future__ import annotations

from dataclasses import dataclass

from src.tools.base import BaseTool

from agents.collection_agent.tools.entity_extract_tool import EntityExtractTool
from agents.collection_agent.tools.schemas import (
    EntityExtractInput,
    VerificationEntityExtractInput,
    VerificationEntityExtractOutput,
)


@dataclass(slots=True)
class VerificationEntityExtractTool(BaseTool[VerificationEntityExtractInput, VerificationEntityExtractOutput]):
    name: str = "verification_entity_extract"
    description: str = "Extract only verification-relevant entities from user text."
    input_schema = VerificationEntityExtractInput
    output_schema = VerificationEntityExtractOutput

    def execute(self, input: VerificationEntityExtractInput) -> VerificationEntityExtractOutput:
        raw = EntityExtractTool().execute(input=EntityExtractInput(text=input.text))
        all_entities = dict(raw.entities)
        required = [str(x).strip() for x in input.required_fields if str(x).strip()]
        keys = set(required)
        if input.include_name:
            keys.add("name")
        filtered = {key: str(value) for key, value in all_entities.items() if key in keys and str(value).strip()}
        detected_fields = sorted([key for key in required if key in filtered])
        missing_fields = sorted([key for key in required if key not in filtered])
        return VerificationEntityExtractOutput(
            entities=filtered,
            required_fields=required,
            detected_fields=detected_fields,
            missing_fields=missing_fields,
        )
