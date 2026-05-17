from __future__ import annotations

from dataclasses import dataclass
import re

from src.tools.base import BaseTool

from agents.collection_agent.tools.schemas import EntityExtractInput, EntityExtractOutput


@dataclass(slots=True)
class EntityExtractTool(BaseTool[EntityExtractInput, EntityExtractOutput]):
    name: str = "entity_extract"
    description: str = "Extract generic entities from free-form collections text."
    input_schema = EntityExtractInput
    output_schema = EntityExtractOutput

    def execute(self, input: EntityExtractInput) -> EntityExtractOutput:
        text = str(input.text or "").strip()
        entities: dict[str, str] = {}
        if not text:
            return EntityExtractOutput(entities=entities, entity_keys=[])

        case_match = re.search(r"(COLL-\d+)", text, re.IGNORECASE)
        if case_match:
            entities["case_id"] = case_match.group(1).upper()

        customer_match = re.search(r"(CUST-\d+)", text, re.IGNORECASE)
        if customer_match:
            entities["customer_id"] = customer_match.group(1).upper()

        loan_match = re.search(r"(LOAN-\d+)", text, re.IGNORECASE)
        if loan_match:
            entities["loan_id"] = loan_match.group(1).upper()

        dob_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        if dob_match:
            entities["dob"] = dob_match.group(1)

        digits_only = re.sub(r"\D", "", text)
        phone_match = re.search(r"(?:\+?91)?([6-9]\d{9})", digits_only)
        if phone_match:
            entities["phone"] = phone_match.group(1)

        zip_match = re.search(
            r"\b(?:zip(?:\s*code)?|pincode|pin(?:\s*code)?)\b[\s:,\-]*(?:is\s+)?(\d{6})\b",
            text,
            re.IGNORECASE,
        )
        if zip_match:
            entities["zip"] = zip_match.group(1)

        pan_last4_match = re.search(
            r"\b(?:last\s*4|last\s*four|ending)\b.{0,30}?\b([a-zA-Z0-9]{4})\b",
            text,
            re.IGNORECASE,
        )
        if pan_last4_match:
            entities["last4_pan"] = pan_last4_match.group(1).upper()

        name_match = re.search(r"\b(?:my name is|i am|this is)\s+([a-zA-Z][a-zA-Z\s'.-]{1,80})", text, re.IGNORECASE)
        if name_match:
            entities["name"] = re.sub(r"\s+", " ", name_match.group(1)).strip()

        return EntityExtractOutput(entities=entities, entity_keys=sorted(entities.keys()))

