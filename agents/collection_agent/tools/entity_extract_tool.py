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

        # Phone extraction: avoid combining unrelated numbers (e.g., DOB + phone).
        phone_value = ""

        phone_labeled = re.search(
            r"\b(?:phone|mobile|contact)(?:\s+number)?\b[\s:,\-]*(?:is\s+)?((?:\+?91[\s\-]*)?[6-9][\d\s\-]{8,14})",
            text,
            re.IGNORECASE,
        )
        if phone_labeled:
            normalized = re.sub(r"\D", "", phone_labeled.group(1))
            if len(normalized) == 12 and normalized.startswith("91"):
                normalized = normalized[-10:]
            if len(normalized) == 10 and normalized[0] in {"6", "7", "8", "9"}:
                phone_value = normalized

        if not phone_value:
            candidates = re.findall(r"(?:\+?91[\s\-]*)?[6-9][\d\s\-]{8,14}", text)
            for candidate in candidates:
                normalized = re.sub(r"\D", "", candidate)
                if len(normalized) == 12 and normalized.startswith("91"):
                    normalized = normalized[-10:]
                if len(normalized) == 10 and normalized[0] in {"6", "7", "8", "9"}:
                    phone_value = normalized
                    break

        if phone_value:
            entities["phone"] = phone_value

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

        name_match = re.search(
            r"\b(?:my name is|i am|this is)\s+([a-zA-Z][a-zA-Z\s'.-]{1,80}?)(?=\s*(?:[,.!?]|$|\bmy\s+dob\b|\bdob\b|\bphone\b|\bnumber\b))",
            text,
            re.IGNORECASE,
        )
        if name_match:
            entities["name"] = re.sub(r"\s+", " ", name_match.group(1)).strip()

        capacity_amount_match = re.search(
            r"\b(?:can pay|pay)\b[^\d]{0,12}(?:inr|rs\.?|rupees)?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\b(?:\s*(?:today|now|right now|this week|this month))?",
            text,
            re.IGNORECASE,
        )
        if capacity_amount_match:
            entities["customer_payment_capacity"] = capacity_amount_match.group(1).replace(",", "")

        if re.search(r"\bhalf\b", text, re.IGNORECASE):
            entities["customer_payment_capacity_pct"] = "50"
        else:
            capacity_pct_match = re.search(
                r"\b(\d+(?:\.\d+)?)\s*(?:%|percent)\b",
                text,
                re.IGNORECASE,
            )
            if capacity_pct_match:
                entities["customer_payment_capacity_pct"] = capacity_pct_match.group(1)

        return EntityExtractOutput(entities=entities, entity_keys=sorted(entities.keys()))
