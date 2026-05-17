from __future__ import annotations

from dataclasses import dataclass

from src.tools.base import BaseTool

from agents.collection_agent.tools.schemas import (
    VerificationMemoryVerifyInput,
    VerificationMemoryVerifyOutput,
)


@dataclass(slots=True)
class VerificationMemoryVerifyTool(BaseTool[VerificationMemoryVerifyInput, VerificationMemoryVerifyOutput]):
    name: str = "verification_memory_verify"
    description: str = "Verify extracted verification entities against expected challenge values in memory."
    input_schema = VerificationMemoryVerifyInput
    output_schema = VerificationMemoryVerifyOutput

    def execute(self, input: VerificationMemoryVerifyInput) -> VerificationMemoryVerifyOutput:
        required_fields = [str(x).strip() for x in input.required_fields if str(x).strip()]
        expected = {
            str(key).strip(): str(value).strip().lower()
            for key, value in input.expected_challenge.items()
            if str(key).strip()
        }
        entities = {
            str(key).strip(): str(value).strip().lower()
            for key, value in input.entities.items()
            if str(key).strip()
        }

        if not required_fields:
            return VerificationMemoryVerifyOutput(
                status="insufficient",
                matched=False,
                missing_fields=[],
                mismatched_fields=[],
                required_fields=[],
                compared_fields=[],
            )

        missing_fields = [field for field in required_fields if not entities.get(field)]
        if missing_fields:
            return VerificationMemoryVerifyOutput(
                status="insufficient",
                matched=False,
                missing_fields=sorted(missing_fields),
                mismatched_fields=[],
                required_fields=required_fields,
                compared_fields=sorted([field for field in required_fields if field in entities]),
            )

        mismatched_fields = [field for field in required_fields if entities.get(field, "") != expected.get(field, "")]
        if input.require_name_match:
            expected_name = str(input.expected_name or "").strip().lower()
            provided_name = entities.get("name", "")
            if expected_name and provided_name and expected_name != provided_name:
                mismatched_fields.append("name")
            elif expected_name and not provided_name:
                missing_fields = sorted(set(missing_fields + ["name"]))
                return VerificationMemoryVerifyOutput(
                    status="insufficient",
                    matched=False,
                    missing_fields=missing_fields,
                    mismatched_fields=sorted(set(mismatched_fields)),
                    required_fields=required_fields,
                    compared_fields=sorted([field for field in required_fields if field in entities]),
                )

        matched = not mismatched_fields
        return VerificationMemoryVerifyOutput(
            status=("verified" if matched else "failed"),
            matched=matched,
            missing_fields=[],
            mismatched_fields=sorted(set(mismatched_fields)),
            required_fields=required_fields,
            compared_fields=sorted(set(required_fields + (["name"] if "name" in entities else []))),
        )

