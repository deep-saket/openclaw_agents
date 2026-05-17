"""Entity extraction node for collection graph."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, Field

from agents.collection_agent.llm_structured import StructuredOutputRunner
from src.nodes.base import BaseGraphNode
from src.nodes.types import AgentState, NodeUpdate


class _EntityPayload(BaseModel):
    entities: dict[str, str] = Field(default_factory=dict)
    entity_descriptions: dict[str, str] = Field(default_factory=dict)


@dataclass(slots=True)
class CollectionEntityExtractNode(BaseGraphNode):
    """Extracts entities into memory/state immediately after relevance gating."""

    llm: Any | None = None
    extract_callback: Callable[[Any, str], None] | None = None
    reconcile_callback: Callable[[Any], None] | None = None
    allow_callback_fallback: bool = False
    system_prompt: str = ""
    user_prompt: str = ""

    def execute(self, state: AgentState) -> NodeUpdate:
        memory = state.get("memory")
        user_input = str(state.get("user_input", ""))
        memory_state = dict(getattr(memory, "state", {})) if memory is not None else {}
        existing_entities = (
            dict(memory_state.get("extracted_entities", {}))
            if isinstance(memory_state.get("extracted_entities"), dict)
            else {}
        )
        existing_descriptions = (
            dict(memory_state.get("extracted_entity_descriptions", {}))
            if isinstance(memory_state.get("extracted_entity_descriptions"), dict)
            else {}
        )
        required_fields = (
            [str(x).strip() for x in memory_state.get("active_verification_required_fields", []) if str(x).strip()]
            if isinstance(memory_state.get("active_verification_required_fields"), list)
            else []
        )

        used_callback_fallback = False
        llm_entities: dict[str, str] = {}
        llm_descriptions: dict[str, str] = {}
        if self.llm is not None and user_input.strip():
            extracted = self._extract_with_llm(
                user_input=user_input,
                required_fields=required_fields,
                existing_entities=existing_entities,
                active_customer_name=str(memory_state.get("active_customer_name", "")).strip(),
            )
            if extracted is not None:
                llm_entities = extracted.entities
                llm_descriptions = extracted.entity_descriptions

        if not llm_entities and callable(self.extract_callback) and self.allow_callback_fallback and memory is not None:
            used_callback_fallback = True
            self.extract_callback(memory=memory, user_input=user_input)
            memory_state = dict(getattr(memory, "state", {}))
            existing_entities = (
                dict(memory_state.get("extracted_entities", {}))
                if isinstance(memory_state.get("extracted_entities"), dict)
                else {}
            )
            existing_descriptions = (
                dict(memory_state.get("extracted_entity_descriptions", {}))
                if isinstance(memory_state.get("extracted_entity_descriptions"), dict)
                else {}
            )

        merged_entities = dict(existing_entities)
        merged_descriptions = dict(existing_descriptions)
        for key, value in llm_entities.items():
            key_norm = str(key).strip()
            val_norm = str(value).strip()
            if key_norm and val_norm:
                merged_entities[key_norm] = val_norm
        for key, value in llm_descriptions.items():
            key_norm = str(key).strip()
            val_norm = str(value).strip()
            if key_norm and val_norm:
                merged_descriptions[key_norm] = val_norm

        verification_entities = (
            dict(memory_state.get("verification_entities", {}))
            if isinstance(memory_state.get("verification_entities"), dict)
            else {}
        )
        for field in required_fields:
            val = str(merged_entities.get(field, "")).strip()
            if val:
                verification_entities[field] = val
        if str(merged_entities.get("name", "")).strip():
            verification_entities["name"] = str(merged_entities["name"]).strip()
            active_name = str(memory_state.get("active_customer_name", "")).strip().lower()
            provided_name = str(merged_entities.get("name", "")).strip().lower()
            if active_name and provided_name:
                verification_entities["name_confirmed"] = active_name == provided_name

        if memory is not None and (merged_entities or merged_descriptions):
            memory.set_state(
                extracted_entities=merged_entities,
                extracted_entity_descriptions=merged_descriptions,
                verification_entities=verification_entities,
                verification_collected=verification_entities,
            )
        if memory is not None and callable(self.reconcile_callback):
            self.reconcile_callback(memory=memory)

        memory_state = dict(getattr(memory, "state", {})) if memory is not None else {}
        extracted_entities = (
            dict(memory_state.get("extracted_entities", {}))
            if isinstance(memory_state.get("extracted_entities"), dict)
            else {}
        )
        extracted_entity_descriptions = (
            dict(memory_state.get("extracted_entity_descriptions", {}))
            if isinstance(memory_state.get("extracted_entity_descriptions"), dict)
            else {}
        )
        verification_entities = (
            dict(memory_state.get("verification_entities", {}))
            if isinstance(memory_state.get("verification_entities"), dict)
            else {}
        )
        identity_verified = bool(memory_state.get("identity_verified", False))
        required_fields = [str(x).strip() for x in required_fields if str(x).strip()]
        missing_required = [field for field in required_fields if not str(verification_entities.get(field, "")).strip()]

        # Provide entity context early so downstream prompt templates can use it.
        context = dict(state.get("memory_context", {})) if isinstance(state.get("memory_context"), dict) else {}
        context["entities"] = extracted_entities
        context["entity_descriptions"] = extracted_entity_descriptions
        context["verification_entities"] = verification_entities
        context["identity_verified"] = identity_verified
        context["verification_missing_fields"] = missing_required

        return {
            "extracted_entities": extracted_entities,
            "extracted_entity_descriptions": extracted_entity_descriptions,
            "verification_entities": verification_entities,
            "identity_verified": identity_verified,
            "verification_missing_fields": missing_required,
            "entity_extraction_source": ("callback_fallback" if used_callback_fallback else "llm"),
            "memory_context": context,
        }

    def _extract_with_llm(
        self,
        *,
        user_input: str,
        required_fields: list[str],
        existing_entities: dict[str, str],
        active_customer_name: str,
    ) -> _EntityPayload | None:
        try:
            user_prompt = self._render_prompt_template(
                self.user_prompt,
                {
                    "user_input": user_input,
                    "required_fields_json": json.dumps(required_fields, ensure_ascii=True),
                    "active_customer_name": active_customer_name,
                    "existing_entities_json": json.dumps(existing_entities, ensure_ascii=True),
                },
            )
            payload = StructuredOutputRunner(self.llm, max_retries=2).run(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                schema=_EntityPayload,
            )
        except Exception:
            return None

        entities: dict[str, str] = {}
        descriptions: dict[str, str] = {}
        for key, value in dict(payload.entities).items():
            key_norm = str(key).strip()
            val_norm = str(value).strip()
            if key_norm and val_norm:
                entities[key_norm] = val_norm
        for key, value in dict(payload.entity_descriptions).items():
            key_norm = str(key).strip()
            val_norm = str(value).strip()
            if key_norm and val_norm:
                descriptions[key_norm] = val_norm
        return _EntityPayload(entities=entities, entity_descriptions=descriptions)

    @staticmethod
    def _render_prompt_template(template: str, values: dict[str, Any]) -> str:
        rendered = template
        for key, value in values.items():
            rendered = rendered.replace(f"{{{key}}}", str(value))
        return rendered
