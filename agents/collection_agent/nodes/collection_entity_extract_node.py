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
    field_evidence: dict[str, str] = Field(default_factory=dict)


@dataclass(slots=True)
class CollectionEntityExtractNode(BaseGraphNode):
    """Extracts entities into memory/state immediately after relevance gating.

    State Keys Read:
    - `user_input`
    - `conversation_history`
    - `memory` (reads and mutates `memory.state`, including extracted and verification entities)

    State Keys Write:
    - `extracted_entities`
    - `extracted_entities_turn`
    - `extracted_entity_descriptions`
    - `verification_entities`
    - `customer_payment_capacity`
    - `customer_payment_capacity_pct`
    - `extracted_entities_updated_fields`
    - `entity_extraction_source`
    - `prompt`
    - `system_prompt`
    - `llm_response`
    - `llm_error`
    """

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
        conversation_history = (
            state.get("conversation_history")
            if isinstance(state.get("conversation_history"), list)
            else (memory_state.get("conversation_history") if isinstance(memory_state.get("conversation_history"), list) else [])
        )
        conversation_history_compact = self._compact_conversation_history(conversation_history)
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
        llm_debug: dict[str, Any] = {
            "prompt": None,
            "system_prompt": self.system_prompt or None,
            "llm_response": None,
            "llm_error": None,
        }
        if self.llm is not None and user_input.strip():
            extracted, llm_debug = self._extract_with_llm(
                user_input=user_input,
                required_fields=required_fields,
                existing_entities=existing_entities,
                existing_entity_descriptions=existing_descriptions,
                active_customer_name=str(memory_state.get("active_customer_name", "")).strip(),
                conversation_history_compact=conversation_history_compact,
            )
            if extracted is not None:
                llm_entities = extracted.entities
                llm_descriptions = extracted.entity_descriptions

        # LLM-first, deterministic backfill when required verification fields are still missing.
        llm_merged_preview = dict(existing_entities)
        llm_merged_preview.update(
            {
                str(key).strip(): str(value).strip()
                for key, value in llm_entities.items()
                if str(key).strip() and str(value).strip()
            }
        )
        missing_required_after_llm = [
            field for field in required_fields if not str(llm_merged_preview.get(field, "")).strip()
        ]
        should_backfill = (not llm_entities) or bool(missing_required_after_llm)

        if should_backfill and callable(self.extract_callback) and self.allow_callback_fallback and memory is not None:
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
        turn_entities: dict[str, str] = {}
        updated_fields: list[str] = []
        for key, value in llm_entities.items():
            key_norm = str(key).strip()
            val_norm = str(value).strip()
            if key_norm and val_norm:
                turn_entities[key_norm] = val_norm
                prior_val = str(existing_entities.get(key_norm, "")).strip()
                if prior_val and prior_val != val_norm:
                    updated_fields.append(key_norm)
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
        customer_payment_capacity = self._normalize_optional_float(
            llm_entities.get("customer_payment_capacity", memory_state.get("customer_payment_capacity"))
        )
        customer_payment_capacity_pct = self._normalize_optional_pct(
            llm_entities.get("customer_payment_capacity_pct", memory_state.get("customer_payment_capacity_pct"))
        )
        # Verification entities should only be updated from current-turn extracted values
        # to avoid stale/hallucinated carry-forward.
        for field in required_fields:
            val = str(llm_entities.get(field, "")).strip()
            if val:
                verification_entities[field] = val
        if str(llm_entities.get("name", "")).strip():
            verification_entities["name"] = str(llm_entities["name"]).strip()
            active_name = str(memory_state.get("active_customer_name", "")).strip().lower()
            provided_name = str(llm_entities.get("name", "")).strip().lower()
            if active_name and provided_name:
                verification_entities["name_confirmed"] = active_name == provided_name

        if memory is not None and (merged_entities or merged_descriptions):
            memory.set_state(
                extracted_entities=merged_entities,
                extracted_entities_turn=turn_entities,
                extracted_entity_descriptions=merged_descriptions,
                verification_entities=verification_entities,
                verification_collected=verification_entities,
                customer_payment_capacity=customer_payment_capacity,
                customer_payment_capacity_pct=customer_payment_capacity_pct,
                extracted_entities_updated_fields=sorted(set(updated_fields)),
            )
        if memory is not None and callable(self.reconcile_callback):
            self.reconcile_callback(memory=memory)

        memory_state = dict(getattr(memory, "state", {})) if memory is not None else {}
        extracted_entities = (
            dict(memory_state.get("extracted_entities", {}))
            if isinstance(memory_state.get("extracted_entities"), dict)
            else {}
        )
        extracted_entities_turn = (
            dict(memory_state.get("extracted_entities_turn", {}))
            if isinstance(memory_state.get("extracted_entities_turn"), dict)
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

        return {
            "extracted_entities": extracted_entities,
            "extracted_entities_turn": extracted_entities_turn,
            "extracted_entity_descriptions": extracted_entity_descriptions,
            "verification_entities": verification_entities,
            "customer_payment_capacity": customer_payment_capacity,
            "customer_payment_capacity_pct": customer_payment_capacity_pct,
            "extracted_entities_updated_fields": sorted(set(updated_fields)),
            "entity_extraction_source": ("callback_fallback" if used_callback_fallback else "llm"),
            "prompt": llm_debug.get("prompt"),
            "system_prompt": llm_debug.get("system_prompt"),
            "llm_response": llm_debug.get("llm_response"),
            "llm_error": llm_debug.get("llm_error"),
        }

    def _extract_with_llm(
        self,
        *,
        user_input: str,
        required_fields: list[str],
        existing_entities: dict[str, str],
        existing_entity_descriptions: dict[str, str],
        active_customer_name: str,
        conversation_history_compact: str,
    ) -> tuple[_EntityPayload | None, dict[str, Any]]:
        debug_payload: dict[str, Any] = {
            "prompt": None,
            "system_prompt": self.system_prompt or None,
            "llm_response": None,
            "llm_error": None,
        }
        try:
            user_prompt = self._render_prompt_template(
                self.user_prompt,
                {
                    "user_input": user_input,
                    "required_fields_json": json.dumps(required_fields, ensure_ascii=True),
                    "active_customer_name": active_customer_name,
                    "existing_entities_json": json.dumps(existing_entities, ensure_ascii=True),
                    "existing_entity_descriptions_json": json.dumps(existing_entity_descriptions, ensure_ascii=True),
                    "conversation_history_compact": conversation_history_compact,
                },
            )
            debug_payload["prompt"] = user_prompt
            payload = StructuredOutputRunner(self.llm, max_retries=2).run(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                schema=_EntityPayload,
            )
            debug_payload["llm_response"] = payload.model_dump(mode="json")
        except Exception as exc:
            debug_payload["llm_error"] = str(exc)
            return None, debug_payload

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
        return _EntityPayload(entities=entities, entity_descriptions=descriptions), debug_payload

    @staticmethod
    def _compact_conversation_history(history: list[dict[str, Any]], *, max_turns: int = 16, max_chars: int = 1800) -> str:
        if not history:
            return ""
        rows: list[str] = []
        for row in history[-max_turns:]:
            if not isinstance(row, dict):
                continue
            role = str(row.get("role", "")).strip() or str(row.get("speaker", "")).strip() or "unknown"
            text = str(row.get("content", "")).strip() or str(row.get("text", "")).strip()
            if not text:
                continue
            rows.append(f"{role}: {text}")
        joined = "\n".join(rows)
        if len(joined) <= max_chars:
            return joined
        return joined[-max_chars:]

    @staticmethod
    def _render_prompt_template(template: str, values: dict[str, Any]) -> str:
        rendered = template
        for key, value in values.items():
            rendered = rendered.replace(f"{{{key}}}", str(value))
        return rendered

    @staticmethod
    def _normalize_optional_float(value: Any) -> float | None:
        if value is None:
            return None
        text = str(value).strip().replace(",", "")
        if not text:
            return None
        try:
            return float(text)
        except Exception:
            return None

    @staticmethod
    def _normalize_optional_pct(value: Any) -> float | None:
        if value is None:
            return None
        text = str(value).strip().replace("%", "").replace(",", "")
        if not text:
            return None
        try:
            pct = float(text)
        except Exception:
            return None
        return max(0.0, min(100.0, pct))
