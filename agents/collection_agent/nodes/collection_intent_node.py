"""Collection-specific intent node with namespaced state output keys."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from pydantic import BaseModel, Field

from agents.collection_agent.llm_structured import StructuredOutputRunner
from src.nodes.intent_node import IntentNode
from src.nodes.types import AgentState, NodeUpdate


class _IntentPayload(BaseModel):
    intent: str
    confidence: float = Field(default=0.0)
    reason: str | None = None


@dataclass(slots=True)
class CollectionIntentNode(IntentNode):
    """Writes and routes intent payloads using a dedicated state key."""

    output_key: str = "intent"
    allow_deterministic_fallback: bool = False
    enable_relevance_guard: bool = True

    def classify(
        self,
        *,
        user_input: str,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        intent_labels: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        labels = intent_labels if intent_labels is not None else self.intent_labels
        rendered_system_prompt = system_prompt if system_prompt is not None else self.system_prompt
        rendered_user_prompt = self._render_user_prompt(
            user_prompt=user_prompt if user_prompt is not None else (self.user_prompt or "{user_input}"),
            user_input=user_input,
            intent_labels=labels,
        )
        context_block = self._build_intent_context_block(context or {})
        if context_block:
            rendered_user_prompt = f"{rendered_user_prompt}\n\n{context_block}"
        debug_payload: dict[str, Any] = {
            "prompt": rendered_user_prompt,
            "system_prompt": rendered_system_prompt or None,
            "llm_response": None,
            "llm_error": None,
        }

        if self.llm is None:
            deterministic = self._deterministic_classify(user_input=user_input, labels=labels)
            deterministic["_debug"] = debug_payload
            return deterministic

        try:
            runner = StructuredOutputRunner(self.llm, max_retries=2)
            payload = runner.run(
                system_prompt=rendered_system_prompt or "",
                user_prompt=rendered_user_prompt,
                schema=_IntentPayload,
            )
            debug_payload["llm_response"] = payload.model_dump(mode="json")
            intent_name = self._normalize_intent_name(payload.intent)
            if labels and intent_name not in {self._normalize_intent_name(v) for v in labels}:
                raise ValueError(f"Intent `{payload.intent}` is not in allowed labels: {labels}")
            return {
                "intent": intent_name or self.default_intent,
                "confidence": float(payload.confidence),
                "reason": payload.reason,
                "_debug": debug_payload,
            }
        except Exception as exc:
            debug_payload["llm_error"] = str(exc)
            if not self.allow_deterministic_fallback:
                raise RuntimeError(
                    "CollectionIntentNode failed to produce structured LLM intent output. "
                    f"Fallback disabled. Error: {exc}"
                ) from exc
            deterministic = self._deterministic_classify(user_input=user_input, labels=labels)
            deterministic["_debug"] = debug_payload
            return deterministic

    def execute(self, state: AgentState) -> NodeUpdate:
        self._record_llm_usage(state, node_name="intent")
        if self.llm is None and not self.allow_deterministic_fallback:
            raise RuntimeError(
                "CollectionIntentNode requires an active LLM. "
                "Deterministic intent fallback is disabled for collection agent."
            )
        context = self._build_context_for_intent(state)
        intent = self.classify(
            user_input=state["user_input"],
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
            context=context,
        )
        pre_guard_debug = intent.get("_debug") if isinstance(intent, dict) else {}
        intent = self._apply_relevance_guard(
            intent=intent,
            user_input=str(state.get("user_input", "")),
            context=context,
        )
        if isinstance(intent, dict) and "_debug" not in intent and isinstance(pre_guard_debug, dict):
            intent["_debug"] = pre_guard_debug
        debug_payload = intent.pop("_debug", {}) if isinstance(intent, dict) else {}
        update: NodeUpdate = {
            self.output_key: intent,
            # Compatibility channel for shared routing assumptions in existing
            # graph/runtime reducers. This can be removed after full migration.
            "intent": intent,
            "prompt": debug_payload.get("prompt"),
            "system_prompt": debug_payload.get("system_prompt"),
            "llm_response": debug_payload.get("llm_response"),
            "llm_error": debug_payload.get("llm_error"),
        }
        intent_name = self._normalize_intent_name(intent.get("intent") if isinstance(intent, dict) else None)
        mapped_response = self._lookup_mapped_value(self.response_map, intent_name)
        if mapped_response is not None:
            update["response"] = mapped_response
        return update

    def route(self, state: AgentState) -> str:
        intent_payload = state.get(self.output_key)
        if intent_payload is None:
            intent_payload = state.get("intent")
        intent_name = None
        if isinstance(intent_payload, dict):
            intent_name = intent_payload.get("intent")
        normalized = self._normalize_intent_name(intent_name)
        route = self._lookup_mapped_value(self.route_map, normalized)
        return route if route is not None else self.default_route

    @staticmethod
    def _build_intent_context_block(context: dict[str, Any]) -> str:
        items = []
        for key in (
            "message_source",
            "conversation_phase",
            "active_case_id",
            "active_user_id",
            "active_customer_name",
            "identity_verified",
            "verification_missing_fields",
            "extracted_entities",
            "extracted_entity_descriptions",
            "verification_entities",
            "last_response_target",
            "plan_current_node_id",
            "plan_current_node_label",
            "last_agent_response",
        ):
            value = context.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            items.append(f"- {key}: {text}")
        if not items:
            return ""
        return "Conversation context:\n" + "\n".join(items)

    def _build_context_for_intent(self, state: AgentState) -> dict[str, Any]:
        memory = state.get("memory")
        memory_state = dict(getattr(memory, "state", {})) if memory is not None else {}
        plan = memory_state.get("active_conversation_plan")
        if not isinstance(plan, dict):
            plan = state.get("conversation_plan") if isinstance(state.get("conversation_plan"), dict) else {}

        current_node_id = str(plan.get("current_node_id", "")).strip() if isinstance(plan, dict) else ""
        current_node_label = ""
        if current_node_id and isinstance(plan, dict):
            for node in plan.get("nodes", []):
                if isinstance(node, dict) and str(node.get("id", "")).strip() == current_node_id:
                    current_node_label = str(node.get("label", "")).strip()
                    break

        return {
            "message_source": state.get("message_source") or memory_state.get("last_message_source"),
            "conversation_phase": state.get("conversation_phase"),
            "active_case_id": state.get("case_id") or memory_state.get("active_case_id"),
            "active_user_id": state.get("user_id") or memory_state.get("active_user_id"),
            "active_customer_name": memory_state.get("active_customer_name"),
            "identity_verified": bool(memory_state.get("identity_verified", False)),
            "verification_missing_fields": state.get("verification_missing_fields")
            or memory_state.get("verification_missing_fields"),
            "extracted_entities": state.get("extracted_entities") or memory_state.get("extracted_entities"),
            "verification_entities": state.get("verification_entities") or memory_state.get("verification_entities"),
            "extracted_entity_descriptions": state.get("extracted_entity_descriptions")
            or memory_state.get("extracted_entity_descriptions"),
            "last_response_target": memory_state.get("last_response_target"),
            "last_agent_response": memory_state.get("last_agent_response"),
            "plan_current_node_id": current_node_id,
            "plan_current_node_label": current_node_label,
        }

    def _apply_relevance_guard(
        self,
        *,
        intent: dict[str, Any],
        user_input: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.enable_relevance_guard:
            return intent
        if self.output_key != "relevance_intent":
            return intent
        current_intent = self._normalize_intent_name(intent.get("intent"))
        if current_intent in {"relevant", "empty"}:
            return intent
        if not self._is_active_collections_context(context):
            return intent
        if not self._looks_like_identity_or_verification_reply(user_input):
            return intent
        return {
            "intent": "relevant",
            "confidence": max(float(intent.get("confidence", 0.0) or 0.0), 0.88),
            "reason": "Context-aware relevance guard: identity/verification reply during active collections flow.",
        }

    @staticmethod
    def _is_active_collections_context(context: dict[str, Any]) -> bool:
        has_case = bool(str(context.get("active_case_id", "")).strip())
        last_response = str(context.get("last_agent_response", "")).lower()
        verify_hints = (
            "confirm your identity",
            "confirm your full name",
            "last 4 digits",
            "account number",
            "overdue amount",
            "payment today",
            "collections team",
        )
        has_verify_prompt = any(hint in last_response for hint in verify_hints)
        return has_case or has_verify_prompt

    @staticmethod
    def _looks_like_identity_or_verification_reply(text: str) -> bool:
        normalized = str(text or "").strip().lower()
        if not normalized:
            return False
        patterns = (
            r"\bmy name is\b",
            r"\bi am\b",
            r"\bthis is\b",
            r"\bmy full name\b",
            r"\blast\s*4\b",
            r"\baccount\b",
            r"\bending\b",
            r"\bxxxx\b",
            r"\b\d{4}\b",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)
