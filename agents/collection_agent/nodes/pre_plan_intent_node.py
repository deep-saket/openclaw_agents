"""Pre-plan intent node for collection agent."""

from __future__ import annotations

import re
from typing import Any

from agents.collection_agent.nodes.collection_intent_node import CollectionIntentNode
from src.nodes.types import AgentState


class PrePlanIntentNode(CollectionIntentNode):
    """Intent node dedicated to pre-plan routing.

    State Keys Read:
    - `user_input`
    - `message_source`
    - `conversation_phase`
    - `case_id`
    - `user_id`
    - `conversation_plan`
    - `observations`
    - `observation` (latest compatibility mirror)
    - `verification_verified_fields`
    - `verification_missing_fields`
    - `verification_entities`
    - `extracted_entities`
    - `extracted_entities_turn`
    - `extracted_entity_descriptions`
    - `memory` (reads `memory.state`, including tool observation history and customer context)

    State Keys Write:
    - `pre_plan_intent`
    - `intent` (compatibility mirror)
    - `prompt`
    - `system_prompt`
    - `llm_response`
    - `llm_error`
    - `llm_status`
    - `fallback_reason` (optional)
    """
    max_tool_catalog_chars: int = 1400
    max_tool_observations: int = 10

    def __init__(
        self,
        *,
        llm: Any | None,
        allow_deterministic_fallback: bool,
        system_prompt: str,
        user_prompt: str,
    ) -> None:
        super().__init__(
            llm=llm,
            allow_deterministic_fallback=allow_deterministic_fallback,
            allow_rate_limit_fallback=False,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_key="pre_plan_intent",
            intent_labels=["plan", "decide"],
            default_intent="decide",
            default_confidence=0.4,
            route_map={
                "plan": "plan",
                "decide": "decide",
                "unknown": "decide",
            },
            default_route="decide",
            fallback_keyword_map={},
        )

    def _build_context_for_intent(self, state: AgentState) -> dict[str, Any]:
        from agents.collection_agent.prompts import render_collection_tool_catalog_yaml

        memory_state = self._get_memory_state(state)
        plan = memory_state.get("active_conversation_plan")
        if not isinstance(plan, dict):
            plan = state.get("conversation_plan") if isinstance(state.get("conversation_plan"), dict) else {}

        current_node_id = str(plan.get("current_node_id", "")).strip() if isinstance(plan, dict) else ""
        current_node_label = ""
        next_node_ids: list[str] = []
        if isinstance(plan, dict):
            raw_next = plan.get("next_node_ids")
            if isinstance(raw_next, list):
                next_node_ids = [str(x).strip() for x in raw_next if str(x).strip()]
            if current_node_id:
                for node in plan.get("nodes", []):
                    if isinstance(node, dict) and str(node.get("id", "")).strip() == current_node_id:
                        current_node_label = str(node.get("label", "")).strip()
                        break
        observation = None
        observations = state.get("observations")
        if isinstance(observations, list):
            for item in reversed(observations):
                if isinstance(item, dict):
                    observation = item
                    break
        if observation is None:
            observation = state.get("observation")
        obs_payload: dict[str, Any] = {}
        if isinstance(observation, dict):
            tool_phase = observation.get("tool_phase")
            if isinstance(tool_phase, dict):
                obs_payload = tool_phase
            else:
                obs_payload = observation
        observation_status = obs_payload.get("status")
        if observation_status is None and isinstance(obs_payload.get("output"), dict):
            observation_status = obs_payload.get("output", {}).get("status")
        tool_observations_history = (
            list(memory_state.get("tool_observations_history", []))
            if isinstance(memory_state.get("tool_observations_history"), list)
            else []
        )

        tool_catalog_text = self._compact_text(
            render_collection_tool_catalog_yaml(),
            self.max_tool_catalog_chars,
        )

        return {
            "message_source": state.get("message_source") or memory_state.get("last_message_source"),
            "conversation_phase": state.get("conversation_phase"),
            "active_case_id": state.get("case_id") or memory_state.get("active_case_id"),
            "active_user_id": state.get("user_id") or memory_state.get("active_user_id"),
            "active_customer_name": memory_state.get("active_customer_name"),
            "identity_verified": bool(state.get("identity_verified", memory_state.get("identity_verified", False))),
            "conversation_mode": state.get("conversation_mode") or memory_state.get("conversation_mode"),
            "negotiation_stage": state.get("negotiation_stage") or memory_state.get("negotiation_stage"),
            "customer_payment_posture": state.get("customer_payment_posture")
            or memory_state.get("customer_payment_posture"),
            "hardship_context": state.get("hardship_context") or memory_state.get("hardship_context"),
            "response_mode": state.get("response_mode") or memory_state.get("response_mode"),
            "active_dialogue_owner": state.get("active_dialogue_owner") or memory_state.get("active_dialogue_owner"),
            "active_verification_required_fields": memory_state.get("active_verification_required_fields"),
            "verification_verified_fields": state.get("verification_verified_fields")
            or memory_state.get("verification_verified_fields"),
            "verification_missing_fields": state.get("verification_missing_fields")
            or memory_state.get("verification_missing_fields"),
            "verification_entities": state.get("verification_entities") or memory_state.get("verification_entities"),
            "extracted_entities": state.get("extracted_entities") or memory_state.get("extracted_entities"),
            "extracted_entities_turn": state.get("extracted_entities_turn") or memory_state.get("extracted_entities_turn"),
            "extracted_entity_descriptions": state.get("extracted_entity_descriptions")
            or memory_state.get("extracted_entity_descriptions"),
            "observation_tool_name": obs_payload.get("tool_name"),
            "observation_status": observation_status,
            "observation_output": obs_payload.get("output") if isinstance(obs_payload.get("output"), dict) else {},
            "tool_observations_history": tool_observations_history[-self.max_tool_observations :],
            "tool_catalog_compact": tool_catalog_text,
            "last_agent_response": memory_state.get("last_agent_response"),
            "last_response_target": memory_state.get("last_response_target"),
            "plan_current_node_id": current_node_id,
            "plan_current_node_label": current_node_label,
            "plan_next_node_ids": next_node_ids,
            "turn_index": memory_state.get("turn_index"),
        }

    @staticmethod
    def _compact_text(value: str, max_chars: int) -> str:
        text = str(value or "").strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + " ...[truncated]"

    def _apply_node_specific_pre_rule(
        self,
        *,
        state: AgentState,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        identity_verified = bool(context.get("identity_verified", False))
        if identity_verified:
            return None

        verified_raw = context.get("verification_verified_fields")
        verified_list = verified_raw if isinstance(verified_raw, list) else []
        verified_fields = {
            str(x).strip().lower()
            for x in verified_list
            if str(x).strip()
        }

        entities_turn_raw = context.get("extracted_entities_turn")
        entities_turn = dict(entities_turn_raw) if isinstance(entities_turn_raw, dict) else {}
        provided_from_entities = {
            str(k).strip().lower()
            for k, v in entities_turn.items()
            if str(k).strip() and str(v).strip()
        }
        provided_obvious = self._detect_obvious_verification_fields(str(state.get("user_input", "")))
        provided_fields = provided_from_entities | provided_obvious

        # Simple explicit policy:
        # - if verification incomplete and dob not verified yet but dob provided -> decide
        # - if verification incomplete and phone not verified yet but phone provided -> decide
        dob_needs_check = ("dob" not in verified_fields) and ("dob" in provided_fields)
        phone_needs_check = ("phone" not in verified_fields) and ("phone" in provided_fields)
        if not (dob_needs_check or phone_needs_check):
            return None

        matched_fields: list[str] = []
        if dob_needs_check:
            matched_fields.append("dob")
        if phone_needs_check:
            matched_fields.append("phone")

        # Pre-plan rule: skip LLM and force orchestration path when current turn
        # provides verification evidence for a field that is not yet verified.
        return {
            "skip_llm": True,
            "reason": "Pre-plan pre-rule: unverified verification field evidence present in current turn.",
            "intent": {
                "intent": "decide",
                "confidence": 0.98,
                "reason": f"Verification incomplete and turn provided: {', '.join(matched_fields)}.",
            },
        }

    def _apply_node_specific_intent_override(
        self,
        *,
        intent: dict[str, Any],
        state: AgentState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        # No post-override at pre-plan node.
        return intent

    @staticmethod
    def _detect_obvious_verification_fields(text: str) -> set[str]:
        lowered = str(text or "").lower()
        fields: set[str] = set()
        if re.search(r"\b(19|20)\d{2}-\d{2}-\d{2}\b", lowered):
            fields.add("dob")
        digits = re.sub(r"\D", "", lowered)
        if len(digits) >= 10:
            fields.add("phone")
        return fields
