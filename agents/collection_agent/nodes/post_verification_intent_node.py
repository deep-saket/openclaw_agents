"""Post-verification routing intent node for collection agent."""

from __future__ import annotations

from typing import Any

from agents.collection_agent.nodes.collection_intent_node import CollectionIntentNode
from src.nodes.types import AgentState


class PostVerificationIntentNode(CollectionIntentNode):
    """Routes after verification workflow completes its current tool loop."""

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
            output_key="post_verification_intent",
            intent_labels=["plan", "react"],
            default_intent="plan",
            default_confidence=0.4,
            route_map={
                "plan": "plan",
                "react": "react",
                "unknown": "plan",
            },
            default_route="plan",
            fallback_keyword_map={},
        )

    def _build_context_for_intent(self, state: AgentState) -> dict[str, Any]:
        memory_state = self._get_memory_state(state)
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
        return {
            "message_source": state.get("message_source") or memory_state.get("last_message_source"),
            "conversation_phase": state.get("conversation_phase"),
            "active_case_id": state.get("case_id") or memory_state.get("active_case_id"),
            "active_user_id": state.get("user_id") or memory_state.get("active_user_id"),
            "identity_verified": bool(state.get("identity_verified", memory_state.get("identity_verified", False))),
            "conversation_mode": state.get("conversation_mode") or memory_state.get("conversation_mode"),
            "negotiation_stage": state.get("negotiation_stage") or memory_state.get("negotiation_stage"),
            "customer_payment_posture": state.get("customer_payment_posture")
            or memory_state.get("customer_payment_posture"),
            "hardship_context": state.get("hardship_context") or memory_state.get("hardship_context"),
            "response_mode": state.get("response_mode") or memory_state.get("response_mode"),
            "active_dialogue_owner": state.get("active_dialogue_owner") or memory_state.get("active_dialogue_owner"),
            "verification_missing_fields": state.get("verification_missing_fields")
            or memory_state.get("verification_missing_fields"),
            "verification_verified_fields": state.get("verification_verified_fields")
            or memory_state.get("verification_verified_fields"),
            "verification_entities": state.get("verification_entities") or memory_state.get("verification_entities"),
            "extracted_entities_turn": state.get("extracted_entities_turn") or memory_state.get("extracted_entities_turn"),
            "last_agent_response": memory_state.get("last_agent_response"),
            "last_response_target": memory_state.get("last_response_target"),
            "observation_tool_name": obs_payload.get("tool_name"),
            "observation_status": observation_status,
            "observation_output": obs_payload.get("output") if isinstance(obs_payload.get("output"), dict) else {},
            "turn_index": memory_state.get("turn_index"),
        }

    def _apply_node_specific_pre_rule(
        self,
        *,
        state: AgentState,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        del state
        if not bool(context.get("identity_verified", False)):
            return {
                "skip_llm": True,
                "reason": "Verification remains incomplete; return to plan/response path to ask only for missing fields.",
                "intent": {
                    "intent": "plan",
                    "confidence": 0.99,
                    "reason": "Identity verification is still incomplete after verification path.",
                },
            }
        return None

    def _apply_node_specific_intent_override(
        self,
        *,
        intent: dict[str, Any],
        state: AgentState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        del state, context
        return intent
