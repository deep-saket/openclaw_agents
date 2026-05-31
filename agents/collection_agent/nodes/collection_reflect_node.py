"""Reflect node variant for collection demo loops."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from src.nodes.reflect_node import ReflectNode
from src.nodes.types import AgentState, NodeUpdate


VALID_FAILURE_TYPES = {
    "invalid_json",
    "empty_response",
    "policy_violation",
    "missing_required_action",
    "unsafe_disclosure",
    "placeholder_leakage",
    "invalid_state_claim",
    "none",
}

RETRYABLE_FAILURES = {
    "invalid_json",
    "empty_response",
}


@dataclass(slots=True)
class CollectionReflectNode(ReflectNode):
    """Collection-specific routing with LLM-driven reflection judgment.

    State Keys Read:
    - `user_input`
    - `observations`
    - `observation` (latest compatibility mirror)
    - `decision`
    - `routing_context`
    - `plan_proposal`
    - `conversation_plan`
    - `response_target`
    - `response`
    - `identity_verified`
    - `verification_missing_fields`
    - `verification_entities`
    - `reflection_plan_retry_count`

    State Keys Write:
    - `reflection_feedback`
    - `reflection_complete`
    - `reflection_retry_count`
    - `reflection_plan_retry_count`
    - `retry_target`
    - `failure_type`
    - `correction_hints`
    - base reflect debug keys (if emitted by parent):
      `prompt`, `system_prompt`, `llm_response`, `llm_error`
    """
    max_retry_loops: int = 2

    def execute(self, state: AgentState) -> NodeUpdate:
        routing_context = state.get("routing_context") if isinstance(state.get("routing_context"), dict) else {}
        plan_origin = str(routing_context.get("plan_origin", "react"))
        reflect_state: AgentState = dict(state)
        if plan_origin in {"pre_plan_intent", "post_memory_plan_intent"}:
            # For planning path validation, reflect must evaluate the proposal payload,
            # not the raw user greeting/utterance in isolation.
            plan_observation = self._build_plan_reflection_observation(state)
            reflect_state["observation"] = plan_observation
            prior_observations = (
                list(reflect_state.get("observations", []))
                if isinstance(reflect_state.get("observations"), list)
                else []
            )
            prior_observations.append(plan_observation)
            reflect_state["observations"] = prior_observations

        # Explicit base-class invocation avoids the dataclass(slots=True) +
        # zero-arg super() runtime edge case observed in this environment.
        result = ReflectNode.execute(self, reflect_state)
        feedback = result.get("reflection_feedback") if isinstance(result.get("reflection_feedback"), dict) else {}
        complete = bool(result.get("reflection_complete", self.default_is_complete))
        retries = int(state.get("reflection_plan_retry_count", 0) or 0)

        normalized = self._normalize_reflection_result(
            feedback=feedback,
            complete=complete,
            state=reflect_state,
            llm_response=result.get("llm_response"),
            llm_error=result.get("llm_error"),
        )
        feedback = normalized["feedback"]
        complete = normalized["complete"]
        result["reflection_feedback"] = feedback
        result["reflection_complete"] = complete
        result["failure_type"] = normalized["failure_type"]
        result["correction_hints"] = normalized["correction_hints"]
        result["retry_target"] = "plan_proposal" if normalized["retryable"] else "none"
        if complete:
            result["reflection_retry_count"] = 0
            result["reflection_plan_retry_count"] = 0
            return result

        retries += 1
        result["reflection_retry_count"] = retries
        result["reflection_plan_retry_count"] = retries
        if retries >= max(1, int(self.max_retry_loops)):
            feedback = dict(result.get("reflection_feedback", {})) if isinstance(result.get("reflection_feedback"), dict) else {}
            feedback["reason"] = (
                f"{feedback.get('reason', 'Reflection retry limit reached.')} Continuing without another retry."
            )
            feedback["is_complete"] = True
            result["reflection_feedback"] = feedback
            result["reflection_complete"] = True
            result["reflection_retry_count"] = 0
            result["reflection_plan_retry_count"] = 0
            result["retry_target"] = "none"
        return result

    @staticmethod
    def _build_plan_reflection_observation(state: AgentState) -> dict[str, Any]:
        plan_proposal_raw = state.get("plan_proposal") if isinstance(state.get("plan_proposal"), dict) else {}
        plan_proposal = {
            "target": str(plan_proposal_raw.get("target", "")).strip(),
            "intent": str(plan_proposal_raw.get("intent", "")).strip(),
            "plan_outline": str(plan_proposal_raw.get("plan_outline", "")).strip(),
            "draft_response": str(plan_proposal_raw.get("draft_response", "")).strip(),
            "next_actions": [str(x).strip() for x in (plan_proposal_raw.get("next_actions") or []) if str(x).strip()][:8],
            "plan_tree_update": plan_proposal_raw.get("plan_tree_update", {}),
            "conversation_plan_current_node": str(plan_proposal_raw.get("conversation_plan_current_node", "")).strip(),
            "response_directive": plan_proposal_raw.get("response_directive", {}),
            "compiled_response_directive": plan_proposal_raw.get("compiled_response_directive", {}),
            "handoff_payload": (
                plan_proposal_raw.get("handoff_payload", {})
                if isinstance(plan_proposal_raw.get("handoff_payload"), dict)
                else (
                    state.get("handoff_payload", {})
                    if isinstance(state.get("handoff_payload"), dict)
                    else {}
                )
            ),
        }
        conversation_plan = state.get("conversation_plan") if isinstance(state.get("conversation_plan"), dict) else {}
        plan_node_ctx = {
            "current_node_id": conversation_plan.get("current_node_id"),
            "previous_node_id": conversation_plan.get("previous_node_id"),
            "next_node_ids": conversation_plan.get("next_node_ids"),
            "status": conversation_plan.get("status"),
        }
        verification_ctx = {
            "identity_verified": bool(state.get("identity_verified", False)),
            "verification_missing_fields": state.get("verification_missing_fields", []),
            "verification_entities": state.get("verification_entities", {}),
        }
        negotiation_ctx = {
            "conversation_mode": state.get("conversation_mode"),
            "negotiation_stage": state.get("negotiation_stage"),
            "customer_payment_posture": state.get("customer_payment_posture"),
            "customer_payment_capacity": state.get("customer_payment_capacity"),
            "customer_payment_capacity_pct": state.get("customer_payment_capacity_pct"),
            "customer_payment_willingness": state.get("customer_payment_willingness"),
            "customer_payment_posture_history": state.get("customer_payment_posture_history", []),
            "discount_stage": state.get("discount_stage"),
            "discount_requested": state.get("discount_requested"),
            "discount_offered": state.get("discount_offered"),
            "discount_accepted": state.get("discount_accepted"),
            "discount_rejected": state.get("discount_rejected"),
            "counter_offer_present": state.get("counter_offer_present"),
            "hardship_context": state.get("hardship_context", {}),
        }
        return {
            "kind": "plan_proposal_review",
            "plan_proposal": plan_proposal,
            "response_target": state.get("response_target"),
            "plan_node_context": plan_node_ctx,
            "verification_context": verification_ctx,
            "negotiation_context": negotiation_ctx,
            "plan_signals": state.get("plan_signals", {}),
            "extracted_entities_turn": state.get("extracted_entities_turn", {}),
            "previous_response": state.get("response"),
        }

    @staticmethod
    def _normalize_reflection_result(
        *,
        feedback: dict[str, Any],
        complete: bool,
        state: AgentState,
        llm_response: Any,
        llm_error: Any,
    ) -> dict[str, Any]:
        reason = str(feedback.get("reason", "")).strip()
        parsed_feedback = CollectionReflectNode._parse_raw_reflection_payload(llm_response)
        correction_hints: list[str] = []
        failure_type = "none"

        observation = {}
        observations = state.get("observations")
        if isinstance(observations, list):
            for item in reversed(observations):
                if isinstance(item, dict):
                    observation = item
                    break
        if not observation and isinstance(state.get("observation"), dict):
            observation = dict(state.get("observation", {}))
        observed_tool = str(observation.get("tool_name", "")).strip().lower()
        observed_output = observation.get("output") if isinstance(observation.get("output"), dict) else {}
        observed_status = str(observed_output.get("status", "")).strip().lower()
        plan_payload = observation.get("plan_proposal") if isinstance(observation.get("plan_proposal"), dict) else {}
        verification_ctx = observation.get("verification_context") if isinstance(observation.get("verification_context"), dict) else {}
        negotiation_ctx = observation.get("negotiation_context") if isinstance(observation.get("negotiation_context"), dict) else {}
        plan_signals = observation.get("plan_signals") if isinstance(observation.get("plan_signals"), dict) else {}
        extracted_entities_turn = (
            observation.get("extracted_entities_turn")
            if isinstance(observation.get("extracted_entities_turn"), dict)
            else {}
        )
        identity_verified = bool(verification_ctx.get("identity_verified", False))
        next_actions = [str(x).strip().lower() for x in (plan_payload.get("next_actions") or []) if str(x).strip()]
        selected_next = (
            str(plan_payload.get("plan_tree_update", {}).get("selected_next_node_id", "")).strip().lower()
            if isinstance(plan_payload.get("plan_tree_update"), dict)
            else ""
        )
        response_target = str(
            observation.get("response_target", state.get("response_target", plan_payload.get("target", "")))
        ).strip().lower()
        customer_text = CollectionReflectNode._first_non_empty_text(
            [
                plan_payload.get("draft_response"),
                state.get("response"),
                observation.get("previous_response"),
            ]
        )
        plan_tree_update = plan_payload.get("plan_tree_update")
        compiled_response_directive = (
            plan_payload.get("compiled_response_directive")
            if isinstance(plan_payload.get("compiled_response_directive"), dict)
            else {}
        )
        legacy_response_directive = (
            plan_payload.get("response_directive")
            if isinstance(plan_payload.get("response_directive"), dict)
            else {}
        )
        handoff_payload = plan_payload.get("handoff_payload") if isinstance(plan_payload.get("handoff_payload"), dict) else {}
        current_node_id = str(
            observation.get("plan_node_context", {}).get("current_node_id", "")
            if isinstance(observation.get("plan_node_context"), dict)
            else ""
        ).strip().lower()
        customer_payment_posture = str(negotiation_ctx.get("customer_payment_posture", "")).strip().lower()
        discount_stage = str(negotiation_ctx.get("discount_stage", "")).strip().lower()
        customer_payment_capacity = negotiation_ctx.get("customer_payment_capacity")
        customer_payment_capacity_pct = negotiation_ctx.get("customer_payment_capacity_pct")
        hardship_context = negotiation_ctx.get("hardship_context") if isinstance(negotiation_ctx.get("hardship_context"), dict) else {}
        hardship_active = bool(hardship_context.get("hardship_detected", False))

        if llm_error:
            failure_type = "invalid_json"
            reason = str(llm_error).strip() or "Reflection validator returned an invalid payload."
            correction_hints.append("Retry reflection with valid strict JSON.")
        elif parsed_feedback is None:
            failure_type = "invalid_json"
            reason = "Reflection validator returned invalid JSON."
            correction_hints.append("Retry reflection with valid strict JSON.")
        elif not CollectionReflectNode._is_valid_plan_structure(plan_payload, plan_tree_update):
            failure_type = "invalid_json"
            reason = "Plan proposal structure is invalid."
            correction_hints.append("Return a structurally valid plan proposal payload.")
        elif CollectionReflectNode._is_empty_payload(
            plan_payload=plan_payload,
            response_target=response_target,
            customer_text=customer_text,
            compiled_response_directive=compiled_response_directive,
            legacy_response_directive=legacy_response_directive,
        ):
            failure_type = "empty_response"
            reason = "Plan output is empty and cannot be rendered safely."
            correction_hints.append("Provide either a render directive, a draft response, or a concrete next action.")
        elif CollectionReflectNode._has_placeholder_leakage(customer_text):
            failure_type = "placeholder_leakage"
            reason = "Customer-facing text contains placeholders or internal planning text."
        elif CollectionReflectNode._has_unsafe_disclosure(customer_text, identity_verified=identity_verified):
            failure_type = "unsafe_disclosure"
            reason = "Restricted account details were disclosed before verification."
        elif CollectionReflectNode._has_invalid_state_claim(customer_text, identity_verified=identity_verified):
            failure_type = "invalid_state_claim"
            reason = "Response claims verification completion without state evidence."
        elif not CollectionReflectNode._has_valid_negotiation_state_values(
            customer_payment_posture=customer_payment_posture,
            discount_stage=discount_stage,
        ):
            failure_type = "invalid_state_claim"
            reason = "Negotiation state is missing required posture or discount lifecycle values."
        elif CollectionReflectNode._capacity_missing_when_present_in_turn(
            user_input=str(state.get("user_input", "")),
            extracted_entities_turn=extracted_entities_turn,
            customer_payment_capacity=customer_payment_capacity,
            customer_payment_capacity_pct=customer_payment_capacity_pct,
        ):
            failure_type = "invalid_state_claim"
            reason = "Customer payment capacity was present in the turn but was not captured in state."
        elif CollectionReflectNode._should_route_to_discount_planning(
            identity_verified=identity_verified,
            customer_payment_posture=customer_payment_posture,
            discount_stage=discount_stage,
            hardship_active=hardship_active,
            plan_signals=plan_signals,
            user_input=str(state.get("user_input", "")),
        ) and response_target != "discount_planning_agent":
            failure_type = "policy_violation"
            reason = "Discount planning specialist should have been invoked for the current negotiation state."
        elif response_target == "discount_planning_agent" and not CollectionReflectNode._has_valid_discount_handoff(
            handoff_payload=handoff_payload,
            customer_payment_posture=customer_payment_posture,
            discount_stage=discount_stage,
        ):
            failure_type = "missing_required_action"
            reason = "Discount-planning handoff payload is missing required negotiation context."
        elif CollectionReflectNode._missing_required_action(
            current_node_id=current_node_id,
            selected_next=selected_next,
            next_actions=next_actions,
        ):
            failure_type = "missing_required_action"
            reason = "Payment stage is active but no concrete next action was provided."
        elif response_target and response_target not in {"customer", "self", "discount_planning_agent"}:
            failure_type = "policy_violation"
            reason = "Response target is invalid."
        else:
            failure_type = "none"
            reason = str(parsed_feedback.get("reason", "")).strip() or reason or "Deterministic validation passed."

        retryable = failure_type in RETRYABLE_FAILURES
        complete = failure_type == "none" or not retryable
        return {
            "feedback": {"reason": reason, "is_complete": bool(complete)},
            "complete": bool(complete),
            "failure_type": failure_type,
            "correction_hints": correction_hints,
            "retryable": retryable,
        }

    def route(self, state: dict[str, Any]) -> str:
        if state.get("reflection_complete", self.default_is_complete):
            return self.complete_route
        if str(state.get("failure_type", "none")).strip().lower() not in RETRYABLE_FAILURES:
            return self.complete_route
        return "retry_plan_proposal"

    @staticmethod
    def _parse_raw_reflection_payload(raw: Any) -> dict[str, Any] | None:
        text = str(raw or "").strip()
        if not text:
            return None
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            return None
        try:
            payload = json.loads(match.group(0))
        except Exception:
            return None
        failure_type = str(payload.get("failure_type", "none")).strip().lower() or "none"
        if failure_type not in VALID_FAILURE_TYPES:
            failure_type = "none"
        return {
            "reason": str(payload.get("reason", "")).strip(),
            "is_complete": bool(payload.get("is_complete", True)),
            "failure_type": failure_type,
        }

    @staticmethod
    def _is_valid_plan_structure(plan_payload: dict[str, Any], plan_tree_update: Any) -> bool:
        if not isinstance(plan_payload, dict):
            return False
        if plan_tree_update not in ({}, None) and not isinstance(plan_tree_update, dict):
            return False
        if isinstance(plan_tree_update, dict) and plan_tree_update:
            operation = str(plan_tree_update.get("operation", "")).strip()
            if not operation:
                return False
        return True

    @staticmethod
    def _is_empty_payload(
        *,
        plan_payload: dict[str, Any],
        response_target: str,
        customer_text: str,
        compiled_response_directive: dict[str, Any],
        legacy_response_directive: dict[str, Any],
    ) -> bool:
        has_next_actions = bool(plan_payload.get("next_actions"))
        has_plan_outline = bool(str(plan_payload.get("plan_outline", "")).strip())
        has_directive = bool(compiled_response_directive) or bool(legacy_response_directive)
        has_customer_text = bool(customer_text)
        if response_target == "customer":
            return not (has_customer_text or has_directive or has_next_actions or has_plan_outline)
        return not (has_customer_text or has_directive or has_next_actions)

    @staticmethod
    def _has_placeholder_leakage(text: str) -> bool:
        lowered = str(text or "").lower()
        if not lowered:
            return False
        return any(
            token in lowered
            for token in [
                "todo",
                "placeholder",
                "dummy response",
                "lorem ipsum",
                "internal planning",
                "continue internal planning",
                "[insert",
                "{missing",
                "{amount",
            ]
        ) or bool(re.search(r"\{[^}]+\}|\[[^\]]+\]", str(text or "")))

    @staticmethod
    def _has_unsafe_disclosure(text: str, *, identity_verified: bool) -> bool:
        if identity_verified:
            return False
        lowered = str(text or "").lower()
        return any(token in lowered for token in ["inr ", "overdue", "dues", "emi", "late fee"])

    @staticmethod
    def _has_invalid_state_claim(text: str, *, identity_verified: bool) -> bool:
        if identity_verified:
            return False
        lowered = str(text or "").lower()
        return any(
            token in lowered
            for token in [
                "verification is complete",
                "your identity is verified",
                "identity verification is complete",
                "now that you are verified",
            ]
        )

    @staticmethod
    def _missing_required_action(
        *,
        current_node_id: str,
        selected_next: str,
        next_actions: list[str],
    ) -> bool:
        payment_nodes = {"collect_payment_intent", "resolve_outcome"}
        payment_active = current_node_id in payment_nodes or selected_next in payment_nodes
        if not payment_active:
            return False
        return not next_actions and not selected_next

    @staticmethod
    def _has_valid_negotiation_state_values(*, customer_payment_posture: str, discount_stage: str) -> bool:
        valid_postures = {
            "unknown",
            "pay_now",
            "partial_now",
            "promise_to_pay",
            "cannot_pay",
            "refuses_to_pay",
            "negotiating",
        }
        valid_discount_stages = {
            "none",
            "requested",
            "planning",
            "offered",
            "accepted",
            "rejected",
            "counter_offer",
            "closed",
        }
        return customer_payment_posture in valid_postures and discount_stage in valid_discount_stages

    @staticmethod
    def _capacity_missing_when_present_in_turn(
        *,
        user_input: str,
        extracted_entities_turn: dict[str, Any],
        customer_payment_capacity: Any,
        customer_payment_capacity_pct: Any,
    ) -> bool:
        if extracted_entities_turn.get("customer_payment_capacity") not in {None, ""}:
            return customer_payment_capacity in {None, ""}
        if extracted_entities_turn.get("customer_payment_capacity_pct") not in {None, ""}:
            return customer_payment_capacity_pct in {None, ""}
        lowered = str(user_input or "").lower()
        mentions_capacity = any(token in lowered for token in ["can pay", "%", "percent", "half"])
        if not mentions_capacity:
            return False
        return customer_payment_capacity in {None, ""} and customer_payment_capacity_pct in {None, ""}

    @staticmethod
    def _should_route_to_discount_planning(
        *,
        identity_verified: bool,
        customer_payment_posture: str,
        discount_stage: str,
        hardship_active: bool,
        plan_signals: dict[str, Any],
        user_input: str,
    ) -> bool:
        if not identity_verified:
            return False
        lowered = str(user_input or "").lower()
        direct_discount_request = any(
            token in lowered
            for token in ["discount", "settlement", "waiver", "counter-offer", "counter offer", "partial payment"]
        )
        return bool(plan_signals.get("needs_discount_specialist")) or direct_discount_request or (
            discount_stage in {"requested", "counter_offer"}
            or customer_payment_posture == "partial_now"
            or (hardship_active and customer_payment_posture == "cannot_pay")
        )

    @staticmethod
    def _has_valid_discount_handoff(
        *,
        handoff_payload: dict[str, Any],
        customer_payment_posture: str,
        discount_stage: str,
    ) -> bool:
        if not handoff_payload:
            return False
        if not str(handoff_payload.get("case_id", "")).strip():
            return False
        payload_posture = str(handoff_payload.get("customer_payment_posture", "")).strip().lower()
        payload_stage = str(handoff_payload.get("discount_stage", "")).strip().lower()
        return payload_posture == customer_payment_posture and payload_stage in {
            discount_stage,
            "requested",
            "counter_offer",
            "planning",
        }

    @staticmethod
    def _first_non_empty_text(values: list[Any]) -> str:
        for value in values:
            text = str(value or "").strip()
            if text:
                return text
        return ""
