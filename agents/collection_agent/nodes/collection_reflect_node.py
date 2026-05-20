"""Reflect node variant for collection demo loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.nodes.reflect_node import ReflectNode
from src.nodes.types import AgentState, NodeUpdate


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
        )
        feedback = normalized["feedback"]
        complete = normalized["complete"]
        result["reflection_feedback"] = feedback
        result["reflection_complete"] = complete
        result["failure_type"] = normalized["failure_type"]
        result["correction_hints"] = normalized["correction_hints"]
        result["retry_target"] = "none" if complete else "plan_proposal"
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
                "Reflection retry limit reached; forcing completion to avoid loop. "
                "Use latest plan proposal and continue."
            )
            feedback["is_complete"] = True
            result["reflection_feedback"] = feedback
            result["reflection_complete"] = True
            result["reflection_retry_count"] = 0
            result["reflection_plan_retry_count"] = 0
            result["retry_target"] = "none"
            result["failure_type"] = "retry_limit_forced_complete"
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
        return {
            "kind": "plan_proposal_review",
            "plan_proposal": plan_proposal,
            "response_target": state.get("response_target"),
            "plan_node_context": plan_node_ctx,
            "verification_context": verification_ctx,
            "previous_response": state.get("response"),
        }

    @staticmethod
    def _normalize_reflection_result(
        *,
        feedback: dict[str, Any],
        complete: bool,
        state: AgentState,
    ) -> dict[str, Any]:
        reason = str(feedback.get("reason", "")).strip()
        lowered_reason = reason.lower()
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
        missing_fields = verification_ctx.get("verification_missing_fields") if isinstance(verification_ctx.get("verification_missing_fields"), list) else []
        identity_verified = bool(verification_ctx.get("identity_verified", False))
        next_actions = [str(x).strip().lower() for x in (plan_payload.get("next_actions") or []) if str(x).strip()]
        selected_next = ""
        if isinstance(plan_payload.get("plan_tree_update"), dict):
            selected_next = str(plan_payload.get("plan_tree_update", {}).get("selected_next_node_id", "")).strip().lower()

        asks_verification = (
            "verify_identity" in next_actions
            or "verification" in selected_next
            or "verify_identity" == selected_next
            or "missing verification" in lowered_reason
            or "missing verification fields" in lowered_reason
            or "request only missing verification fields" in lowered_reason
        )

        if (not identity_verified) and bool(missing_fields) and asks_verification and not complete:
            complete = True
            reason = "Plan correctly requests only missing verification fields for current stage."
            failure_type = "none"
        elif identity_verified and (not complete) and "verification is in progress" in lowered_reason:
            complete = True
            reason = "Verification is already complete in state; no plan retry needed."
            failure_type = "none"
        elif (
            not complete
            and ("clarify your request" in lowered_reason or "user input is a greeting" in lowered_reason)
            and asks_verification
        ):
            # Reflection validates plan quality, not user verbosity.
            complete = True
            reason = "Plan is stage-correct; short greeting input does not require plan retry."
            failure_type = "none"
        elif (
            not complete
            and observed_tool in {"verify_dob", "verify_mobile"}
            and observed_status in {"verified", "failed", "locked"}
            and (
                "verification evidence" in lowered_reason
                or "required fields" in lowered_reason
                or "plan is incomplete" in lowered_reason
            )
        ):
            # For iterative verification tools, observation itself is valid progression input.
            # The plan proposal directive stage should decide next step using updated verification state.
            complete = True
            reason = "Verification tool observation is valid; continue with updated verification state."
            failure_type = "none"

        if not complete:
            if "rate limit" in lowered_reason or "429" in lowered_reason:
                failure_type = "provider_rate_limit"
                correction_hints.append("Regenerate with shorter prompt context and keep same stage intent.")
            elif (not identity_verified) and bool(missing_fields) and not asks_verification:
                failure_type = "verification_path_mismatch"
                correction_hints.append("Keep plan at verification stage and request only missing verification fields.")
                correction_hints.append("Do not advance to dues/payment before identity_verified=true.")
            else:
                failure_type = "plan_correction_needed"
                correction_hints.append("Adjust proposal to current stage and keep valid next-node transition.")
        else:
            failure_type = "none"

        return {
            "feedback": {"reason": reason, "is_complete": bool(complete)},
            "complete": bool(complete),
            "failure_type": failure_type,
            "correction_hints": correction_hints,
        }

    def route(self, state: dict[str, Any]) -> str:
        if state.get("reflection_complete", self.default_is_complete):
            return self.complete_route
        return "retry_plan_proposal"
