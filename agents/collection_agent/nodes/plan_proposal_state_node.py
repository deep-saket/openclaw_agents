"""Plan proposal state preparation node for collections planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents.collection_agent.llm_structured import StructuredOutputRunner
from agents.collection_agent.nodes.plan_proposal_models import PlanSignalPayload
from agents.collection_agent.nodes.plan_proposal_utils import (
    compact_existing_plan_for_prompt,
    compact_memory_state_for_prompt,
    effective_mode,
    fresh_debug_state,
    get_existing_conversation_plan,
    is_plan_rejection,
    is_plan_request,
    json_compact,
    latest_observation,
    needs_discount_specialist,
    overlay_negotiation_state_from_graph,
    overlay_verification_state_from_graph,
    render_prompt_template,
)
from src.nodes.base import BaseGraphNode
from src.nodes.types import AgentState, NodeUpdate


@dataclass(slots=True)
class PlanProposalStateNode(BaseGraphNode):
    """Prepares planning state and classifies planning signals."""

    llm: Any | None = None
    system_prompt: str = ""
    user_prompt: str = ""
    classifier_system_prompt: str = ""
    classifier_user_prompt: str = ""
    strict_llm_mode: bool = True
    max_json_chars: int = 900
    last_debug: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    last_state_debug: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def execute(self, state: AgentState) -> NodeUpdate:
        self._record_llm_usage(state, node_name="plan_proposal_state")
        self.last_debug = fresh_debug_state()
        self.last_state_debug = dict(self.last_debug)

        memory = state.get("memory")
        memory_state = dict(getattr(memory, "state", {})) if memory is not None else {}
        routing_context = state.get("routing_context") if isinstance(state.get("routing_context"), dict) else {}
        plan_origin = str(routing_context.get("plan_origin", "react")).strip() or "react"

        observation = latest_observation(state)
        latest_obs = dict(observation) if isinstance(observation, dict) else None
        observed_tool = ""
        observed_tool_output: dict[str, Any] = {}
        if isinstance(observation, dict):
            if isinstance(observation.get("tool_phase"), dict):
                phase = observation.get("tool_phase", {})
                observed_tool = str(phase.get("tool_name", "") or "").strip()
                observed_tool_output = phase.get("output", {}) if isinstance(phase.get("output"), dict) else {}
            else:
                observed_tool = str(observation.get("tool_name", "") or "").strip()
                observed_tool_output = observation.get("output", {}) if isinstance(observation.get("output"), dict) else {}

        existing_plan = get_existing_conversation_plan(state=state, memory_state=memory_state)
        prepared_memory_state = overlay_verification_state_from_graph(state=state, memory_state=memory_state)
        prepared_memory_state = overlay_negotiation_state_from_graph(state=state, memory_state=prepared_memory_state)
        plan_mode = effective_mode(
            memory_state=prepared_memory_state,
            default=str(memory_state.get("mode", "strict_collections")),
        )

        plan_signals = self._classify_plan_signals(
            user_input=str(state.get("user_input", "")),
            mode=plan_mode,
            memory_state=prepared_memory_state,
            existing_plan=existing_plan,
        )
        suggested_mode = str(plan_signals.get("suggested_plan_mode", plan_mode)).strip().lower()
        if suggested_mode in {"strict_collections", "hardship_negotiation"} and suggested_mode != plan_mode:
            plan_mode = suggested_mode
            if memory is not None:
                memory.set_state(mode=plan_mode)

        self.last_state_debug = dict(self.last_debug)
        return {
            "route": "continue",
            "plan_prepared_memory_state": prepared_memory_state,
            "plan_signals": plan_signals,
            "plan_mode": plan_mode,
            "plan_origin": plan_origin,
            "effective_identity_verified": bool(prepared_memory_state.get("identity_verified", False)),
            "latest_observation": latest_obs,
            "observed_tool": observed_tool,
            "observed_tool_output": observed_tool_output,
            "existing_conversation_plan": existing_plan,
            "plan_state_prompt": self.last_debug.get("prompt"),
            "plan_state_system_prompt": self.last_debug.get("system_prompt"),
            "plan_state_llm_response": self.last_debug.get("llm_response"),
            "plan_state_llm_error": self.last_debug.get("llm_error"),
            "conversation_mode": str(prepared_memory_state.get("conversation_mode", "collections")).strip(),
            "negotiation_stage": str(prepared_memory_state.get("negotiation_stage", "none")).strip(),
            "customer_payment_posture": str(prepared_memory_state.get("customer_payment_posture", "unknown")).strip(),
            "hardship_context": (
                dict(prepared_memory_state.get("hardship_context", {}))
                if isinstance(prepared_memory_state.get("hardship_context"), dict)
                else {}
            ),
            "response_mode": str(prepared_memory_state.get("response_mode", "informational")).strip(),
            "active_dialogue_owner": str(prepared_memory_state.get("active_dialogue_owner", "collections")).strip(),
        }

    def route(self, state: AgentState) -> str:
        return str(state.get("route", "continue")).strip().lower() or "continue"

    def _classify_plan_signals(
        self,
        *,
        user_input: str,
        mode: str,
        memory_state: dict[str, Any],
        existing_plan: dict[str, Any],
    ) -> dict[str, Any]:
        llm_payload = self._classify_plan_signals_with_llm(
            user_input=user_input,
            mode=mode,
            memory_state=memory_state,
            existing_plan=existing_plan,
        )
        hardship_context = memory_state.get("hardship_context") if isinstance(memory_state.get("hardship_context"), dict) else {}
        if llm_payload is not None:
            if bool(hardship_context.get("hardship_detected", False)):
                llm_payload["hardship_signal"] = True
                llm_payload["hardship_reason"] = str(
                    hardship_context.get("hardship_reason") or llm_payload.get("hardship_reason") or "financial_hardship"
                )
                llm_payload["suggested_plan_mode"] = "hardship_negotiation"
            llm_payload.setdefault(
                "customer_payment_posture",
                str(memory_state.get("customer_payment_posture", "unknown")).strip().lower() or "unknown",
            )
            llm_payload.setdefault("customer_payment_capacity", memory_state.get("customer_payment_capacity"))
            llm_payload.setdefault("customer_payment_capacity_pct", memory_state.get("customer_payment_capacity_pct"))
            llm_payload.setdefault(
                "discount_stage",
                str(memory_state.get("discount_stage", "none")).strip().lower() or "none",
            )
            llm_payload.setdefault("discount_requested", bool(memory_state.get("discount_requested", False)))
            llm_payload.setdefault("counter_offer_present", bool(memory_state.get("counter_offer_present", False)))
            return llm_payload
        if self.strict_llm_mode:
            return {
                "needs_discount_specialist": False,
                "is_plan_request": False,
                "is_plan_rejection": False,
                "hardship_signal": bool(hardship_context.get("hardship_detected", False)),
                "hardship_reason": str(
                    hardship_context.get("hardship_reason") or memory_state.get("hardship_reason", "income_reduction")
                ),
                "suggested_plan_mode": "hardship_negotiation" if bool(hardship_context.get("hardship_detected", False)) else mode,
                "reason": "strict_mode_no_classifier_fallback",
            }
        result = {
            "needs_discount_specialist": needs_discount_specialist(user_input),
            "is_plan_request": is_plan_request(user_input),
            "is_plan_rejection": is_plan_rejection(user_input),
            "hardship_signal": any(token in user_input.lower() for token in ["cannot pay", "hardship", "vulnerability", "emi"]),
            "hardship_reason": str(memory_state.get("hardship_reason", "income_reduction")),
            "suggested_plan_mode": mode,
            "customer_payment_posture": str(memory_state.get("customer_payment_posture", "unknown")).strip().lower() or "unknown",
            "customer_payment_capacity": memory_state.get("customer_payment_capacity"),
            "customer_payment_capacity_pct": memory_state.get("customer_payment_capacity_pct"),
            "discount_stage": str(memory_state.get("discount_stage", "none")).strip().lower() or "none",
            "discount_requested": bool(memory_state.get("discount_requested", False)),
            "counter_offer_present": bool(memory_state.get("counter_offer_present", False)),
            "reason": "heuristic_fallback",
        }
        if bool(hardship_context.get("hardship_detected", False)):
            result["hardship_signal"] = True
            result["hardship_reason"] = str(
                hardship_context.get("hardship_reason") or result.get("hardship_reason") or "financial_hardship"
            )
            result["suggested_plan_mode"] = "hardship_negotiation"
        return result

    def _classify_plan_signals_with_llm(
        self,
        *,
        user_input: str,
        mode: str,
        memory_state: dict[str, Any],
        existing_plan: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self.llm is None:
            return None
        if not self.classifier_system_prompt.strip():
            raise ValueError("Missing required prompt: plan_proposal_state.classifier_system_prompt")
        if not self.classifier_user_prompt.strip():
            raise ValueError("Missing required prompt: plan_proposal_state.classifier_user_prompt")

        vars_map = {
            "user_input": user_input,
            "mode": mode,
            "memory_state_json": json_compact(
                compact_memory_state_for_prompt(memory_state),
                max_chars=self.max_json_chars,
            ),
            "existing_plan_json": json_compact(
                compact_existing_plan_for_prompt(existing_plan),
                max_chars=self.max_json_chars,
            ),
        }
        system_prompt = render_prompt_template(self.classifier_system_prompt, vars_map)
        user_prompt = render_prompt_template(self.classifier_user_prompt, vars_map)
        if not self.last_debug.get("prompt"):
            self.last_debug["prompt"] = user_prompt
            self.last_debug["system_prompt"] = system_prompt or None
        try:
            payload = StructuredOutputRunner(self.llm, max_retries=2).run(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=PlanSignalPayload,
            )
            prior = self.last_debug.get("llm_response")
            merged = dict(prior) if isinstance(prior, dict) else {}
            merged["classifier"] = payload.model_dump(mode="json")
            self.last_debug["llm_response"] = merged
        except Exception as exc:
            self.last_debug["llm_error"] = str(exc)
            return None
        normalized_mode = str(payload.suggested_plan_mode).strip().lower()
        if normalized_mode not in {"strict_collections", "hardship_negotiation"}:
            normalized_mode = mode
        return {
            "needs_discount_specialist": bool(payload.needs_discount_specialist),
            "is_plan_request": bool(payload.is_plan_request),
            "is_plan_rejection": bool(payload.is_plan_rejection),
            "hardship_signal": bool(payload.hardship_signal),
            "hardship_reason": str(payload.hardship_reason or memory_state.get("hardship_reason", "income_reduction")),
            "suggested_plan_mode": normalized_mode,
            "customer_payment_posture": str(
                payload.customer_payment_posture or memory_state.get("customer_payment_posture", "unknown")
            ).strip().lower()
            or "unknown",
            "customer_payment_capacity": payload.customer_payment_capacity
            if payload.customer_payment_capacity is not None
            else memory_state.get("customer_payment_capacity"),
            "customer_payment_capacity_pct": payload.customer_payment_capacity_pct
            if payload.customer_payment_capacity_pct is not None
            else memory_state.get("customer_payment_capacity_pct"),
            "discount_stage": str(payload.discount_stage or memory_state.get("discount_stage", "none")).strip().lower()
            or "none",
            "discount_requested": bool(
                payload.discount_requested
                if payload.discount_requested is not None
                else memory_state.get("discount_requested", False)
            ),
            "counter_offer_present": bool(
                payload.counter_offer_present
                if payload.counter_offer_present is not None
                else memory_state.get("counter_offer_present", False)
            ),
            "reason": str(payload.reason or ""),
        }
