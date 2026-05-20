"""Plan proposal directive node for collections planning."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from agents.collection_agent.llm_structured import StructuredOutputRunner
from agents.collection_agent.nodes.plan_proposal_models import PlanProposalPayload
from agents.collection_agent.nodes.plan_proposal_utils import (
    compact_existing_plan_for_prompt,
    effective_mode,
    extract_amount,
    fresh_debug_state,
    get_existing_conversation_plan,
    is_provider_rate_limit_error,
    json_compact,
    node_label,
    overlay_negotiation_state_from_graph,
    overlay_verification_state_from_graph,
    render_prompt_template,
    truncate_text,
)
from src.nodes.base import BaseGraphNode
from src.nodes.types import AgentState, NodeUpdate


@dataclass(slots=True)
class PlanProposalDirectiveNode(BaseGraphNode):
    """Builds the final plan proposal and response directive."""

    llm: Any | None = None
    system_prompt: str = ""
    user_prompt: str = ""
    classifier_system_prompt: str = ""
    classifier_user_prompt: str = ""
    strict_llm_mode: bool = True
    max_json_chars: int = 900
    last_debug: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def execute(self, state: AgentState) -> NodeUpdate:
        self._record_llm_usage(state, node_name="plan_proposal_directive")
        self.last_debug = fresh_debug_state()
        memory = state.get("memory")
        memory_state = (
            dict(state.get("plan_prepared_memory_state"))
            if isinstance(state.get("plan_prepared_memory_state"), dict)
            else dict(getattr(memory, "state", {})) if memory is not None else {}
        )
        if not isinstance(state.get("plan_prepared_memory_state"), dict):
            memory_state = overlay_verification_state_from_graph(state=state, memory_state=memory_state)
            memory_state = overlay_negotiation_state_from_graph(state=state, memory_state=memory_state)
        mode = str(
            state.get(
                "plan_mode",
                effective_mode(memory_state=memory_state, default=str(memory_state.get("mode", "strict_collections"))),
            )
        ).strip() or "strict_collections"
        routing_context = state.get("routing_context") if isinstance(state.get("routing_context"), dict) else {}
        plan_origin = str(state.get("plan_origin", routing_context.get("plan_origin", "react"))).strip() or "react"
        observation = self._latest_observation(state)
        if isinstance(observation, dict) and isinstance(observation.get("tool_phase"), dict):
            observation = observation.get("tool_phase")
        user_input = str(state.get("user_input", ""))
        observed_tool = str(observation.get("tool_name", "")) if isinstance(observation, dict) else str(state.get("observed_tool", ""))
        output = observation.get("output", {}) if isinstance(observation, dict) else (state.get("observed_tool_output") if isinstance(state.get("observed_tool_output"), dict) else {})
        decision = state.get("decision")
        existing_plan = (
            state.get("conversation_plan")
            if isinstance(state.get("conversation_plan"), dict)
            else get_existing_conversation_plan(state=state, memory_state=memory_state)
        )
        plan_signals = state.get("plan_signals") if isinstance(state.get("plan_signals"), dict) else {}
        identity_verified = bool(memory_state.get("identity_verified", False))

        def with_debug(update: NodeUpdate) -> NodeUpdate:
            update.setdefault("prompt", self.last_debug.get("prompt"))
            update.setdefault("system_prompt", self.last_debug.get("system_prompt"))
            update.setdefault("llm_response", self.last_debug.get("llm_response"))
            update.setdefault("llm_error", self.last_debug.get("llm_error"))
            return update

        def with_plan(update: NodeUpdate) -> NodeUpdate:
            response_target = str(update.get("response_target", "customer")).strip().lower() or "customer"
            route = str(update.get("route", "continue")).strip().lower() or "continue"
            proposal = update.get("plan_proposal") if isinstance(update.get("plan_proposal"), dict) else {}
            plan = dict(existing_plan) if isinstance(existing_plan, dict) else {}
            update["conversation_plan"] = plan
            if proposal:
                proposal = self._align_customer_proposal_with_plan(
                    proposal=proposal,
                    plan=plan,
                    memory_state=memory_state,
                    plan_signals=plan_signals,
                )
                proposal = self._attach_response_directive(
                    proposal=proposal,
                    state=state,
                    memory_state=memory_state,
                    plan=plan,
                    plan_signals=plan_signals,
                    route=route,
                    response_target=response_target,
                )
                current_id = str(plan.get("current_node_id", ""))
                proposal["conversation_plan_id"] = plan.get("plan_id")
                proposal["conversation_plan_version"] = plan.get("version")
                proposal["conversation_plan_current_node"] = current_id
                proposal["conversation_plan_current_label"] = node_label(plan, current_id)
                proposal["conversation_plan"] = plan
                update["plan_proposal"] = proposal
            if memory is not None and isinstance(plan, dict):
                memory.set_state(active_conversation_plan=plan)
                memory_state_local = dict(getattr(memory, 'state', {}))
                node_llms = dict(memory_state_local.get('node_llms', {}))
                if 'plan_proposal' in node_llms:
                    node_llms['plan_proposal_directive'] = node_llms.pop('plan_proposal')
                    memory.set_state(node_llms=node_llms)
            return with_debug(update)

        if bool(memory_state.get("agent_loop_blocked", False)):
            if memory is not None:
                memory.set_state(agent_loop_blocked=False)
            return with_plan({
                "route": "continue",
                "response_target": "customer",
                "plan_proposal": {
                    "target": "customer",
                    "intent": "loop_guard",
                    "guidance": "Internal planning loop exceeded threshold.",
                    "next_actions": ["pay_now", "plan_revision", "schedule_followup"],
                    "plan_origin": "loop_guard",
                },
            })

        if self._is_conversation_termination(user_input):
            return with_plan({
                "route": "continue",
                "response_target": "customer",
                "plan_proposal": {
                    "target": "customer",
                    "intent": "conversation_termination",
                    "guidance": "Conversation is being closed politely.",
                    "plan_origin": "conversation_termination",
                    "plan_tree_update": {
                        "operation": "complete",
                        "status": "completed",
                        "selected_next_node_id": "resolve_outcome",
                    },
                },
                "additional_targets": ["collection_memory_helper_agent"],
                "memory_helper_trigger": {
                    "reason": "conversation_termination",
                    "final_user_message": user_input,
                },
            })

        discount_recommendation = memory_state.get("discount_recommendation")
        if isinstance(discount_recommendation, dict) and discount_recommendation:
            if memory is not None:
                memory.set_state(discount_recommendation=None)
            return with_plan({
                "route": "continue",
                "response_target": "customer",
                "plan_proposal": {
                    "target": "customer",
                    "intent": "discount_recommendation",
                    "discount_recommendation": discount_recommendation,
                    "plan_origin": "discount_recommendation",
                    "plan_tree_update": {
                        "operation": "advance",
                        "selected_next_node_id": "evaluate_assistance",
                    },
                },
            })

        revision_index = int(memory_state.get("plan_revision_index", 0))
        hardship_reason = str(plan_signals.get("hardship_reason") or memory_state.get("hardship_reason", "income_reduction"))
        case_id = str(memory_state.get("active_case_id", "COLL-1001"))

        if bool(plan_signals.get("needs_discount_specialist")) and case_id:
            if not identity_verified:
                return with_plan({
                    "route": "continue",
                    "response_target": "customer",
                    "plan_proposal": {
                        "target": "customer",
                        "intent": "verification_required_before_discount",
                        "plan_outline": "Complete identity verification before evaluating discount/restructure options.",
                        "next_actions": ["complete_identity_verification", "then_evaluate_assistance"],
                    },
                })
            if memory is not None and hardship_reason:
                memory.set_state(hardship_reason=hardship_reason)
            return with_plan({
                "route": "continue",
                "response": "Trigger discount planning specialist for hardship assistance recommendation.",
                "response_target": "discount_planning_agent",
                "handoff_payload": {
                    "case_id": case_id,
                    "customer_id": str(memory_state.get("active_user_id", "")).strip(),
                    "hardship_reason": hardship_reason,
                    "user_message": user_input,
                    "requested_by": "collection_agent",
                },
                "plan_proposal": {
                    "target": "discount_planning_agent",
                    "intent": "discount_specialist_handoff",
                    "plan_outline": "Escalate to discount planning specialist and return with recommendation.",
                    "next_actions": ["run_discount_specialist", "apply_recommendation", "respond_to_customer"],
                },
            })

        if mode != "hardship_negotiation":
            plan_proposal = self._build_plan_proposal(
                state=state,
                user_input=user_input,
                memory_state=memory_state,
                observation=(observation if isinstance(observation, dict) else None),
                decision=decision,
                default_plan=self._build_generic_plan_outline(
                    user_input=user_input,
                    memory_state=memory_state,
                    plan_signals=plan_signals,
                ),
                plan_origin=plan_origin,
                mode=mode,
                existing_plan=existing_plan,
            )
            return with_plan({
                "route": "continue",
                "response_target": str(plan_proposal.get("target", "customer")),
                "plan_proposal": plan_proposal,
            })

        if observed_tool == "plan_propose":
            if memory is not None and isinstance(output, dict):
                memory.set_state(current_plan=output, plan_revision_index=int(memory_state.get("plan_revision_index", 0)))
            plan_proposal = self._build_plan_proposal(
                state=state,
                user_input=user_input,
                memory_state=memory_state,
                observation=(observation if isinstance(observation, dict) else None),
                decision=decision,
                default_plan=self._build_generic_plan_outline(
                    user_input=user_input,
                    memory_state=memory_state,
                    plan_signals=plan_signals,
                ),
                plan_origin=plan_origin,
                mode=mode,
                existing_plan=existing_plan,
            )
            return with_plan({
                "route": "continue",
                "response_target": "customer",
                "plan_proposal": plan_proposal,
            })

        should_propose = False
        if observed_tool == "offer_eligibility":
            should_propose = True
        if observed_tool == "channel_switch" and not memory_state.get("current_plan"):
            should_propose = True
        if bool(plan_signals.get("is_plan_rejection")) and memory_state.get("current_plan"):
            should_propose = True
            revision_index += 1
        if bool(plan_signals.get("is_plan_request")) and case_id:
            should_propose = True

        if not should_propose:
            plan_proposal = self._build_plan_proposal(
                state=state,
                user_input=user_input,
                memory_state=memory_state,
                observation=(observation if isinstance(observation, dict) else None),
                decision=decision,
                default_plan=self._build_generic_plan_outline(
                    user_input=user_input,
                    memory_state=memory_state,
                    plan_signals=plan_signals,
                ),
                plan_origin=plan_origin,
                mode=mode,
                existing_plan=existing_plan,
            )
            return with_plan({
                "route": "continue",
                "response_target": str(plan_proposal.get("target", "customer")),
                "plan_proposal": plan_proposal,
            })

        max_installment = extract_amount(user_input)
        arguments: dict[str, Any] = {
            "case_id": case_id,
            "hardship_reason": hardship_reason,
            "revision_index": revision_index,
        }
        if max_installment is not None:
            arguments["max_installment_amount"] = max_installment

        if memory is not None:
            memory.set_state(plan_revision_index=revision_index)

        decision = SimpleNamespace(
            thought="Routing to plan proposal tool based on hardship negotiation state.",
            tool_call=SimpleNamespace(tool_name="plan_propose", arguments=arguments),
            respond_directly=False,
            response_text=None,
            done=False,
        )
        return with_plan({
            "route": "continue",
            "decision": decision,
            "response_target": "self",
            "plan_proposal": {
                "target": "self",
                "intent": "tool_plan_proposal",
                "plan_outline": "Need hardship-plan computation before responding to customer.",
                "next_actions": ["run_plan_propose_tool", "review_offer", "respond_to_customer"],
                "plan_tree_update": {
                    "operation": "branch",
                    "selected_next_node_id": "evaluate_assistance",
                },
            },
        })

    def route(self, state: AgentState) -> str:
        return str(state.get("route", "continue")).strip().lower() or "continue"

    @staticmethod
    def _latest_observation(state: AgentState) -> dict[str, Any] | None:
        observations = state.get("observations")
        if isinstance(observations, list):
            for item in reversed(observations):
                if isinstance(item, dict):
                    return item
        observation = state.get("observation")
        return dict(observation) if isinstance(observation, dict) else None

    def _build_generic_plan_outline(
        self,
        *,
        user_input: str,
        memory_state: dict[str, Any],
        plan_signals: dict[str, Any] | None = None,
    ) -> str:
        case_id = str(memory_state.get("active_case_id", "COLL-1001"))
        signals = plan_signals or {}
        hardship_signal = bool(signals.get("hardship_signal", False))
        hardship_context = memory_state.get("hardship_context") if isinstance(memory_state.get("hardship_context"), dict) else {}
        if bool(hardship_context.get("hardship_detected", False)):
            hardship_signal = True
        lowered = user_input.lower()
        if any(token in lowered for token in ["pay now", "payment link", "link", "proceed with payment"]):
            return (
                f"Plan for {case_id}: confirm customer identity and dues context, complete immediate payment flow, "
                "and confirm closure after payment acknowledgment."
            )
        if hardship_signal or any(token in lowered for token in ["cannot pay", "hardship", "discount", "settlement", "waiver", "emi"]):
            return (
                f"Plan for {case_id}: validate hardship constraints, determine eligible assistance options, "
                "propose revised repayment path, and capture next commitment with follow-up."
            )
        return (
            f"Plan for {case_id}: verify account context, provide concise dues explanation, "
            "collect payment intent, and capture commitment or follow-up details."
        )

    def _build_plan_proposal(
        self,
        *,
        state: AgentState,
        user_input: str,
        memory_state: dict[str, Any],
        observation: dict[str, Any] | None,
        decision: Any | None,
        default_plan: str,
        plan_origin: str,
        mode: str,
        existing_plan: dict[str, Any],
    ) -> dict[str, Any]:
        llm_proposal = self._build_plan_proposal_with_llm(
            state=state,
            user_input=user_input,
            memory_state=memory_state,
            observation=observation,
            decision=decision,
            default_plan=default_plan,
            plan_origin=plan_origin,
            mode=mode,
            existing_plan=existing_plan,
        )
        if llm_proposal is not None:
            return self._validate_and_repair_proposal(
                proposal=llm_proposal,
                state=state,
                memory_state=memory_state,
            )
        llm_error = str(self.last_debug.get("llm_error", "")).strip()
        if self.strict_llm_mode and not is_provider_rate_limit_error(llm_error):
            raise RuntimeError(
                "PlanProposalDirectiveNode failed to produce LLM structured output while strict_llm_mode is enabled. "
                f"Underlying error: {llm_error or 'unknown'}"
            )

        decision_text = str(getattr(decision, "response_text", "") or "").strip()
        decision_target = str(getattr(decision, "response_target", "") or "").strip().lower()
        target = decision_target if decision_target in {"customer", "self"} else "customer"
        observed_tool = str(observation.get("tool_name", "")) if isinstance(observation, dict) else ""
        output = observation.get("output", {}) if isinstance(observation, dict) else {}

        plan_outline = default_plan
        if decision_text:
            if decision_text.lower().startswith("proposed plan for ") or decision_text.lower().startswith("plan for "):
                plan_outline = decision_text
            elif decision_text.startswith("Executed "):
                plan_outline = f"Tool execution result observed: {decision_text}"
            else:
                plan_outline = f"Direct response path selected: {decision_text}"
        elif observed_tool:
            plan_outline = (
                f"Observation-driven plan: interpret `{observed_tool}` output, provide the next customer response, "
                "and request one concrete next action."
            )

        fallback = {
            "target": target,
            "intent": "generic_plan",
            "plan_outline": plan_outline,
            "draft_response": decision_text if decision_text and not decision_text.startswith("Executed ") else "",
            "plan_origin": plan_origin or "default_direct_plan",
            "mode": mode,
            "context": {
                "case_id": str(memory_state.get("active_case_id", "COLL-1001")),
                "customer_name": str(memory_state.get("active_customer_name", "Customer")).strip() or "Customer",
                "overdue_amount": float(memory_state.get("active_overdue_amount", 0.0) or 0.0),
                "observed_tool": observed_tool,
                "observed_tool_output": output if isinstance(output, dict) else {},
                "conversation_mode": str(memory_state.get("conversation_mode", "collections")).strip(),
                "negotiation_stage": str(memory_state.get("negotiation_stage", "none")).strip(),
                "customer_payment_posture": str(memory_state.get("customer_payment_posture", "unknown")).strip(),
                "hardship_context": (
                    dict(memory_state.get("hardship_context", {}))
                    if isinstance(memory_state.get("hardship_context"), dict)
                    else {}
                ),
                "response_mode": str(memory_state.get("response_mode", "informational")).strip(),
                "active_dialogue_owner": str(memory_state.get("active_dialogue_owner", "collections")).strip(),
            },
            "next_actions": self._derive_next_actions(user_input=user_input, mode=mode, observed_tool=observed_tool),
        }
        return self._validate_and_repair_proposal(
            proposal=fallback,
            state=state,
            memory_state=memory_state,
        )

    def _build_plan_proposal_with_llm(
        self,
        *,
        state: AgentState,
        user_input: str,
        memory_state: dict[str, Any],
        observation: dict[str, Any] | None,
        decision: Any | None,
        default_plan: str,
        plan_origin: str,
        mode: str,
        existing_plan: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self.llm is None:
            return None
        if not self.system_prompt.strip():
            raise ValueError("Missing required prompt: plan_proposal.system_prompt")
        if not self.user_prompt.strip():
            raise ValueError("Missing required prompt: plan_proposal.user_prompt")

        decision_payload = {
            "response_text": str(getattr(decision, "response_text", "") or "").strip(),
            "respond_directly": bool(getattr(decision, "respond_directly", False)),
            "tool_call": {
                "tool_name": str(getattr(getattr(decision, "tool_call", None), "tool_name", "") or "").strip(),
                "arguments": getattr(getattr(decision, "tool_call", None), "arguments", {}) or {},
            },
        }
        obs_tool = ""
        obs_output: dict[str, Any] = {}
        reflection_feedback: dict[str, Any] = {}
        if isinstance(observation, dict):
            if isinstance(observation.get("tool_phase"), dict):
                tool_phase = observation.get("tool_phase", {})
                obs_tool = str(tool_phase.get("tool_name", "") or "").strip()
                obs_output = tool_phase.get("output", {}) if isinstance(tool_phase.get("output"), dict) else {}
            else:
                obs_tool = str(observation.get("tool_name", "") or "").strip()
                obs_output = observation.get("output", {}) if isinstance(observation.get("output"), dict) else {}
            if isinstance(observation.get("reflection_feedback"), dict):
                reflection_feedback = dict(observation.get("reflection_feedback", {}))

        customer_context_json = json.dumps(
            {
                "case_id": str(memory_state.get("active_case_id", "COLL-1001")),
                "customer_name": str(memory_state.get("active_customer_name", "Customer")).strip() or "Customer",
                "overdue_amount": float(memory_state.get("active_overdue_amount", 0.0) or 0.0),
            },
            ensure_ascii=True,
        )
        verification_context_json = json.dumps(
            {
                "identity_verified": bool(memory_state.get("identity_verified", False)),
                "required_fields": memory_state.get("active_verification_required_fields", []),
                "verification_missing_fields": memory_state.get("verification_missing_fields", []),
                "verification_verified_fields": memory_state.get("verification_verified_fields", []),
                "verification_entities": memory_state.get("verification_entities", {}),
            },
            ensure_ascii=True,
            default=str,
        )
        negotiation_context_json = json.dumps(
            {
                "conversation_mode": memory_state.get("conversation_mode", "collections"),
                "negotiation_stage": memory_state.get("negotiation_stage", "none"),
                "customer_payment_posture": memory_state.get("customer_payment_posture", "unknown"),
                "hardship_context": memory_state.get("hardship_context", {}),
                "response_mode": memory_state.get("response_mode", "informational"),
                "active_dialogue_owner": memory_state.get("active_dialogue_owner", "collections"),
            },
            ensure_ascii=True,
            default=str,
        )
        entities_context_json = json.dumps(
            {
                "extracted_entities": state.get("extracted_entities", {}),
                "extracted_entity_descriptions": state.get("extracted_entity_descriptions", {}),
                "verification_entities": state.get("verification_entities", {}),
                "verification_missing_fields": state.get("verification_missing_fields", []),
                "identity_verified": bool(memory_state.get("identity_verified", False)),
            },
            ensure_ascii=True,
            default=str,
        )
        template_vars = {
            "user_input": user_input,
            "plan_origin": plan_origin,
            "mode": mode,
            "default_plan": default_plan,
            "existing_plan_json": json_compact(
                compact_existing_plan_for_prompt(existing_plan),
                max_chars=self.max_json_chars,
            ),
            "decision_payload_json": json.dumps(decision_payload, ensure_ascii=True, default=str),
            "obs_tool": obs_tool,
            "obs_output_json": json_compact(obs_output, max_chars=900),
            "reflection_feedback_json": json_compact(reflection_feedback, max_chars=700),
            "customer_context_json": customer_context_json,
            "verification_context_json": verification_context_json,
            "negotiation_context_json": negotiation_context_json,
            "entities_context_json": entities_context_json,
        }
        system_prompt = render_prompt_template(self.system_prompt, template_vars)
        user_prompt = render_prompt_template(self.user_prompt, template_vars)
        self.last_debug["prompt"] = user_prompt
        self.last_debug["system_prompt"] = system_prompt or None
        self.last_debug["llm_error"] = None
        try:
            payload = StructuredOutputRunner(self.llm, max_retries=2).run(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=PlanProposalPayload,
            )
            prior = self.last_debug.get("llm_response")
            merged = dict(prior) if isinstance(prior, dict) else {}
            merged["proposal"] = payload.model_dump(mode="json", by_alias=True)
            self.last_debug["llm_response"] = merged
        except Exception as exc:
            self.last_debug["llm_error"] = str(exc)
            minimal_template_vars = {
                "user_input": truncate_text(user_input, 280),
                "plan_origin": plan_origin,
                "mode": mode,
                "default_plan": truncate_text(default_plan, 220),
                "existing_plan_json": json_compact(
                    compact_existing_plan_for_prompt(existing_plan, minimal=True),
                    max_chars=900,
                ),
                "decision_payload_json": json_compact(
                    {
                        "response_text": str(decision_payload.get("response_text", ""))[:180],
                        "respond_directly": bool(decision_payload.get("respond_directly", False)),
                        "tool_name": str(((decision_payload.get("tool_call") or {}).get("tool_name", ""))),
                    },
                    max_chars=500,
                ),
                "obs_tool": obs_tool,
                "obs_output_json": json_compact(
                    {
                        "status": (obs_output or {}).get("status") if isinstance(obs_output, dict) else None,
                        "needs_additional_action": bool((obs_output or {}).get("needs_additional_action", False))
                        if isinstance(obs_output, dict)
                        else False,
                        "keys": sorted([str(k) for k in obs_output.keys()])[:8] if isinstance(obs_output, dict) else [],
                    },
                    max_chars=350,
                ),
                "reflection_feedback_json": json_compact(reflection_feedback, max_chars=500),
                "customer_context_json": customer_context_json,
                "verification_context_json": verification_context_json,
                "negotiation_context_json": negotiation_context_json,
                "entities_context_json": json_compact(
                    {
                        "verification_entities": state.get("verification_entities", {}),
                        "verification_missing_fields": memory_state.get("verification_missing_fields", []),
                        "identity_verified": bool(memory_state.get("identity_verified", False)),
                    },
                    max_chars=500,
                ),
            }
            minimal_system_prompt = render_prompt_template(self.system_prompt, minimal_template_vars)
            minimal_user_prompt = render_prompt_template(self.user_prompt, minimal_template_vars)
            self.last_debug["prompt"] = minimal_user_prompt
            self.last_debug["system_prompt"] = minimal_system_prompt or None
            try:
                payload = StructuredOutputRunner(self.llm, max_retries=4).run(
                    system_prompt=minimal_system_prompt,
                    user_prompt=minimal_user_prompt,
                    schema=PlanProposalPayload,
                )
                prior = self.last_debug.get("llm_response")
                merged = dict(prior) if isinstance(prior, dict) else {}
                merged["proposal"] = payload.model_dump(mode="json", by_alias=True)
                merged["attempt"] = "minimal_recovery"
                self.last_debug["llm_response"] = merged
                self.last_debug["llm_error"] = None
            except Exception as exc2:
                self.last_debug["llm_error"] = f"primary={exc}; recovery={exc2}"
                return None

        proposal = payload.model_dump(mode="json", by_alias=True)
        target = str(proposal.get("target", "customer")).strip().lower()
        if target not in {"customer", "self"}:
            target = "customer"
        proposal["target"] = target
        proposal["plan_origin"] = plan_origin or "default_direct_plan"
        proposal["mode"] = mode
        proposal["context"] = {
            "case_id": str(memory_state.get("active_case_id", "COLL-1001")),
            "customer_name": str(memory_state.get("active_customer_name", "Customer")).strip() or "Customer",
            "overdue_amount": float(memory_state.get("active_overdue_amount", 0.0) or 0.0),
            "observed_tool": obs_tool,
            "observed_tool_output": obs_output if isinstance(obs_output, dict) else {},
            "conversation_mode": str(memory_state.get("conversation_mode", "collections")).strip(),
            "negotiation_stage": str(memory_state.get("negotiation_stage", "none")).strip(),
            "customer_payment_posture": str(memory_state.get("customer_payment_posture", "unknown")).strip(),
            "hardship_context": (
                dict(memory_state.get("hardship_context", {}))
                if isinstance(memory_state.get("hardship_context"), dict)
                else {}
            ),
            "response_mode": str(memory_state.get("response_mode", "informational")).strip(),
            "active_dialogue_owner": str(memory_state.get("active_dialogue_owner", "collections")).strip(),
        }
        if not isinstance(proposal.get("next_actions"), list) or not proposal.get("next_actions"):
            proposal["next_actions"] = self._derive_next_actions(
                user_input=user_input, mode=mode, observed_tool=obs_tool
            )
        return self._validate_and_repair_proposal(
            proposal=proposal,
            state=state,
            memory_state=memory_state,
        )

    def _validate_and_repair_proposal(
        self,
        *,
        proposal: dict[str, Any],
        state: AgentState,
        memory_state: dict[str, Any],
    ) -> dict[str, Any]:
        patched = dict(proposal) if isinstance(proposal, dict) else {}
        warnings: list[str] = []
        identity_verified = bool(memory_state.get("identity_verified", False))
        if not identity_verified:
            next_actions = patched.get("next_actions")
            next_actions_list = [str(x).strip() for x in next_actions if str(x).strip()] if isinstance(next_actions, list) else []
            normalized_actions: list[str] = []
            for action in next_actions_list:
                lowered = action.lower()
                if lowered in {"ask_for_verification", "request_verification", "collect_verification"}:
                    lowered = "verify_identity"
                normalized_actions.append(lowered)
            if "verify_identity" not in normalized_actions:
                normalized_actions.insert(0, "verify_identity")
                warnings.append("Inserted verify_identity into next_actions while identity_verified=false.")
            # Keep verification as the leading actionable step until completion.
            deduped_actions: list[str] = []
            for action in normalized_actions:
                if action not in deduped_actions:
                    deduped_actions.append(action)
            next_actions_list = ["verify_identity", *[a for a in deduped_actions if a != "verify_identity"]]
            patched["next_actions"] = next_actions_list

            tree_update = patched.get("plan_tree_update")
            if not isinstance(tree_update, dict):
                tree_update = {}
                patched["plan_tree_update"] = tree_update
            selected_next = str(tree_update.get("selected_next_node_id", "")).strip().lower()
            if selected_next not in {"verify_identity"}:
                tree_update["selected_next_node_id"] = "verify_identity"
                warnings.append("Reset selected_next_node_id to verify_identity until verification completes.")
            tree_update["current_node_id"] = "verify_identity"
            tree_update["operation"] = "advance"
            tree_update["new_nodes"] = []
            tree_update["remove_node_ids"] = []

            new_edges = tree_update.get("new_edges")
            edge_list = list(new_edges) if isinstance(new_edges, list) else []
            canonical_nodes: set[str] = set()
            convo_plan = state.get("conversation_plan")
            if isinstance(convo_plan, dict) and isinstance(convo_plan.get("nodes"), list):
                canonical_nodes = {
                    str(node.get("id", "")).strip().lower()
                    for node in convo_plan.get("nodes", [])
                    if isinstance(node, dict) and str(node.get("id", "")).strip()
                }
            normalized_edges: list[dict[str, Any]] = []
            for raw in edge_list:
                if not isinstance(raw, dict):
                    continue
                src = str(raw.get("from", "")).strip().lower()
                dst = str(raw.get("to", "")).strip().lower()
                if not src or not dst:
                    continue
                if canonical_nodes and (src not in canonical_nodes or dst not in canonical_nodes):
                    continue
                if src == dst:
                    continue
                normalized_edges.append(
                    {
                        "from": src,
                        "to": dst,
                        "condition": str(raw.get("condition", "")).strip(),
                    }
                )
            tree_update["new_edges"] = normalized_edges

            draft_response = str(patched.get("draft_response", "")).strip()
            if draft_response and re.search(r"\boverdue amount\b|\binr\b|\bemi\b", draft_response, re.IGNORECASE):
                warnings.append("Draft response contains dues before verification completion; response node will guard this.")
        elif identity_verified:
            next_actions = patched.get("next_actions")
            next_actions_list = [str(x).strip() for x in next_actions if str(x).strip()] if isinstance(next_actions, list) else []
            patched["next_actions"] = [x for x in next_actions_list if x.lower() != "verify_identity"]
            tree_update = patched.get("plan_tree_update")
            if not isinstance(tree_update, dict):
                tree_update = {}
                patched["plan_tree_update"] = tree_update
            selected_next = str(tree_update.get("selected_next_node_id", "")).strip().lower()
            current_node = str(tree_update.get("current_node_id", "")).strip().lower()
            if selected_next in {"", "verify_identity"}:
                tree_update["selected_next_node_id"] = "explain_dues"
            if current_node in {"", "verify_identity"}:
                tree_update["current_node_id"] = "explain_dues"
            mark_done = [str(x).strip() for x in tree_update.get("mark_done", []) if str(x).strip()]
            if "verify_identity" not in mark_done:
                mark_done.append("verify_identity")
            tree_update["mark_done"] = mark_done

        if warnings:
            patched["plan_validation_warnings"] = warnings
        return patched

    @staticmethod
    def _derive_next_actions(*, user_input: str, mode: str, observed_tool: str) -> list[str]:
        lowered = user_input.lower()
        actions: list[str] = ["verify_identity", "explain_dues", "collect_payment_intent"]
        if "pay now" in lowered or "payment link" in lowered:
            actions.append("complete_payment_flow")
        if mode == "hardship_negotiation":
            actions.append("evaluate_assistance_options")
        if observed_tool:
            actions.append("interpret_tool_observation")
        actions.append("resolve_outcome")
        return actions

    @staticmethod
    def _is_conversation_termination(text: str) -> bool:
        lowered = text.lower().strip()
        if not lowered:
            return False
        signals = [
            "bye",
            "goodbye",
            "thanks that's all",
            "thank you that's all",
            "close this",
            "done for now",
            "that's all",
            "end conversation",
            "you can close",
        ]
        return any(signal in lowered for signal in signals)

    @staticmethod
    def _align_customer_proposal_with_plan(
        *,
        proposal: dict[str, Any],
        plan: dict[str, Any],
        memory_state: dict[str, Any],
        plan_signals: dict[str, Any],
    ) -> dict[str, Any]:
        aligned = dict(proposal)
        target = str(aligned.get("target", "customer")).strip().lower() or "customer"
        if target != "customer":
            return aligned

        current_node_id = str(plan.get("current_node_id", "")).strip()
        identity_verified = bool(memory_state.get("identity_verified", False))
        intent = str(aligned.get("intent", "")).strip().lower()
        draft_text = str(aligned.get("draft_response", "")).strip().lower()
        outline_text = str(aligned.get("plan_outline", "")).strip().lower()
        verification_language = any(
            token in f"{draft_text} {outline_text}"
            for token in ["confirm your identity", "verify customer identity", "identity verification"]
        )
        hardship_signal = bool(plan_signals.get("hardship_signal", False))

        # Guardrail: while identity is incomplete, keep customer proposal pinned
        # to verification path and prevent unrelated plan branches.
        if (not identity_verified) and current_node_id in {"verify_identity", ""}:
            aligned["intent"] = "verify_identity"
            aligned["plan_outline"] = "Request only remaining verification fields to complete identity verification."
            aligned["next_actions"] = ["verify_identity"]
            tree = aligned.get("plan_tree_update") if isinstance(aligned.get("plan_tree_update"), dict) else {}
            tree["selected_next_node_id"] = "verify_identity"
            tree["current_node_id"] = "verify_identity"
            tree["new_edges"] = []
            aligned["plan_tree_update"] = tree

        if identity_verified and current_node_id == "explain_dues":
            if intent in {"verify_identity", "verification_required_before_discount"} or verification_language:
                aligned["intent"] = "case_snapshot"
                aligned["plan_outline"] = "Explain dues context and collect payment intent."
                aligned["draft_response"] = ""
                aligned["next_actions"] = ["collect_payment_intent"]
                aligned["case_snapshot"] = {
                    "customer_name": str(memory_state.get("active_customer_name", "Customer")).strip() or "Customer",
                    "case_id": str(memory_state.get("active_case_id", "COLL-1001")).strip() or "COLL-1001",
                    "overdue_amount": float(memory_state.get("active_overdue_amount", 0.0) or 0.0),
                    "emi_amount": float(memory_state.get("active_emi_amount", 0.0) or 0.0),
                    "late_fee": float(memory_state.get("active_late_fee", 0.0) or 0.0),
                    "dpd": int(memory_state.get("active_dpd", 0) or 0),
                }
            elif intent == "discount_recommendation" and not hardship_signal:
                aligned["intent"] = "case_snapshot"
                aligned["plan_outline"] = "Explain dues context and collect payment intent."
                aligned["draft_response"] = ""
                aligned["next_actions"] = ["collect_payment_intent"]
                aligned["case_snapshot"] = {
                    "customer_name": str(memory_state.get("active_customer_name", "Customer")).strip() or "Customer",
                    "case_id": str(memory_state.get("active_case_id", "COLL-1001")).strip() or "COLL-1001",
                    "overdue_amount": float(memory_state.get("active_overdue_amount", 0.0) or 0.0),
                    "emi_amount": float(memory_state.get("active_emi_amount", 0.0) or 0.0),
                    "late_fee": float(memory_state.get("active_late_fee", 0.0) or 0.0),
                    "dpd": int(memory_state.get("active_dpd", 0) or 0),
                }
        return aligned

    def _attach_response_directive(
        self,
        *,
        proposal: dict[str, Any],
        state: AgentState,
        memory_state: dict[str, Any],
        plan: dict[str, Any],
        plan_signals: dict[str, Any],
        route: str,
        response_target: str,
    ) -> dict[str, Any]:
        attached = dict(proposal)
        directive = self._build_response_directive(
            state=state,
            memory_state=memory_state,
            plan=plan,
            proposal=attached,
            plan_signals=plan_signals,
            route=route,
            response_target=response_target,
        )
        attached["response_directive"] = directive
        attached["conversation_objective"] = directive.get("conversation_objective")
        attached["dialogue_action"] = directive.get("dialogue_action")
        attached["response_mode"] = directive.get("response_mode")
        attached["required_response_elements"] = list(directive.get("required_response_elements", []))
        attached["forbidden_dialogue_actions"] = list(directive.get("forbidden_dialogue_actions", []))
        attached["allowed_dialogue_actions"] = list(directive.get("allowed_dialogue_actions", []))
        attached["customer_facing_goal"] = directive.get("customer_facing_goal")
        attached["handoff_target"] = directive.get("handoff_target")
        return attached

    def _build_response_directive(
        self,
        *,
        state: AgentState,
        memory_state: dict[str, Any],
        plan: dict[str, Any],
        proposal: dict[str, Any],
        plan_signals: dict[str, Any],
        route: str,
        response_target: str,
    ) -> dict[str, Any]:
        del state, plan, proposal, plan_signals, route
        objective, action, mode = self._infer_response_objective(memory_state=memory_state, response_target=response_target)
        if response_target == "discount_planning_agent":
            objective = "handoff_to_offer_agent"
            action = "handoff"
            mode = "firm" if mode == "informational" else mode
        directive = {
            "conversation_objective": objective,
            "dialogue_action": action,
            "response_mode": mode,
            "required_response_elements": self._required_response_elements_for_objective(objective=objective),
            "forbidden_dialogue_actions": self._forbidden_dialogue_actions_for_objective(objective=objective),
            "allowed_dialogue_actions": self._allowed_dialogue_actions_for_objective(objective=objective),
            "customer_facing_goal": self._customer_facing_goal_for_objective(
                objective=objective,
                memory_state=memory_state,
            ),
            "handoff_target": ("discount_planning_agent" if response_target == "discount_planning_agent" else None),
        }
        return directive

    @staticmethod
    def _infer_response_objective(
        *,
        memory_state: dict[str, Any],
        response_target: str,
    ) -> tuple[str, str, str]:
        if response_target == "discount_planning_agent":
            return "handoff_to_offer_agent", "handoff", "firm"

        identity_verified = bool(memory_state.get("identity_verified", False))
        conversation_mode = str(memory_state.get("conversation_mode", "collections")).strip().lower()
        negotiation_stage = str(memory_state.get("negotiation_stage", "none")).strip().lower()
        customer_payment_posture = str(memory_state.get("customer_payment_posture", "unknown")).strip().lower()
        hardship_context = (
            memory_state.get("hardship_context")
            if isinstance(memory_state.get("hardship_context"), dict)
            else {}
        )
        hardship_active = conversation_mode == "hardship_negotiation" or bool(hardship_context.get("hardship_detected", False))
        if not identity_verified:
            return "collect_verification", "ask_verification", "compliance"
        if hardship_active:
            if negotiation_stage in {"discovering_hardship", "assessing_capacity"}:
                return "assess_affordability", "ask_affordable_amount", "empathetic"
            if negotiation_stage in {"evaluating_options"}:
                return "present_arrangement_options", "present_offer", "negotiation"
            if negotiation_stage in {"negotiating_plan"}:
                return "negotiate_installment", "discuss_arrangement", "negotiation"
            if negotiation_stage in {"confirming_commitment"}:
                return "confirm_commitment", "ask_commitment_date", "negotiation"
            if negotiation_stage in {"awaiting_customer_decision"}:
                return "present_arrangement_options", "ask_affordable_amount", "negotiation"
            if customer_payment_posture == "needs_arrangement":
                return "present_arrangement_options", "discuss_arrangement", "negotiation"
            return "assess_affordability", "ask_affordable_amount", "empathetic"
        if customer_payment_posture == "needs_arrangement":
            return "present_arrangement_options", "present_offer", "negotiation"
        return "explain_dues", "present_due_amount", "informational"

    @staticmethod
    def _required_response_elements_for_objective(*, objective: str) -> list[str]:
        mapping = {
            "collect_verification": ["ask_only_missing_verification_fields"],
            "explain_dues": ["mention_due_amount", "ask_next_step"],
            "assess_affordability": ["acknowledge_hardship", "ask_affordable_amount"],
            "present_arrangement_options": ["discuss_arrangement", "ask_next_step"],
            "negotiate_installment": ["ask_affordable_amount", "discuss_arrangement"],
            "confirm_commitment": ["ask_commitment_date", "confirm_amount_or_date"],
            "capture_promise": ["ask_commitment_date", "confirm_amount_or_date"],
            "handoff_to_offer_agent": ["handoff_payload"],
            "close_conversation": ["closure"],
        }
        return list(mapping.get(objective, []))

    @staticmethod
    def _forbidden_dialogue_actions_for_objective(*, objective: str) -> list[str]:
        mapping = {
            "collect_verification": ["disclose_dues_before_verification", "restart_collections_menu", "mention_internal_processing"],
            "explain_dues": ["disclose_dues_before_verification", "mention_internal_processing"],
            "assess_affordability": ["restart_collections_menu", "ask_pay_now_or_arrangement", "mention_internal_processing"],
            "present_arrangement_options": ["restart_collections_menu", "ask_pay_now_or_arrangement", "mention_internal_processing"],
            "negotiate_installment": ["restart_collections_menu", "ask_pay_now_or_arrangement", "mention_internal_processing"],
            "confirm_commitment": ["restart_collections_menu", "ask_pay_now_or_arrangement", "mention_internal_processing"],
            "capture_promise": ["restart_collections_menu", "ask_pay_now_or_arrangement", "mention_internal_processing"],
            "handoff_to_offer_agent": ["restart_collections_menu", "mention_internal_processing"],
            "close_conversation": ["restart_collections_menu", "mention_internal_processing"],
        }
        return list(mapping.get(objective, []))

    @staticmethod
    def _allowed_dialogue_actions_for_objective(*, objective: str) -> list[str]:
        mapping = {
            "collect_verification": ["ask_verification"],
            "explain_dues": ["present_due_amount", "ask_next_step"],
            "assess_affordability": ["acknowledge_hardship", "ask_affordable_amount"],
            "present_arrangement_options": ["present_offer", "discuss_arrangement"],
            "negotiate_installment": ["discuss_arrangement", "ask_affordable_amount"],
            "confirm_commitment": ["ask_commitment_date", "confirm_payment_intent"],
            "capture_promise": ["ask_commitment_date", "confirm_payment_intent"],
            "handoff_to_offer_agent": ["handoff"],
            "close_conversation": ["close_conversation"],
        }
        return list(mapping.get(objective, []))

    @staticmethod
    def _customer_facing_goal_for_objective(*, objective: str, memory_state: dict[str, Any]) -> str:
        name = str(memory_state.get("active_customer_name", "Customer")).strip() or "Customer"
        goals = {
            "collect_verification": "Ask only for the missing verification details needed to continue securely.",
            "explain_dues": "Explain the overdue amount clearly and ask the next useful payment question.",
            "assess_affordability": "Ask what monthly amount is realistically manageable after hardship disclosure.",
            "present_arrangement_options": "Continue arrangement discussion with practical repayment options.",
            "negotiate_installment": "Refine the repayment arrangement toward a manageable installment.",
            "confirm_commitment": "Confirm the amount and payment date the customer can commit to.",
            "capture_promise": "Capture the promise details and close on a specific commitment.",
            "handoff_to_offer_agent": "Prepare a specialist handoff for offer or discount review.",
            "close_conversation": "Close the conversation cleanly and professionally.",
        }
        goal = goals.get(objective, "Continue the conversation naturally.")
        return f"{name}: {goal}" if name else goal
