"""Plan proposal node for collections conversation planning."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel, Field

from agents.collection_agent.llm_structured import StructuredOutputRunner
from src.nodes.base import BaseGraphNode
from src.nodes.types import AgentState, NodeUpdate


class _PlanTreeNode(BaseModel):
    id: str
    label: str
    owner: str = "collection_agent"
    status: str = "pending"


class _PlanTreeEdge(BaseModel):
    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    condition: str | None = None


class _PlanTreeUpdate(BaseModel):
    operation: str = "advance"
    current_node_id: str | None = None
    selected_next_node_id: str | None = None
    new_nodes: list[_PlanTreeNode] = Field(default_factory=list)
    new_edges: list[_PlanTreeEdge] = Field(default_factory=list)
    remove_node_ids: list[str] = Field(default_factory=list)
    mark_done: list[str] = Field(default_factory=list)
    mark_skipped: list[str] = Field(default_factory=list)
    mark_blocked: list[str] = Field(default_factory=list)
    status: str | None = None


class _PlanProposalPayload(BaseModel):
    target: str = "customer"
    intent: str = "generic_plan"
    plan_outline: str
    draft_response: str | None = None
    next_actions: list[str] = Field(default_factory=list)
    plan_tree_update: _PlanTreeUpdate | None = None


class _PlanSignalPayload(BaseModel):
    needs_discount_specialist: bool = False
    is_plan_request: bool = False
    is_plan_rejection: bool = False
    hardship_signal: bool = False
    hardship_reason: str = "income_reduction"
    suggested_plan_mode: str = "strict_collections"
    reason: str | None = None


@dataclass(slots=True)
class PlanProposalNode(BaseGraphNode):
    """Builds conversation plan proposals and updates a per-session plan tree.

    State Keys Read:
    - `user_input`
    - `routing_context`
    - `observations`
    - `observation` (latest compatibility mirror)
    - `decision`
    - `response_target`
    - `route`
    - `plan_proposal`
    - `conversation_plan`
    - `verification_*` keys from state/memory
    - `memory` (reads and updates `memory.state` for plan mode, plan revision, active plan, etc.)

    State Keys Write:
    - `route`
    - `response`
    - `response_target`
    - `plan_proposal`
    - `conversation_plan`
    - `additional_targets` (optional)
    - `memory_helper_trigger` (optional)
    - `handoff_payload` (optional)
    - `prompt`
    - `system_prompt`
    - `llm_response`
    - `llm_error`
    """

    llm: Any | None = None
    system_prompt: str = ""
    user_prompt: str = ""
    classifier_system_prompt: str = ""
    classifier_user_prompt: str = ""
    strict_llm_mode: bool = True
    last_debug: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    max_json_chars: int = 900

    def execute(self, state: AgentState) -> NodeUpdate:
        self._record_llm_usage(state, node_name="plan_proposal")
        self.last_debug = {
            "prompt": None,
            "system_prompt": None,
            "llm_response": None,
            "llm_error": None,
        }
        memory = state.get("memory")
        memory_state = dict(getattr(memory, "state", {})) if memory is not None else {}
        mode = str(memory_state.get("mode", "strict_collections"))
        routing_context = state.get("routing_context") if isinstance(state.get("routing_context"), dict) else {}
        plan_origin = str(routing_context.get("plan_origin", "react"))
        observation = None
        observations = state.get("observations")
        if isinstance(observations, list):
            for item in reversed(observations):
                if isinstance(item, dict):
                    observation = item
                    break
        if observation is None:
            observation = state.get("observation")
        user_input = str(state.get("user_input", ""))
        memory_state = self._overlay_verification_state_from_graph(
            state=state,
            memory_state=memory_state,
        )

        if isinstance(observation, dict) and isinstance(observation.get("tool_phase"), dict):
            observation = observation.get("tool_phase")
        observed_tool = str(observation.get("tool_name", "")) if isinstance(observation, dict) else ""
        output = observation.get("output", {}) if isinstance(observation, dict) else {}
        decision = state.get("decision")
        existing_plan = self._get_existing_conversation_plan(state=state, memory_state=memory_state)

        plan_signals = self._classify_plan_signals(
            user_input=user_input,
            mode=mode,
            memory_state=memory_state,
            existing_plan=existing_plan,
        )
        identity_verified = bool(memory_state.get("identity_verified", False))
        suggested_mode = str(plan_signals.get("suggested_plan_mode", mode)).strip().lower()
        if suggested_mode in {"strict_collections", "hardship_negotiation"} and suggested_mode != mode:
            mode = suggested_mode
            if memory is not None:
                memory.set_state(mode=mode)

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
            plan = self._build_or_update_conversation_plan(
                existing_plan=existing_plan,
                user_input=user_input,
                memory_state=memory_state,
                mode=mode,
                plan_origin=plan_origin,
                response_target=response_target,
                route=route,
                observed_tool=observed_tool,
                proposal=proposal,
                plan_signals=plan_signals,
            )
            update["conversation_plan"] = plan
            if proposal:
                proposal = self._align_customer_proposal_with_plan(
                    proposal=proposal,
                    plan=plan,
                    memory_state=memory_state,
                    plan_signals=plan_signals,
                )
                current_id = str(plan.get("current_node_id", ""))
                proposal["conversation_plan_id"] = plan.get("plan_id")
                proposal["conversation_plan_version"] = plan.get("version")
                proposal["conversation_plan_current_node"] = current_id
                proposal["conversation_plan_current_label"] = self._node_label(plan, current_id)
                proposal["conversation_plan"] = plan
                update["plan_proposal"] = proposal
            if memory is not None:
                memory.set_state(active_conversation_plan=plan)
            return with_debug(update)

        if bool(memory_state.get("agent_loop_blocked", False)):
            if memory is not None:
                memory.set_state(agent_loop_blocked=False)
            return with_plan(
                {
                    "route": "continue",
                    "response_target": "customer",
                    "plan_proposal": {
                        "target": "customer",
                        "intent": "loop_guard",
                        "guidance": "Internal planning loop exceeded threshold.",
                        "next_actions": ["pay_now", "plan_revision", "schedule_followup"],
                        "plan_origin": "loop_guard",
                    },
                }
            )

        if self._is_conversation_termination(user_input):
            return with_plan(
                {
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
                }
            )

        discount_recommendation = memory_state.get("discount_recommendation")
        if isinstance(discount_recommendation, dict) and discount_recommendation:
            if memory is not None:
                memory.set_state(discount_recommendation=None)
            return with_plan(
                {
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
                }
            )

        revision_index = int(memory_state.get("plan_revision_index", 0))
        hardship_reason = str(plan_signals.get("hardship_reason") or memory_state.get("hardship_reason", "income_reduction"))
        case_id = str(memory_state.get("active_case_id", "COLL-1001"))

        if bool(plan_signals.get("needs_discount_specialist")) and case_id:
            if not identity_verified:
                return with_plan(
                    {
                        "route": "continue",
                        "response_target": "customer",
                        "plan_proposal": {
                            "target": "customer",
                            "intent": "verification_required_before_discount",
                            "plan_outline": "Complete identity verification before evaluating discount/restructure options.",
                            "next_actions": ["complete_identity_verification", "then_evaluate_assistance"],
                        },
                    }
                )
            if memory is not None and hardship_reason:
                memory.set_state(hardship_reason=hardship_reason)
            return with_plan(
                {
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
                }
            )

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
            return with_plan(
                {
                    "route": "continue",
                    "response_target": str(plan_proposal.get("target", "customer")),
                    "plan_proposal": plan_proposal,
                }
            )

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
            return with_plan(
                {
                    "route": "continue",
                    "response_target": "customer",
                    "plan_proposal": plan_proposal,
                }
            )

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
            return with_plan(
                {
                    "route": "continue",
                    "response_target": str(plan_proposal.get("target", "customer")),
                    "plan_proposal": plan_proposal,
                }
            )

        max_installment = self._extract_amount(user_input)
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
        return with_plan(
            {
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
            }
        )

    def route(self, state: AgentState) -> str:
        return str(state.get("route", "continue"))

    @staticmethod
    def _is_plan_rejection(text: str) -> bool:
        lowered = text.lower()
        return any(key in lowered for key in ["not work", "can't", "cannot", "too high", "reject", "no,", "no "])

    @staticmethod
    def _is_plan_request(text: str) -> bool:
        lowered = text.lower()
        return any(key in lowered for key in ["payment plan", "plan option", "need plan", "proposal"])

    @staticmethod
    def _extract_amount(text: str) -> float | None:
        match = re.search(r"(?:\\$|inr\\s*)?(\\d+(?:\\.\\d+)?)", text, re.IGNORECASE)
        if not match:
            return None
        return float(match.group(1))

    @staticmethod
    def _needs_discount_specialist(text: str) -> bool:
        lowered = text.lower()
        keywords = [
            "discount",
            "waiver",
            "concession",
            "benefit",
            "lower emi",
            "reduce emi",
            "settlement",
        ]
        return any(keyword in lowered for keyword in keywords)

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
        if self.strict_llm_mode:
            llm_error = str(self.last_debug.get("llm_error", "")).strip()
            lowered = llm_error.lower()
            if (
                "rate limit" in lowered
                or "rate_limit_exceeded" in lowered
                or "error code: 429" in lowered
                or "tokens per day" in lowered
                or "tpm" in lowered
            ):
                raise RuntimeError(
                    "PlanProposalNode rate-limited by provider while strict_llm_mode is enabled. "
                    f"Underlying error: {llm_error}"
                )
            raise RuntimeError(
                "PlanProposalNode failed to produce LLM structured output while strict_llm_mode is enabled. "
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
            "existing_plan_json": self._json_compact(
                self._compact_existing_plan_for_prompt(existing_plan),
                max_chars=self.max_json_chars,
            ),
            "decision_payload_json": json.dumps(decision_payload, ensure_ascii=True, default=str),
            "obs_tool": obs_tool,
            "obs_output_json": self._json_compact(obs_output, max_chars=900),
            "reflection_feedback_json": self._json_compact(reflection_feedback, max_chars=700),
            "customer_context_json": customer_context_json,
            "verification_context_json": verification_context_json,
            "entities_context_json": entities_context_json,
        }
        system_prompt = self._render_prompt_template(self.system_prompt, template_vars)
        user_prompt = self._render_prompt_template(self.user_prompt, template_vars)
        self.last_debug["prompt"] = user_prompt
        self.last_debug["system_prompt"] = system_prompt or None
        self.last_debug["llm_error"] = None
        try:
            payload = StructuredOutputRunner(self.llm, max_retries=2).run(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=_PlanProposalPayload,
            )
            prior = self.last_debug.get("llm_response")
            merged = dict(prior) if isinstance(prior, dict) else {}
            merged["proposal"] = payload.model_dump(mode="json", by_alias=True)
            self.last_debug["llm_response"] = merged
        except Exception as exc:
            self.last_debug["llm_error"] = str(exc)
            minimal_template_vars = {
                "user_input": self._truncate_text(user_input, 280),
                "plan_origin": plan_origin,
                "mode": mode,
                "default_plan": self._truncate_text(default_plan, 220),
                "existing_plan_json": self._json_compact(
                    self._compact_existing_plan_for_prompt(existing_plan, minimal=True),
                    max_chars=900,
                ),
                "decision_payload_json": self._json_compact(
                    {
                        "response_text": str(decision_payload.get("response_text", ""))[:180],
                        "respond_directly": bool(decision_payload.get("respond_directly", False)),
                        "tool_name": str(((decision_payload.get("tool_call") or {}).get("tool_name", ""))),
                    },
                    max_chars=500,
                ),
                "obs_tool": obs_tool,
                "obs_output_json": self._json_compact(
                    {
                        "status": (obs_output or {}).get("status") if isinstance(obs_output, dict) else None,
                        "needs_additional_action": bool((obs_output or {}).get("needs_additional_action", False))
                        if isinstance(obs_output, dict)
                        else False,
                        "keys": sorted([str(k) for k in obs_output.keys()])[:8] if isinstance(obs_output, dict) else [],
                    },
                    max_chars=350,
                ),
                "reflection_feedback_json": self._json_compact(reflection_feedback, max_chars=500),
                "customer_context_json": customer_context_json,
                "verification_context_json": verification_context_json,
                "entities_context_json": self._json_compact(
                    {
                        "verification_entities": state.get("verification_entities", {}),
                        "verification_missing_fields": memory_state.get("verification_missing_fields", []),
                        "identity_verified": bool(memory_state.get("identity_verified", False)),
                    },
                    max_chars=500,
                ),
            }
            minimal_system_prompt = self._render_prompt_template(self.system_prompt, minimal_template_vars)
            minimal_user_prompt = self._render_prompt_template(self.user_prompt, minimal_template_vars)
            self.last_debug["prompt"] = minimal_user_prompt
            self.last_debug["system_prompt"] = minimal_system_prompt or None
            try:
                payload = StructuredOutputRunner(self.llm, max_retries=4).run(
                    system_prompt=minimal_system_prompt,
                    user_prompt=minimal_user_prompt,
                    schema=_PlanProposalPayload,
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
    def _verification_required_fields(memory_state: dict[str, Any]) -> list[str]:
        required = memory_state.get("active_verification_required_fields")
        required_fields = [str(x).strip().lower() for x in required if str(x).strip()] if isinstance(required, list) else []
        if required_fields:
            return sorted(set(required_fields))
        return ["dob", "phone"]

    def _overlay_verification_state_from_graph(
        self,
        *,
        state: AgentState,
        memory_state: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(memory_state)
        if isinstance(state.get("verification_entities"), dict):
            merged["verification_entities"] = dict(state.get("verification_entities", {}))
            merged["verification_collected"] = dict(state.get("verification_entities", {}))
        if isinstance(state.get("verification_missing_fields"), list):
            merged["verification_missing_fields"] = [
                str(x).strip().lower() for x in state.get("verification_missing_fields", []) if str(x).strip()
            ]
        if isinstance(state.get("verification_verified_fields"), list):
            merged["verification_verified_fields"] = [
                str(x).strip().lower() for x in state.get("verification_verified_fields", []) if str(x).strip()
            ]
        for key in ("verified_dob", "verified_mobile", "identity_verified"):
            if key in state:
                merged[key] = bool(state.get(key))
        return merged

    @staticmethod
    def _render_prompt_template(template: str, values: dict[str, Any]) -> str:
        rendered = template
        for key, value in values.items():
            rendered = rendered.replace(f"{{{key}}}", str(value))
        return rendered

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
    def _get_existing_conversation_plan(*, state: AgentState, memory_state: dict[str, Any]) -> dict[str, Any]:
        state_plan = state.get("conversation_plan")
        if isinstance(state_plan, dict) and state_plan:
            return dict(state_plan)
        memory_plan = memory_state.get("active_conversation_plan")
        if isinstance(memory_plan, dict) and memory_plan:
            return dict(memory_plan)
        return {}

    def _build_or_update_conversation_plan(
        self,
        *,
        existing_plan: dict[str, Any],
        user_input: str,
        memory_state: dict[str, Any],
        mode: str,
        plan_origin: str,
        response_target: str,
        route: str,
        observed_tool: str,
        proposal: dict[str, Any],
        plan_signals: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        plan = self._create_initial_plan_graph(memory_state=memory_state, mode=mode) if not existing_plan else dict(existing_plan)
        plan.setdefault("nodes", [])
        plan.setdefault("edges", [])
        plan.setdefault("timeline", [])
        plan.setdefault("revision_log", [])
        plan.setdefault("status", "active")
        plan.setdefault("version", 1)
        plan.setdefault("objective", "Drive compliant collections conversation toward payment resolution.")
        plan.setdefault("plan_id", f"plan-{str(memory_state.get('active_case_id', 'COLL-1001')).strip().upper()}")
        plan.setdefault("root_node_id", "open_and_context")
        plan.setdefault("step_markers", {})

        previous_current = str(plan.get("current_node_id", "")) or str(plan.get("root_node_id", "open_and_context"))
        plan_update = proposal.get("plan_tree_update") if isinstance(proposal.get("plan_tree_update"), dict) else {}
        proposal_context = proposal.get("context") if isinstance(proposal.get("context"), dict) else {}
        observed_output = (
            proposal_context.get("observed_tool_output")
            if isinstance(proposal_context.get("observed_tool_output"), dict)
            else {}
        )
        inferred_next = self._infer_current_node_id(
            user_input=user_input,
            observed_tool=observed_tool,
            response_target=response_target,
            route=route,
            proposal=proposal,
            plan_update=plan_update,
            previous_current=previous_current,
        )
        self._apply_plan_tree_update(plan=plan, plan_update=plan_update)
        markers = self._init_or_reconcile_step_markers(plan=plan)
        self._apply_step_marker_updates(
            plan=plan,
            plan_update=plan_update,
            previous_current=previous_current,
            candidate=inferred_next,
            user_input=user_input,
            observed_tool=observed_tool,
            observed_output=(observed_output if isinstance(observed_output, dict) else {}),
            memory_identity_verified=bool(memory_state.get("identity_verified", False)),
        )
        self._enforce_verification_marker_consistency(
            plan=plan,
            memory_identity_verified=bool(memory_state.get("identity_verified", False)),
            observed_tool=observed_tool,
            observed_output=(observed_output if isinstance(observed_output, dict) else {}),
        )
        self._remove_verify_identity_node_if_verified(
            plan=plan,
            identity_verified=bool(memory_state.get("identity_verified", False)),
        )
        markers = self._init_or_reconcile_step_markers(plan=plan)
        next_current = self._resolve_next_current_node(
            plan=plan,
            previous_current=previous_current,
            candidate=inferred_next,
            markers=markers,
        )
        self._prune_disconnected_nodes(plan=plan, keep_ids={next_current})
        self._enforce_status_consistency(
            plan=plan,
            current_node_id=next_current,
            response_target=response_target,
            markers=markers,
        )
        markers = self._init_or_reconcile_step_markers(plan=plan)

        lowered = user_input.lower()
        signals = plan_signals or {}
        should_revise = (
            bool(signals.get("is_plan_rejection", False))
            or bool(signals.get("needs_discount_specialist", False))
            or bool(signals.get("hardship_signal", False))
            or "hardship" in lowered
            or "cannot pay" in lowered
            or response_target != "customer"
            or bool(plan_update.get("new_nodes"))
            or bool(plan_update.get("remove_node_ids"))
        )
        if should_revise:
            plan["version"] = int(plan.get("version", 1)) + 1
            plan["revision_log"].append(
                {
                    "revision": int(plan.get("version", 1)),
                    "reason": f"context shift: target={response_target}, route={route}, origin={plan_origin}",
                    "at_utc": datetime.now(UTC).isoformat(),
                }
            )

        plan["current_node_id"] = next_current
        plan["previous_node_id"] = previous_current or None
        plan["next_node_ids"] = self._next_nodes_from_edges(nodes=plan.get("edges", []), current_node_id=next_current)
        plan["updated_from"] = plan_origin or "react"
        plan["last_response_target"] = response_target
        status_override = str(plan_update.get("status", "")).strip().lower() if isinstance(plan_update, dict) else ""
        if status_override in {"active", "completed"}:
            plan["status"] = status_override
        else:
            plan["status"] = "completed" if self._is_conversation_termination(user_input) else "active"
        self._append_timeline_snapshot(
            plan=plan,
            update={
                "origin": plan_origin,
                "route": route,
                "response_target": response_target,
                "previous_node_id": previous_current,
                "current_node_id": next_current,
                "plan_outline": str(proposal.get("plan_outline", "")).strip(),
                "operation": str(plan_update.get("operation", "advance")) if isinstance(plan_update, dict) else "advance",
            },
        )
        return plan

    def _remove_verify_identity_node_if_verified(self, *, plan: dict[str, Any], identity_verified: bool) -> None:
        if not identity_verified:
            return
        # Keep completed steps visible in topology: do not remove verify_identity.
        nodes = [dict(node) for node in plan.get("nodes", []) if isinstance(node, dict)]
        for node in nodes:
            if str(node.get("id", "")).strip() == "verify_identity":
                node["status"] = "done"
        plan["nodes"] = nodes
        markers = plan.get("step_markers")
        if isinstance(markers, dict):
            prior = markers.get("verify_identity") if isinstance(markers.get("verify_identity"), dict) else {}
            markers["verify_identity"] = {
                "state": "done",
                "updated_at": datetime.now(UTC).isoformat(),
                "source": "memory_identity_verified",
                "reason": str(prior.get("reason", "identity_verified_in_memory")),
            }
            plan["step_markers"] = markers

    @staticmethod
    def _create_initial_plan_graph(*, memory_state: dict[str, Any], mode: str) -> dict[str, Any]:
        case_id = str(memory_state.get("active_case_id", "COLL-1001")).strip().upper() or "COLL-1001"
        identity_verified = bool(memory_state.get("identity_verified", False))
        nodes = [
            {"id": "open_and_context", "label": "Initialize case context", "owner": "collection_agent", "status": "in_progress"},
            {"id": "verify_identity", "label": "Verify customer identity", "owner": "customer", "status": ("done" if identity_verified else "pending")},
            {"id": "explain_dues", "label": "Explain dues and policy options", "owner": "customer", "status": "pending"},
            {"id": "collect_payment_intent", "label": "Collect payment intent", "owner": "customer", "status": "pending"},
            {"id": "evaluate_assistance", "label": "Evaluate discount/restructure assistance", "owner": "collection_agent", "status": "pending"},
            {"id": "resolve_outcome", "label": "Finalize payment, promise, or follow-up", "owner": "customer", "status": "pending"},
        ]
        edges = [
            {"from": "open_and_context", "to": "verify_identity", "condition": "case_context_ready"},
            {"from": "verify_identity", "to": "explain_dues", "condition": "identity_verified"},
            {"from": "explain_dues", "to": "collect_payment_intent", "condition": "dues_explained"},
            {"from": "collect_payment_intent", "to": "resolve_outcome", "condition": "pay_now"},
            {"from": "collect_payment_intent", "to": "evaluate_assistance", "condition": "cannot_pay_full"},
            {"from": "evaluate_assistance", "to": "resolve_outcome", "condition": "assistance_ready"},
        ]
        next_node_ids = ["verify_identity"] if not identity_verified else ["explain_dues"]
        return {
            "plan_id": f"plan-{case_id}",
            "version": 1,
            "status": "active",
            "mode": mode,
            "objective": "Move borrower conversation to payment, promise-to-pay, or compliant follow-up.",
            "root_node_id": "open_and_context",
            "current_node_id": "open_and_context",
            "previous_node_id": None,
            "next_node_ids": next_node_ids,
            "nodes": nodes,
            "edges": edges,
            "timeline": [],
            "revision_log": [],
            "updated_from": "initial",
            "last_response_target": "customer",
        }

    def _apply_plan_tree_update(self, *, plan: dict[str, Any], plan_update: dict[str, Any]) -> None:
        if not isinstance(plan_update, dict) or not plan_update:
            return
        nodes = [dict(node) for node in plan.get("nodes", []) if isinstance(node, dict)]
        edges = [dict(edge) for edge in plan.get("edges", []) if isinstance(edge, dict)]
        node_map = {str(node.get("id", "")): node for node in nodes if str(node.get("id", "")).strip()}
        historical_ids = self._historical_node_ids(plan=plan)

        for node_id in [str(x).strip() for x in plan_update.get("remove_node_ids", []) if str(x).strip()]:
            if node_id in historical_ids:
                continue
            node_map.pop(node_id, None)
        if plan_update.get("remove_node_ids"):
            removed_ids = {str(x).strip() for x in plan_update.get("remove_node_ids", []) if str(x).strip()}
            removed_ids = {node_id for node_id in removed_ids if node_id not in historical_ids}
            edges = [
                edge
                for edge in edges
                if str(edge.get("from", "")).strip() not in removed_ids and str(edge.get("to", "")).strip() not in removed_ids
            ]

        for raw in plan_update.get("new_nodes", []):
            if not isinstance(raw, dict):
                continue
            node_id = str(raw.get("id", "")).strip()
            if not node_id:
                continue
            node_map[node_id] = {
                "id": node_id,
                "label": str(raw.get("label", node_id)).strip() or node_id,
                "owner": str(raw.get("owner", "collection_agent")).strip() or "collection_agent",
                "status": str(raw.get("status", "pending")).strip() or "pending",
            }

        edge_set = set()
        for edge in edges:
            src = str(edge.get("from", "")).strip()
            dst = str(edge.get("to", "")).strip()
            cond = str(edge.get("condition", "")).strip()
            if not src or not dst:
                continue
            if src not in node_map or dst not in node_map:
                continue
            edge_set.add((src, dst, cond))
        for raw in plan_update.get("new_edges", []):
            if not isinstance(raw, dict):
                continue
            src = str(raw.get("from", "")).strip()
            dst = str(raw.get("to", "")).strip()
            cond = str(raw.get("condition", "")).strip()
            if not src or not dst:
                continue
            if src not in node_map or dst not in node_map:
                continue
            edge_set.add((src, dst, cond))

        mark_done = {str(x).strip() for x in plan_update.get("mark_done", []) if str(x).strip()}
        mark_skipped = {str(x).strip() for x in plan_update.get("mark_skipped", []) if str(x).strip()}
        mark_blocked = {str(x).strip() for x in plan_update.get("mark_blocked", []) if str(x).strip()}
        for node_id, node in node_map.items():
            if node_id in mark_done:
                node["status"] = "done"
            elif node_id in mark_skipped:
                node["status"] = "skipped"
            elif node_id in mark_blocked:
                node["status"] = "blocked"

        plan["nodes"] = list(node_map.values())
        plan["edges"] = [
            {"from": src, "to": dst, "condition": cond}
            for (src, dst, cond) in sorted(edge_set)
        ]

    def _resolve_next_current_node(
        self,
        *,
        plan: dict[str, Any],
        previous_current: str,
        candidate: str,
        markers: dict[str, Any],
    ) -> str:
        node_ids = {str(node.get("id")) for node in plan.get("nodes", []) if isinstance(node, dict)}
        parent_map = self._parent_map(nodes=plan.get("edges", []))

        def is_actionable(node_id: str) -> bool:
            if node_id not in node_ids:
                return False
            if not self._is_node_unlocked(plan=plan, node_id=node_id, markers=markers):
                return False
            return self._marker_state(markers=markers, node_id=node_id) == "pending"

        def first_actionable_descendant(start_node_id: str) -> str:
            queue: list[str] = [start_node_id]
            visited: set[str] = set()
            while queue:
                nid = queue.pop(0)
                if not nid or nid in visited:
                    continue
                visited.add(nid)
                if is_actionable(nid):
                    return nid
                children = self._next_nodes_from_edges(nodes=plan.get("edges", []), current_node_id=nid)
                for child in children:
                    if child not in visited:
                        queue.append(child)
            return ""

        def nearest_unlocked(node_id: str) -> str:
            cursor = node_id
            visited: set[str] = set()
            while cursor and cursor not in visited:
                visited.add(cursor)
                if cursor in node_ids and self._is_node_unlocked(
                    plan=plan,
                    node_id=cursor,
                    markers=markers,
                ):
                    return cursor
                cursor = parent_map.get(cursor, "")
            root_candidate = str(plan.get("root_node_id", "open_and_context"))
            return root_candidate if root_candidate in node_ids else node_id

        if not previous_current or previous_current not in node_ids:
            root = str(plan.get("root_node_id", "open_and_context"))
            if is_actionable(candidate):
                return candidate
            first = first_actionable_descendant(root)
            if first:
                return first
            if candidate in node_ids and self._is_node_unlocked(plan=plan, node_id=candidate, markers=markers):
                return candidate
            return root if root in node_ids else candidate

        previous_current = nearest_unlocked(previous_current)
        allowed_next = self._next_nodes_from_edges(nodes=plan.get("edges", []), current_node_id=previous_current)
        if not allowed_next:
            if is_actionable(candidate):
                return candidate
            if candidate in node_ids and self._is_node_unlocked(plan=plan, node_id=candidate, markers=markers):
                return candidate
            first = first_actionable_descendant(previous_current)
            if first:
                return first
            return previous_current

        if candidate in allowed_next and is_actionable(candidate):
            return candidate

        # Enforce sequential progression with marker gates.
        for node_id in allowed_next:
            if is_actionable(node_id):
                return node_id
        for node_id in allowed_next:
            first = first_actionable_descendant(node_id)
            if first:
                return first
        return previous_current

    @staticmethod
    def _parent_map(*, nodes: list[Any]) -> dict[str, str]:
        parent_map: dict[str, str] = {}
        for edge in nodes:
            if not isinstance(edge, dict):
                continue
            src = str(edge.get("from", "")).strip()
            dst = str(edge.get("to", "")).strip()
            if not src or not dst:
                continue
            if dst not in parent_map:
                parent_map[dst] = src
        return parent_map

    def _enforce_status_consistency(
        self,
        *,
        plan: dict[str, Any],
        current_node_id: str,
        response_target: str,
        markers: dict[str, Any],
    ) -> None:
        nodes = [node for node in plan.get("nodes", []) if isinstance(node, dict)]
        node_map = {str(node.get("id", "")).strip(): node for node in nodes if str(node.get("id", "")).strip()}
        if current_node_id not in node_map:
            return

        for node_id, node in node_map.items():
            marker_state = self._marker_state(markers=markers, node_id=node_id)
            if node_id == current_node_id:
                if marker_state in {"done", "skipped", "blocked"}:
                    node["status"] = marker_state
                else:
                    node["status"] = "in_progress"
                    node["owner"] = "collection_agent" if response_target == "self" else node.get("owner", "customer")
                continue
            if marker_state in {"done", "skipped", "blocked"}:
                node["status"] = marker_state
            else:
                node["status"] = "pending"

    @staticmethod
    def _marker_state(*, markers: dict[str, Any], node_id: str) -> str:
        raw = markers.get(node_id)
        if isinstance(raw, dict):
            state = str(raw.get("state", "pending")).strip().lower()
        else:
            state = str(raw or "pending").strip().lower()
        if state in {"done", "skipped", "blocked", "pending"}:
            return state
        return "pending"

    def _init_or_reconcile_step_markers(self, *, plan: dict[str, Any]) -> dict[str, Any]:
        existing = plan.get("step_markers") if isinstance(plan.get("step_markers"), dict) else {}
        markers: dict[str, Any] = dict(existing)
        has_existing_markers = bool(existing)
        root_id = str(plan.get("root_node_id", "open_and_context")).strip() or "open_and_context"
        now = datetime.now(UTC).isoformat()

        node_ids: set[str] = set()
        for node in plan.get("nodes", []):
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue
            node_ids.add(node_id)
            status = str(node.get("status", "pending")).strip().lower()
            previous = markers.get(node_id) if isinstance(markers.get(node_id), dict) else {}
            previous_state = str(previous.get("state", "pending")).strip().lower()
            if previous_state not in {"done", "skipped", "blocked", "pending"}:
                previous_state = "pending"

            if previous_state in {"done", "skipped", "blocked"}:
                state = previous_state
            elif (not has_existing_markers) and status in {"done", "skipped", "blocked"}:
                # Bootstrap marker states from node status for initial sessions.
                state = status
            elif node_id == root_id and not existing:
                state = "pending"
            else:
                state = "pending"

            markers[node_id] = {
                "state": state,
                "updated_at": str(previous.get("updated_at", now)),
                "source": str(previous.get("source", "reconciler")),
                "reason": str(previous.get("reason", "")),
            }

        for marker_id in list(markers.keys()):
            if marker_id not in node_ids:
                markers.pop(marker_id, None)

        plan["step_markers"] = markers
        return markers

    def _apply_step_marker_updates(
        self,
        *,
        plan: dict[str, Any],
        plan_update: dict[str, Any],
        previous_current: str,
        candidate: str,
        user_input: str,
        observed_tool: str,
        observed_output: dict[str, Any],
        memory_identity_verified: bool,
    ) -> None:
        markers = plan.get("step_markers") if isinstance(plan.get("step_markers"), dict) else {}
        now = datetime.now(UTC).isoformat()
        if not isinstance(plan_update, dict):
            plan["step_markers"] = markers
            return

        mark_done = {str(x).strip() for x in plan_update.get("mark_done", []) if str(x).strip()}
        mark_skipped = {str(x).strip() for x in plan_update.get("mark_skipped", []) if str(x).strip()}
        mark_blocked = {str(x).strip() for x in plan_update.get("mark_blocked", []) if str(x).strip()}

        def set_state(node_id: str, state: str, reason: str) -> None:
            prior = markers.get(node_id) if isinstance(markers.get(node_id), dict) else {}
            markers[node_id] = {
                "state": state,
                "updated_at": now,
                "source": "plan_tree_update",
                "reason": reason or str(prior.get("reason", "")),
            }

        for node_id in sorted(mark_done):
            if self._can_accept_done_marker(
                plan=plan,
                node_id=node_id,
                user_input=user_input,
                observed_tool=observed_tool,
                observed_output=observed_output,
                memory_identity_verified=memory_identity_verified,
            ):
                set_state(node_id, "done", "mark_done")
        for node_id in sorted(mark_skipped):
            if self._can_accept_skip_marker(
                plan=plan,
                node_id=node_id,
                previous_current=previous_current,
                user_input=user_input,
                memory_identity_verified=memory_identity_verified,
                markers=markers,
            ):
                set_state(node_id, "skipped", "mark_skipped")
        for node_id in sorted(mark_blocked):
            set_state(node_id, "blocked", "mark_blocked")

        if memory_identity_verified:
            # Deterministic safety: if identity is already verified in memory state,
            # keep the marker aligned even when the model omits mark_done.
            set_state("verify_identity", "done", "memory_identity_verified")

        root_id = str(plan.get("root_node_id", "open_and_context")).strip() or "open_and_context"
        if (
            previous_current == root_id
            and candidate
            and candidate != root_id
            and self._marker_state(markers=markers, node_id=root_id) == "pending"
        ):
            # Root step completes only when planner advances to next actionable step.
            set_state(root_id, "done", "case_context_ready_transition")

        if (
            previous_current
            and candidate
            and candidate != previous_current
            and self._marker_state(markers=markers, node_id=previous_current) == "pending"
        ):
            # Require explicit completion/skip marker before moving from previous step.
            set_state(previous_current, "pending", "awaiting_completion_or_skip_marker")

        plan["step_markers"] = markers

    def _enforce_verification_marker_consistency(
        self,
        *,
        plan: dict[str, Any],
        memory_identity_verified: bool,
        observed_tool: str,
        observed_output: dict[str, Any],
    ) -> None:
        markers = plan.get("step_markers") if isinstance(plan.get("step_markers"), dict) else {}
        verify_raw = markers.get("verify_identity")
        if not isinstance(verify_raw, dict):
            return
        verify_state = str(verify_raw.get("state", "pending")).strip().lower()
        if verify_state != "done":
            return

        if memory_identity_verified:
            return

        verify_raw["state"] = "pending"
        verify_raw["source"] = "reconciler"
        verify_raw["reason"] = "identity_not_verified_in_current_state"
        verify_raw["updated_at"] = datetime.now(UTC).isoformat()
        markers["verify_identity"] = verify_raw
        plan["step_markers"] = markers

    @staticmethod
    def _node_owner(*, plan: dict[str, Any], node_id: str) -> str:
        for node in plan.get("nodes", []):
            if not isinstance(node, dict):
                continue
            if str(node.get("id", "")).strip() != node_id:
                continue
            owner = str(node.get("owner", "collection_agent")).strip().lower()
            return owner or "collection_agent"
        return "collection_agent"

    @staticmethod
    def _node_label_by_id(*, plan: dict[str, Any], node_id: str) -> str:
        for node in plan.get("nodes", []):
            if not isinstance(node, dict):
                continue
            if str(node.get("id", "")).strip() != node_id:
                continue
            return str(node.get("label", node_id)).strip() or node_id
        return node_id

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9]+", str(text).lower())
        stop = {
            "and",
            "or",
            "the",
            "a",
            "an",
            "to",
            "of",
            "for",
            "with",
            "customer",
            "collection",
            "agent",
            "call",
            "context",
            "details",
            "outcome",
            "finalize",
        }
        return {token for token in tokens if token and token not in stop}

    def _tool_matches_node(self, *, node_id: str, node_label: str, observed_tool: str) -> bool:
        tool_tokens = self._tokenize(observed_tool)
        if not tool_tokens:
            return False
        node_tokens = self._tokenize(node_id) | self._tokenize(node_label)
        if not node_tokens:
            return False
        return bool(tool_tokens.intersection(node_tokens))

    def _can_accept_done_marker(
        self,
        *,
        plan: dict[str, Any],
        node_id: str,
        user_input: str,
        observed_tool: str,
        observed_output: dict[str, Any],
        memory_identity_verified: bool,
    ) -> bool:
        owner = self._node_owner(plan=plan, node_id=node_id)
        if node_id == str(plan.get("root_node_id", "open_and_context")).strip():
            return True
        if node_id == "verify_identity":
            # Identity completion is gated by effective verification state,
            # not by single-tool success in isolation.
            return bool(memory_identity_verified)
        if observed_tool:
            tool_status = str(observed_output.get("status", "")).strip().lower()
            if tool_status in {"failed", "locked", "error", "rejected", "denied", "invalid"}:
                return False
            owner = self._node_owner(plan=plan, node_id=node_id)
            if owner in {"customer", "borrower"}:
                label = self._node_label_by_id(plan=plan, node_id=node_id)
                return self._tool_matches_node(node_id=node_id, node_label=label, observed_tool=observed_tool)
            return True
        lowered = user_input.lower()
        if any(token in lowered for token in ["skip", "skipped", "defer", "later"]):
            return True
        if owner in {"customer", "borrower"}:
            # Customer-owned steps must be grounded in an observed tool result.
            return False
        return True

    def _can_accept_skip_marker(
        self,
        *,
        plan: dict[str, Any],
        node_id: str,
        previous_current: str,
        user_input: str,
        memory_identity_verified: bool,
        markers: dict[str, Any],
    ) -> bool:
        root_id = str(plan.get("root_node_id", "open_and_context")).strip() or "open_and_context"
        if node_id == root_id:
            return False
        if node_id == "verify_identity" and not memory_identity_verified:
            return False

        lowered = str(user_input).lower()
        explicit_skip = any(
            token in lowered
            for token in [
                "skip",
                "skipped",
                "defer",
                "later",
                "not now",
                "can't talk",
                "cannot talk",
                "call me later",
            ]
        )
        if not explicit_skip:
            return False

        current = str(previous_current or "").strip()
        if not current:
            return False
        if node_id == current:
            return True

        allowed_next = self._next_nodes_from_edges(nodes=plan.get("edges", []), current_node_id=current)
        if node_id not in allowed_next:
            return False
        return self._is_node_unlocked(plan=plan, node_id=node_id, markers=markers)

    def _is_node_unlocked(self, *, plan: dict[str, Any], node_id: str, markers: dict[str, Any]) -> bool:
        parents: list[str] = []
        for edge in plan.get("edges", []):
            if not isinstance(edge, dict):
                continue
            src = str(edge.get("from", "")).strip()
            dst = str(edge.get("to", "")).strip()
            if dst == node_id and src:
                parents.append(src)
        if not parents:
            return True
        for parent_id in parents:
            parent_state = self._marker_state(markers=markers, node_id=parent_id)
            if parent_state not in {"done", "skipped"}:
                return False
        return True

    def _prune_disconnected_nodes(self, *, plan: dict[str, Any], keep_ids: set[str]) -> None:
        nodes = [node for node in plan.get("nodes", []) if isinstance(node, dict)]
        node_map = {str(node.get("id", "")).strip(): node for node in nodes if str(node.get("id", "")).strip()}
        if not node_map:
            return
        root_id = str(plan.get("root_node_id", "")).strip() or next(iter(node_map.keys()))
        if root_id not in node_map:
            root_id = next(iter(node_map.keys()))
            plan["root_node_id"] = root_id

        forward_adj: dict[str, list[str]] = {node_id: [] for node_id in node_map}
        edges = [edge for edge in plan.get("edges", []) if isinstance(edge, dict)]
        filtered_edges: list[dict[str, Any]] = []
        for edge in edges:
            src = str(edge.get("from", "")).strip()
            dst = str(edge.get("to", "")).strip()
            if src not in node_map or dst not in node_map:
                continue
            filtered_edges.append({"from": src, "to": dst, "condition": str(edge.get("condition", "")).strip()})
            forward_adj[src].append(dst)

        reachable: set[str] = set()
        queue: list[str] = [root_id]
        while queue:
            nid = queue.pop(0)
            if nid in reachable:
                continue
            reachable.add(nid)
            for child in forward_adj.get(nid, []):
                if child not in reachable:
                    queue.append(child)

        historical_ids = self._historical_node_ids(plan=plan)
        keep = set(reachable) | {kid for kid in keep_ids if kid in node_map} | {hid for hid in historical_ids if hid in node_map}
        plan["nodes"] = [node_map[node_id] for node_id in node_map if node_id in keep]
        plan["edges"] = [edge for edge in filtered_edges if edge["from"] in keep and edge["to"] in keep]

    @staticmethod
    def _historical_node_ids(*, plan: dict[str, Any]) -> set[str]:
        markers = plan.get("step_markers") if isinstance(plan.get("step_markers"), dict) else {}
        historical: set[str] = set()
        for node_id, raw in markers.items():
            node_key = str(node_id).strip()
            if not node_key:
                continue
            if isinstance(raw, dict):
                state = str(raw.get("state", "pending")).strip().lower()
            else:
                state = str(raw or "pending").strip().lower()
            if state in {"done", "skipped", "blocked"}:
                historical.add(node_key)
        return historical

    def _infer_current_node_id(
        self,
        *,
        user_input: str,
        observed_tool: str,
        response_target: str,
        route: str,
        proposal: dict[str, Any],
        plan_update: dict[str, Any],
        previous_current: str,
    ) -> str:
        selected_next = str(plan_update.get("selected_next_node_id", "")).strip()
        if selected_next:
            return selected_next
        current_override = str(plan_update.get("current_node_id", "")).strip()
        if current_override:
            return current_override

        lowered = user_input.lower()
        proposal_intent = str(proposal.get("intent", "")).strip().lower() if isinstance(proposal, dict) else ""
        if self.strict_llm_mode:
            if proposal_intent == "conversation_termination" or self._is_conversation_termination(user_input):
                return "resolve_outcome"
            return str(previous_current or "").strip()
        if proposal_intent == "conversation_termination" or self._is_conversation_termination(user_input):
            return "resolve_outcome"
        if observed_tool in {"verify_dob", "verify_mobile"} or "verify" in lowered:
            return "verify_identity"
        if observed_tool in {"dues_explain_build", "loan_policy_lookup"} or any(
            token in lowered for token in ["dues", "overdue", "emi", "policy", "amount due"]
        ):
            return "explain_dues"
        if observed_tool in {"payment_link_create", "pay_by_phone_collect", "payment_status_check"} or any(
            token in lowered for token in ["pay now", "payment", "link", "settle"]
        ):
            return "resolve_outcome"
        if any(token in lowered for token in ["cannot pay", "hardship", "discount", "waiver", "restructure", "settlement"]):
            return "evaluate_assistance"
        if response_target == "self":
            return "evaluate_assistance"
        return "collect_payment_intent"

    @staticmethod
    def _next_nodes_from_edges(*, nodes: list[Any], current_node_id: str) -> list[str]:
        next_nodes: list[str] = []
        seen: set[str] = set()
        for edge in nodes:
            if not isinstance(edge, dict):
                continue
            if str(edge.get("from", "")) != current_node_id:
                continue
            to = str(edge.get("to", "")).strip()
            if to and to not in seen:
                next_nodes.append(to)
                seen.add(to)
        return next_nodes

    @staticmethod
    def _node_label(plan: dict[str, Any], node_id: str) -> str:
        for node in plan.get("nodes", []):
            if isinstance(node, dict) and str(node.get("id", "")) == node_id:
                return str(node.get("label", node_id)).strip() or node_id
        return node_id

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

    @staticmethod
    def _append_timeline_snapshot(*, plan: dict[str, Any], update: dict[str, Any]) -> None:
        timeline = plan.get("timeline")
        entries = list(timeline) if isinstance(timeline, list) else []
        timestamp = datetime.now(UTC).isoformat()
        entries.append(
            {
                "at_utc": timestamp,
                "version": int(plan.get("version", 1)),
                "status": str(plan.get("status", "active")),
                "current_node_id": str(plan.get("current_node_id", "")),
                "next_node_ids": list(plan.get("next_node_ids", [])) if isinstance(plan.get("next_node_ids"), list) else [],
                "update": update,
            }
        )
        plan["timeline"] = entries[-40:]

        # Persist full plan snapshots across turns so the UI can render a
        # conversation-level plan timeline (prev/next over historical plan states).
        snapshot_plan = {
            "plan_id": str(plan.get("plan_id", "")),
            "version": int(plan.get("version", 1)),
            "status": str(plan.get("status", "active")),
            "mode": str(plan.get("mode", "strict_collections")),
            "objective": str(plan.get("objective", "")),
            "root_node_id": str(plan.get("root_node_id", "")),
            "current_node_id": str(plan.get("current_node_id", "")),
            "previous_node_id": plan.get("previous_node_id"),
            "next_node_ids": list(plan.get("next_node_ids", [])) if isinstance(plan.get("next_node_ids"), list) else [],
            "nodes": [dict(node) for node in plan.get("nodes", []) if isinstance(node, dict)],
            "edges": [dict(edge) for edge in plan.get("edges", []) if isinstance(edge, dict)],
            "step_markers": dict(plan.get("step_markers", {})) if isinstance(plan.get("step_markers"), dict) else {},
            "updated_from": str(plan.get("updated_from", "")),
            "last_response_target": str(plan.get("last_response_target", "")),
        }
        snapshots_raw = plan.get("timeline_snapshots")
        snapshots = list(snapshots_raw) if isinstance(snapshots_raw, list) else []
        snapshots.append(
            {
                "at_utc": timestamp,
                "version": int(plan.get("version", 1)),
                "status": str(plan.get("status", "active")),
                "current_node_id": str(plan.get("current_node_id", "")),
                "update": dict(update) if isinstance(update, dict) else {},
                "plan": snapshot_plan,
            }
        )
        plan["timeline_snapshots"] = snapshots[-80:]

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
        if llm_payload is not None:
            return llm_payload
        if self.strict_llm_mode:
            return {
                "needs_discount_specialist": False,
                "is_plan_request": False,
                "is_plan_rejection": False,
                "hardship_signal": False,
                "hardship_reason": str(memory_state.get("hardship_reason", "income_reduction")),
                "suggested_plan_mode": mode,
                "reason": "strict_mode_no_classifier_fallback",
            }
        return {
            "needs_discount_specialist": self._needs_discount_specialist(user_input),
            "is_plan_request": self._is_plan_request(user_input),
            "is_plan_rejection": self._is_plan_rejection(user_input),
            "hardship_signal": any(token in user_input.lower() for token in ["cannot pay", "hardship", "vulnerability", "emi"]),
            "hardship_reason": str(memory_state.get("hardship_reason", "income_reduction")),
            "suggested_plan_mode": mode,
            "reason": "heuristic_fallback",
        }

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
            raise ValueError("Missing required prompt: plan_proposal.classifier_system_prompt")
        if not self.classifier_user_prompt.strip():
            raise ValueError("Missing required prompt: plan_proposal.classifier_user_prompt")

        vars_map = {
            "user_input": user_input,
            "mode": mode,
            "memory_state_json": self._json_compact(
                self._compact_memory_state_for_prompt(memory_state),
                max_chars=self.max_json_chars,
            ),
            "existing_plan_json": self._json_compact(
                self._compact_existing_plan_for_prompt(existing_plan),
                max_chars=self.max_json_chars,
            ),
        }
        system_prompt = self._render_prompt_template(self.classifier_system_prompt, vars_map)
        user_prompt = self._render_prompt_template(self.classifier_user_prompt, vars_map)
        if not self.last_debug.get("prompt"):
            self.last_debug["prompt"] = user_prompt
            self.last_debug["system_prompt"] = system_prompt or None
        try:
            payload = StructuredOutputRunner(self.llm, max_retries=2).run(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=_PlanSignalPayload,
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
            "reason": str(payload.reason or ""),
        }

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        value = str(text or "")
        if len(value) <= max_chars:
            return value
        return value[: max_chars - 3].rstrip() + "..."

    def _json_compact(self, value: Any, *, max_chars: int) -> str:
        raw = json.dumps(value, ensure_ascii=True, default=str, separators=(",", ":"))
        if len(raw) <= max_chars:
            return raw
        return self._truncate_text(raw, max_chars)

    @staticmethod
    def _compact_existing_plan_for_prompt(plan: dict[str, Any], *, minimal: bool = False) -> dict[str, Any]:
        if not isinstance(plan, dict):
            return {}
        nodes_raw = plan.get("nodes") if isinstance(plan.get("nodes"), list) else []
        edges_raw = plan.get("edges") if isinstance(plan.get("edges"), list) else []
        markers = plan.get("step_markers") if isinstance(plan.get("step_markers"), dict) else {}
        max_nodes = 6 if minimal else 10
        max_edges = 10 if minimal else 16
        nodes: list[dict[str, Any]] = []
        for node in nodes_raw[:max_nodes]:
            if not isinstance(node, dict):
                continue
            nodes.append(
                {
                    "id": str(node.get("id", "")).strip(),
                    "label": str(node.get("label", "")).strip(),
                    "status": str(node.get("status", "")).strip(),
                    "owner": str(node.get("owner", "")).strip(),
                }
            )
        edges: list[dict[str, Any]] = []
        for edge in edges_raw[:max_edges]:
            if not isinstance(edge, dict):
                continue
            edges.append(
                {
                    "from": str(edge.get("from", "")).strip(),
                    "to": str(edge.get("to", "")).strip(),
                    "condition": str(edge.get("condition", "")).strip(),
                }
            )
        marker_view: dict[str, str] = {}
        marker_limit = 8 if minimal else 14
        for key, raw in markers.items():
            if len(marker_view) >= marker_limit:
                break
            if not isinstance(raw, dict):
                continue
            marker_view[str(key)] = str(raw.get("state", "pending")).strip()
        return {
            "plan_id": str(plan.get("plan_id", "")).strip(),
            "version": int(plan.get("version", 1) or 1),
            "status": str(plan.get("status", "active")).strip(),
            "mode": str(plan.get("mode", "strict_collections")).strip(),
            "current_node_id": str(plan.get("current_node_id", "")).strip(),
            "previous_node_id": str(plan.get("previous_node_id", "")).strip(),
            "next_node_ids": [str(x).strip() for x in (plan.get("next_node_ids") or []) if str(x).strip()][:6],
            "nodes": nodes,
            "edges": edges,
            "step_markers": marker_view,
        }

    @staticmethod
    def _compact_memory_state_for_prompt(memory_state: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(memory_state, dict):
            return {}
        return {
            "active_case_id": str(memory_state.get("active_case_id", "")).strip(),
            "active_user_id": str(memory_state.get("active_user_id", "")).strip(),
            "active_customer_name": str(memory_state.get("active_customer_name", "")).strip(),
            "identity_verified": bool(memory_state.get("identity_verified", False)),
            "verification_entities": memory_state.get("verification_entities", {}),
            "verification_missing_fields": memory_state.get("verification_missing_fields", []),
            "mode": str(memory_state.get("mode", "strict_collections")).strip(),
            "last_response_target": str(memory_state.get("last_response_target", "")).strip(),
        }
