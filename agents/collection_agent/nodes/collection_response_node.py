"""Collection-specific response node with target routing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from agents.collection_agent.llm_structured import StructuredOutputRunner
from src.nodes.response_node import ResponseNode
from src.nodes.types import AgentState, NodeUpdate


class _ResponsePayload(BaseModel):
    message: str
    response_target: str = "customer"


class _ResponseDirectivePayload(BaseModel):
    conversation_objective: str = "close_conversation"
    dialogue_action: str = "ask_commitment_date"
    response_mode: str = "informational"
    required_response_elements: list[str] = Field(default_factory=list)
    forbidden_dialogue_actions: list[str] = Field(default_factory=list)
    allowed_dialogue_actions: list[str] = Field(default_factory=list)
    customer_facing_goal: str | None = None
    handoff_target: str | None = None


@dataclass(slots=True)
class CollectionResponseNode(ResponseNode):
    """Emits response text and a response target for next-hop routing.

    State Keys Read:
    - `user_input`
    - `response_target`
    - `plan_proposal`
    - `conversation_plan`
    - `observations`
    - `observation` (latest compatibility mirror)
    - `verification_*` keys (`identity_verified`, `verification_entities`, `verification_missing_fields`)
    - `extracted_entities`
    - `extracted_entities_turn`
    - `extracted_entity_descriptions`
    - `memory` (reads/writes `memory.state`, including conversation history and last response fields)

    State Keys Write:
    - `response`
    - `response_target`
    - `conversation_history`
    - `prompt`
    - `system_prompt`
    - `llm_response`
    - `llm_error`
    - `fallback_reason` (optional)
    """

    default_target: str = "customer"
    render_system_prompt: str = ""
    render_user_prompt: str = ""
    verification_opening_template: str = ""
    verification_followup_template: str = ""
    verification_default_missing_text: str = "your date of birth (YYYY-MM-DD) and your registered phone number"
    verification_hardship_prefix: str = "I am sorry to hear this, and I appreciate you sharing it. "
    verification_ack_template: str = "Thank you{customer_suffix}. "
    strict_llm_mode: bool = True
    max_prompt_chars: int = 4200
    max_json_chars: int = 800
    last_render_debug: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def execute(self, state: AgentState) -> NodeUpdate:
        self.last_render_debug = {
            "prompt": None,
            "system_prompt": None,
            "llm_response": None,
            "llm_error": None,
        }
        plan = state.get("plan_proposal") if isinstance(state.get("plan_proposal"), dict) else {}
        if plan:
            update: NodeUpdate = {"response": self._render_from_proposal(state=state, proposal=plan)}
            plan_target = str(plan.get("target", "")).strip().lower()
            if plan_target:
                update["response_target"] = plan_target
        else:
            update = ResponseNode.execute(self, state)

        target = str(update.get("response_target", state.get("response_target", self.default_target))).strip().lower()
        if target not in {"customer", "self", "discount_planning_agent"}:
            target = self.default_target
        update["response_target"] = target
        update["prompt"] = self.last_render_debug.get("prompt")
        update["system_prompt"] = self.last_render_debug.get("system_prompt")
        update["llm_response"] = self.last_render_debug.get("llm_response")
        update["llm_error"] = self.last_render_debug.get("llm_error")
        if self.last_render_debug.get("fallback_reason"):
            update["fallback_reason"] = self.last_render_debug.get("fallback_reason")
        self._update_conversation_history_memory(state=state, update=update)
        return update

    def _update_conversation_history_memory(self, *, state: AgentState, update: NodeUpdate) -> None:
        memory = state.get("memory")
        if memory is None:
            return
        memory_state = dict(getattr(memory, "state", {}))
        history = list(memory_state.get("conversation_history", [])) if isinstance(memory_state.get("conversation_history"), list) else []

        user_text = str(state.get("user_input", "")).strip()
        source = str(state.get("message_source", "customer")).strip().lower() or "customer"
        if user_text and source not in {"self", "system"}:
            if not history or not (
                isinstance(history[-1], dict)
                and str(history[-1].get("role", "")).strip().lower() == source
                and str(history[-1].get("content", "")).strip() == user_text
            ):
                history.append({"role": source, "content": user_text})

        response_text = str(update.get("response", "")).strip()
        response_target = str(update.get("response_target", "customer")).strip().lower()
        if response_text and response_target == "customer":
            if not history or not (
                isinstance(history[-1], dict)
                and str(history[-1].get("role", "")).strip().lower() == "agent"
                and str(history[-1].get("content", "")).strip() == response_text
            ):
                history.append({"role": "agent", "content": response_text})

        # Keep bounded in memory.
        history = history[-40:]
        memory.set_state(conversation_history=history)
        update["conversation_history"] = history

    def route(self, state: AgentState) -> str:
        target = str(state.get("response_target", self.default_target)).strip().lower()
        if target not in {"customer", "self", "discount_planning_agent"}:
            return self.default_target
        return target

    def _render_from_proposal(self, *, state: AgentState, proposal: dict[str, Any]) -> str:
        render_context = self._resolve_render_context(state=state, proposal=proposal)
        directive = self._resolve_response_directive(
            state=state,
            proposal=proposal,
            context=render_context,
        )
        response_target = str(render_context.get("response_target", "customer")).strip().lower() or "customer"
        if response_target == "discount_planning_agent":
            return self._fallback_from_directive(
                directive=directive,
                context=render_context,
                response_target=response_target,
            )

        if self.llm is not None:
            llm_response = self._llm_render_from_proposal(
                state=state,
                proposal=proposal,
                render_context=render_context,
                response_directive=directive,
            )
            if llm_response:
                validated = self._validate_response_against_directive(
                    text=llm_response,
                    directive=directive,
                    context=render_context,
                )
                if validated:
                    return self._apply_minimal_safety_cleanup(
                        text=validated,
                        context=render_context,
                        directive=directive,
                    )
                self.last_render_debug["fallback_reason"] = "directive_validation_failed"
            if self.strict_llm_mode:
                fallback_reason = str(self.last_render_debug.get("fallback_reason", "")).strip()
                if fallback_reason == "provider_rate_limit":
                    raise RuntimeError(
                        "CollectionResponseNode rate-limited by provider while strict_llm_mode is enabled. "
                        f"Underlying error: {self.last_render_debug.get('llm_error', 'unknown')}"
                    )
        return self._fallback_from_directive(
            directive=directive,
            context=render_context,
            response_target=response_target,
        )

    def _resolve_render_context(self, *, state: AgentState, proposal: dict[str, Any]) -> dict[str, Any]:
        memory = state.get("memory")
        memory_state = dict(getattr(memory, "state", {})) if memory is not None else {}
        user_input = str(state.get("user_input", ""))
        response_target = str(proposal.get("target", state.get("response_target", "customer"))).strip().lower() or "customer"
        conversation_plan = (
            proposal.get("conversation_plan")
            if isinstance(proposal.get("conversation_plan"), dict)
            else (state.get("conversation_plan") if isinstance(state.get("conversation_plan"), dict) else {})
        )
        facts = self._resolve_case_facts(state=state, proposal=proposal)
        current_plan_node_id = (
            str(conversation_plan.get("current_node_id", "")).strip()
            if isinstance(conversation_plan, dict)
            else ""
        )
        verification_guard_context = self._build_verification_guard_context(
            state=state,
            memory_state=memory_state,
            response_target=response_target,
            conversation_plan=conversation_plan,
            customer_name=str(facts.get("customer_name", "Customer")).strip() or "Customer",
            user_input=user_input,
        )
        return {
            "memory_state": memory_state,
            "user_input": user_input,
            "response_target": response_target,
            "conversation_plan": conversation_plan,
            "facts": facts,
            "current_plan_node_id": current_plan_node_id,
            "verification_guard_context": verification_guard_context,
            "conversation_mode": str(state.get("conversation_mode", memory_state.get("conversation_mode", "collections"))).strip(),
            "negotiation_stage": str(state.get("negotiation_stage", memory_state.get("negotiation_stage", "none"))).strip(),
            "customer_payment_posture": str(
                state.get("customer_payment_posture", memory_state.get("customer_payment_posture", "unknown"))
            ).strip(),
            "hardship_context": (
                state.get("hardship_context")
                if isinstance(state.get("hardship_context"), dict)
                else (
                    memory_state.get("hardship_context")
                    if isinstance(memory_state.get("hardship_context"), dict)
                    else {}
                )
            ),
            "response_mode": str(state.get("response_mode", memory_state.get("response_mode", "informational"))).strip(),
            "active_dialogue_owner": str(
                state.get("active_dialogue_owner", memory_state.get("active_dialogue_owner", "collections"))
            ).strip(),
            "observations": state.get("observations") if isinstance(state.get("observations"), list) else [],
            "observation": state.get("observation"),
            "plan_proposal": proposal,
        }

    def _resolve_response_directive(
        self,
        *,
        state: AgentState,
        proposal: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        memory_state = context.get("memory_state") if isinstance(context.get("memory_state"), dict) else {}
        raw_directive = proposal.get("response_directive")
        if not isinstance(raw_directive, dict):
            raw_directive = {
                "conversation_objective": proposal.get("conversation_objective"),
                "dialogue_action": proposal.get("dialogue_action"),
                "response_mode": proposal.get("response_mode"),
                "required_response_elements": proposal.get("required_response_elements"),
                "forbidden_dialogue_actions": proposal.get("forbidden_dialogue_actions"),
                "allowed_dialogue_actions": proposal.get("allowed_dialogue_actions"),
                "customer_facing_goal": proposal.get("customer_facing_goal"),
                "handoff_target": proposal.get("handoff_target"),
            }
        raw_directive = {key: value for key, value in raw_directive.items() if value is not None}
        if not raw_directive:
            directive = self._build_response_directive_from_context(
                state=state,
                memory_state=memory_state,
                response_target=str(context.get("response_target", "customer")).strip().lower() or "customer",
            )
            if str(context.get("response_target", "")).strip().lower() == "discount_planning_agent":
                directive["conversation_objective"] = "handoff_to_offer_agent"
                directive["dialogue_action"] = "handoff"
                directive["response_mode"] = "firm"
                directive["handoff_target"] = "discount_planning_agent"
            return directive
        try:
            payload = _ResponseDirectivePayload.model_validate(raw_directive)
            directive = payload.model_dump(mode="json")
        except Exception:
            directive = self._build_response_directive_from_context(
                state=state,
                memory_state=memory_state,
                response_target=str(context.get("response_target", "customer")).strip().lower() or "customer",
            )
        if not directive.get("conversation_objective"):
            directive = self._build_response_directive_from_context(
                state=state,
                memory_state=memory_state,
                response_target=str(context.get("response_target", "customer")).strip().lower() or "customer",
            )
        if str(context.get("response_target", "")).strip().lower() == "discount_planning_agent":
            directive["conversation_objective"] = "handoff_to_offer_agent"
            directive["dialogue_action"] = "handoff"
            directive["response_mode"] = "firm"
            directive["handoff_target"] = "discount_planning_agent"
        directive.setdefault("required_response_elements", [])
        directive.setdefault("forbidden_dialogue_actions", [])
        directive.setdefault("allowed_dialogue_actions", [])
        directive.setdefault("customer_facing_goal", None)
        directive.setdefault("handoff_target", None)
        return directive

    def _build_response_directive_from_context(
        self,
        *,
        state: AgentState,
        memory_state: dict[str, Any],
        response_target: str,
    ) -> dict[str, Any]:
        objective, action, mode = self._infer_response_objective(
            memory_state=memory_state,
            response_target=response_target,
        )
        if response_target == "discount_planning_agent":
            objective = "handoff_to_offer_agent"
            action = "handoff"
            mode = "firm"
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
        del state
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

    def _llm_render_from_proposal(
        self,
        *,
        state: AgentState,
        proposal: dict[str, Any],
        render_context: dict[str, Any] | None = None,
        response_directive: dict[str, Any] | None = None,
    ) -> str | None:
        context = dict(render_context) if isinstance(render_context, dict) else self._resolve_render_context(state=state, proposal=proposal)
        memory_state = dict(context.get("memory_state", {})) if isinstance(context.get("memory_state"), dict) else {}
        user_input = str(context.get("user_input", ""))
        observation = context.get("observation")
        if observation is None:
            observations = context.get("observations")
            if isinstance(observations, list):
                for item in reversed(observations):
                    if isinstance(item, dict):
                        observation = item
                        break
        response_target = str(context.get("response_target", "customer")).strip().lower() or "customer"
        conversation_plan = context.get("conversation_plan") if isinstance(context.get("conversation_plan"), dict) else {}
        current_plan_node_id = str(context.get("current_plan_node_id", "")).strip()
        current_plan_node_label = self._resolve_plan_node_label(conversation_plan, current_plan_node_id)
        facts = context.get("facts") if isinstance(context.get("facts"), dict) else self._resolve_case_facts(state=state, proposal=proposal)
        customer_name = str(facts.get("customer_name", "Customer"))
        case_id = str(facts.get("case_id", "COLL-1001"))
        overdue_amount = float(facts.get("overdue_amount", 0.0) or 0.0)
        prior_agent_response = str(memory_state.get("last_agent_response", "")).strip()
        turn_index = int(memory_state.get("turn_index", state.get("turn_index", 0)) or 0)
        is_opening_turn = turn_index <= 0
        conversation_mode = str(context.get("conversation_mode", memory_state.get("conversation_mode", "collections"))).strip()
        negotiation_stage = str(context.get("negotiation_stage", memory_state.get("negotiation_stage", "none"))).strip()
        customer_payment_posture = str(
            context.get("customer_payment_posture", memory_state.get("customer_payment_posture", "unknown"))
        ).strip()
        hardship_context = (
            context.get("hardship_context")
            if isinstance(context.get("hardship_context"), dict)
            else (
                memory_state.get("hardship_context")
                if isinstance(memory_state.get("hardship_context"), dict)
                else {}
            )
        )
        response_mode = str(context.get("response_mode", memory_state.get("response_mode", "informational"))).strip()
        active_dialogue_owner = str(
            context.get("active_dialogue_owner", memory_state.get("active_dialogue_owner", "collections"))
        ).strip()
        verification_context_merged = self._build_verification_context(state=state, memory_state=memory_state, proposal=proposal)
        response_directive = (
            dict(response_directive)
            if isinstance(response_directive, dict)
            else self._resolve_response_directive(state=state, proposal=proposal, context=context)
        )
        extracted_entities = context.get("extracted_entities")
        if not isinstance(extracted_entities, dict):
            extracted_entities = memory_state.get("extracted_entities", {}) if isinstance(memory_state.get("extracted_entities"), dict) else {}
        extracted_entities_turn = context.get("extracted_entities_turn")
        if not isinstance(extracted_entities_turn, dict):
            extracted_entities_turn = (
                memory_state.get("extracted_entities_turn", {})
                if isinstance(memory_state.get("extracted_entities_turn"), dict)
                else {}
            )
        extracted_entity_descriptions = context.get("extracted_entity_descriptions")
        if not isinstance(extracted_entity_descriptions, dict):
            extracted_entity_descriptions = (
                memory_state.get("extracted_entity_descriptions", {})
                if isinstance(memory_state.get("extracted_entity_descriptions"), dict)
                else {}
            )
        verification_entities = verification_context_merged.get("verification_entities", {})
        verification_missing_fields = verification_context_merged.get("verification_missing_fields", [])
        required_fields = verification_context_merged.get("required_fields", [])
        compact_plan = self._compact_conversation_plan(conversation_plan)
        compact_proposal = self._compact_plan_proposal(proposal=proposal)
        compact_observation = self._compact_observation(observation)
        prior_agent_response_short = self._truncate_text(prior_agent_response, 280)

        system_prompt = (f"{self.system_prompt or ''}\n{self.render_system_prompt or ''}").strip()
        user_prompt = self._render_template(
            self.render_user_prompt,
            {
                "user_input": user_input,
                "customer_name": customer_name,
                "case_id": case_id,
                "overdue_amount": f"{overdue_amount:.2f}",
                "response_target": response_target,
                "is_opening_turn_json": json.dumps(is_opening_turn),
                "prior_agent_response": prior_agent_response_short,
                "current_plan_node_id": current_plan_node_id,
                "current_plan_node_label": current_plan_node_label,
                "plan_proposal_json": self._json_compact(compact_proposal, max_chars=self.max_json_chars),
                "conversation_plan_json": self._json_compact(compact_plan, max_chars=self.max_json_chars),
                "verification_context_json": self._json_compact(verification_context_merged, max_chars=800),
                "extracted_entities_json": self._json_compact(extracted_entities, max_chars=500),
                "extracted_entities_turn_json": self._json_compact(extracted_entities_turn, max_chars=500),
                "extracted_entity_descriptions_json": self._json_compact(extracted_entity_descriptions, max_chars=500),
                "verification_entities_json": self._json_compact(verification_entities, max_chars=500),
                "verification_missing_fields_json": self._json_compact(verification_missing_fields, max_chars=300),
                "observation_json": self._json_compact(compact_observation, max_chars=700),
                "conversation_mode": conversation_mode,
                "negotiation_stage": negotiation_stage,
                "customer_payment_posture": customer_payment_posture,
                "hardship_context_json": self._json_compact(hardship_context, max_chars=400),
                "response_mode": response_mode,
                "active_dialogue_owner": active_dialogue_owner,
                "response_directive_json": self._json_compact(response_directive, max_chars=800),
                "conversation_objective": str(response_directive.get("conversation_objective", "")).strip(),
                "dialogue_action": str(response_directive.get("dialogue_action", "")).strip(),
                "required_response_elements_json": self._json_compact(
                    response_directive.get("required_response_elements", []),
                    max_chars=300,
                ),
                "forbidden_dialogue_actions_json": self._json_compact(
                    response_directive.get("forbidden_dialogue_actions", []),
                    max_chars=300,
                ),
            },
        )
        if len(user_prompt) > self.max_prompt_chars:
            # Second-stage clamp for strict provider token windows.
            user_prompt = self._truncate_text(user_prompt, self.max_prompt_chars)
        self.last_render_debug = {
            "prompt": user_prompt,
            "system_prompt": system_prompt or None,
            "llm_response": None,
            "llm_error": None,
        }
        try:
            payload = StructuredOutputRunner(self.llm, max_retries=4).run(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=_ResponsePayload,
            )
            self.last_render_debug["llm_response"] = payload.model_dump(mode="json")
        except Exception as exc:
            err_text = str(exc)
            self.last_render_debug["llm_error"] = err_text
            if self._is_provider_rate_limit_error(err_text):
                self.last_render_debug["fallback_reason"] = "provider_rate_limit"
                return None
            return None
        response = str(payload.message).strip()
        response_target_payload = str(payload.response_target).strip().lower()
        if response_target_payload in {"customer", "self"}:
            proposal["target"] = response_target_payload
        if not response:
            return None
        return response

    def _build_verification_context(
        self,
        *,
        state: AgentState,
        memory_state: dict[str, Any],
        proposal: dict[str, Any],
    ) -> dict[str, Any]:
        verification_entities = state.get("verification_entities")
        if not isinstance(verification_entities, dict):
            verification_entities = (
                memory_state.get("verification_entities", {})
                if isinstance(memory_state.get("verification_entities"), dict)
                else {}
            )
        required = memory_state.get("active_verification_required_fields")
        required_fields = [str(x).strip().lower() for x in required if str(x).strip()] if isinstance(required, list) else []
        if not required_fields:
            required_fields = ["dob", "phone"]
        missing_raw = state.get("verification_missing_fields")
        if not isinstance(missing_raw, list):
            missing_raw = memory_state.get("verification_missing_fields")
        missing_fields = (
            [str(x).strip().lower() for x in missing_raw if str(x).strip()]
            if isinstance(missing_raw, list)
            else [field for field in required_fields if not str(verification_entities.get(field, "")).strip()]
        )
        verified_raw = state.get("verification_verified_fields")
        if not isinstance(verified_raw, list):
            verified_raw = memory_state.get("verification_verified_fields")
        verified_fields = (
            [str(x).strip().lower() for x in verified_raw if str(x).strip()]
            if isinstance(verified_raw, list)
            else [field for field in required_fields if field not in missing_fields]
        )
        identity_verified = bool(state.get("identity_verified", memory_state.get("identity_verified", False)))
        verification_incomplete = (not identity_verified) or bool(missing_fields)
        return {
            "identity_verified": identity_verified,
            "required_fields": required_fields,
            "verification_entities": verification_entities,
            "verification_missing_fields": missing_fields,
            "verification_verified_fields": verified_fields,
            "verification_incomplete": verification_incomplete,
            "current_plan_node_id": str(
                (
                    proposal.get("conversation_plan", {}).get("current_node_id")
                    if isinstance(proposal.get("conversation_plan"), dict)
                    else ""
                )
                or ""
            ).strip(),
        }

    def _build_verification_guard_context(
        self,
        *,
        state: AgentState,
        memory_state: dict[str, Any],
        response_target: str,
        conversation_plan: dict[str, Any],
        customer_name: str,
        user_input: str,
    ) -> dict[str, Any] | None:
        if response_target != "customer":
            return None
        if bool(state.get("identity_verified", memory_state.get("identity_verified", False))):
            return None

        collected = (
            state.get("verification_entities")
            if isinstance(state.get("verification_entities"), dict)
            else (
                memory_state.get("verification_collected")
                if isinstance(memory_state.get("verification_collected"), dict)
                else {}
            )
        )
        required = memory_state.get("active_verification_required_fields")
        required_fields = [str(x).strip() for x in required if str(x).strip()] if isinstance(required, list) else []
        # Safety clamp: verification scope for this agent must stay within supported identity fields.
        allowed_identity_fields = {"dob", "phone", "last4_pan", "zip", "name"}
        required_fields = [field for field in required_fields if field in allowed_identity_fields]
        missing_from_state_raw = state.get("verification_missing_fields")
        if not isinstance(missing_from_state_raw, list):
            missing_from_state_raw = memory_state.get("verification_missing_fields")
        missing_from_state = (
            [str(x).strip() for x in missing_from_state_raw if str(x).strip()]
            if isinstance(missing_from_state_raw, list)
            else []
        )

        missing_labels: list[str] = []
        name_confirmed = bool(collected.get("name_confirmed"))
        # Prefer tool-driven missing fields from state (source of truth under iterative verify_dob/verify_mobile flow).
        if missing_from_state:
            for field in missing_from_state:
                if field in {"name", "name_confirmed"}:
                    continue
                missing_labels.append(self._verification_field_label(field))
        else:
            for field in required_fields:
                if field in {"name", "name_confirmed"}:
                    continue
                if collected.get(field):
                    continue
                missing_labels.append(self._verification_field_label(field))
        mismatched_raw = (
            memory_state.get("verification_mismatched_fields")
            if isinstance(memory_state.get("verification_mismatched_fields"), list)
            else []
        )
        mismatched_fields = [str(x).strip() for x in mismatched_raw if str(x).strip()]
        mismatch_labels = [self._verification_field_label(field) for field in mismatched_fields]
        mismatch_mode = False
        if not missing_labels and mismatch_labels:
            missing_labels = mismatch_labels
            mismatch_mode = True

        hardship_context = (
            state.get("hardship_context")
            if isinstance(state.get("hardship_context"), dict)
            else (
                memory_state.get("hardship_context")
                if isinstance(memory_state.get("hardship_context"), dict)
                else {}
            )
        )
        lowered_input = str(user_input or "").lower()
        hardship_tokens = (
            "job loss",
            "lost my job",
            "lost job",
            "cannot pay",
            "can't pay",
            "hardship",
            "vulnerable",
            "medical",
            "salary cut",
            "income loss",
        )
        hardship_signal = bool(hardship_context.get("hardship_detected", False)) or any(
            token in lowered_input for token in hardship_tokens
        )
        turn_index = int(memory_state.get("turn_index", state.get("turn_index", 0)) or 0)
        is_opening_turn = turn_index <= 0

        return {
            "verification_incomplete": True,
            "is_opening_turn": is_opening_turn,
            "name_confirmed": bool(name_confirmed),
            "customer_name": customer_name,
            "required_fields": required_fields,
            "missing_field_labels": missing_labels,
            "missing_fields_human": self._join_human_list(missing_labels) if missing_labels else "",
            "mismatch_mode": mismatch_mode,
            "hardship_signal": hardship_signal,
            "guidance": (
                "Ask only for missing verification fields in a natural conversational way. "
                "Do not disclose dues or policy-sensitive details until verification completes."
            ),
            "current_plan_node_id": str(conversation_plan.get("current_node_id", "")).strip(),
            "collected_fields": collected,
        }

    def _render_verification_first_message(self, *, customer_name: str, guard: dict[str, Any]) -> str:
        missing_labels = guard.get("missing_field_labels") if isinstance(guard.get("missing_field_labels"), list) else []
        missing_human = str(guard.get("missing_fields_human", "")).strip()
        hardship_signal = bool(guard.get("hardship_signal", False))
        is_opening_turn = bool(guard.get("is_opening_turn", False))
        hardship_prefix = self.verification_hardship_prefix if hardship_signal else ""
        customer_suffix = f", {customer_name}" if customer_name else ""
        ack_prefix = self._render_template(
            self.verification_ack_template,
            {"customer_suffix": customer_suffix},
        ).strip()
        if ack_prefix:
            ack_prefix = f"{ack_prefix} "
        if missing_human:
            ask_target = missing_human
        elif missing_labels:
            ask_target = self._join_human_list([str(x) for x in missing_labels if str(x).strip()])
        else:
            ask_target = self.verification_default_missing_text

        template = self.verification_opening_template if is_opening_turn else self.verification_followup_template
        return self._render_template(
            template,
            {
                "customer_name": customer_name,
                "hardship_prefix": hardship_prefix,
                "missing_human": ask_target,
                "ack_prefix": ack_prefix,
            },
        ).strip()

    def _apply_customer_continuity_policy(
        self,
        *,
        text: str,
        state: AgentState,
        proposal: dict[str, Any],
        response_target: str,
    ) -> str:
        if str(response_target).strip().lower() != "customer":
            return text

        context = self._resolve_render_context(state=state, proposal=proposal)
        directive = self._resolve_response_directive(state=state, proposal=proposal, context=context)
        rendered = self._apply_minimal_safety_cleanup(text=str(text or ""), context=context, directive=directive)
        validated = self._validate_response_against_directive(text=rendered, directive=directive, context=context)
        if validated:
            return validated
        return self._fallback_from_directive(
            directive=directive,
            context=context,
            response_target=str(response_target).strip().lower() or "customer",
        )

    @staticmethod
    def _resolve_negotiation_render_context(*, state: AgentState, proposal: dict[str, Any]) -> dict[str, Any]:
        memory = state.get("memory")
        memory_state = dict(getattr(memory, "state", {})) if memory is not None else {}
        proposal_context = proposal.get("context") if isinstance(proposal.get("context"), dict) else {}
        resolved = {
            "customer_name": str(
                proposal_context.get("customer_name", memory_state.get("active_customer_name", "Customer"))
            ).strip()
            or "Customer",
            "conversation_mode": str(
                state.get(
                    "conversation_mode",
                    proposal_context.get("conversation_mode", memory_state.get("conversation_mode", "collections")),
                )
            ).strip().lower(),
            "negotiation_stage": str(
                state.get(
                    "negotiation_stage",
                    proposal_context.get("negotiation_stage", memory_state.get("negotiation_stage", "none")),
                )
            ).strip().lower(),
            "customer_payment_posture": str(
                state.get(
                    "customer_payment_posture",
                    proposal_context.get("customer_payment_posture", memory_state.get("customer_payment_posture", "unknown")),
                )
            ).strip().lower(),
            "hardship_context": (
                state.get("hardship_context")
                if isinstance(state.get("hardship_context"), dict)
                else (
                    proposal_context.get("hardship_context")
                    if isinstance(proposal_context.get("hardship_context"), dict)
                    else (
                        memory_state.get("hardship_context")
                        if isinstance(memory_state.get("hardship_context"), dict)
                        else {}
                    )
                )
            ),
            "response_mode": str(
                state.get("response_mode", proposal_context.get("response_mode", memory_state.get("response_mode", "informational")))
            ).strip().lower(),
            "active_dialogue_owner": str(
                state.get(
                    "active_dialogue_owner",
                    proposal_context.get("active_dialogue_owner", memory_state.get("active_dialogue_owner", "collections")),
                )
            ).strip().lower(),
        }
        return resolved

    @staticmethod
    def _strip_orchestration_leakage(text: str) -> str:
        cleaned = str(text or "").strip()
        leak_patterns = [
            r"(?i)\bplease wait while i evaluate\b[^\.\!\?]*[\.\!\?]?\s*",
            r"(?i)\bi am processing internal steps\b[^\.\!\?]*[\.\!\?]?\s*",
            r"(?i)\bchecking backend systems\b[^\.\!\?]*[\.\!\?]?\s*",
            r"(?i)\binternal processing\b[^\.\!\?]*[\.\!\?]?\s*",
            r"(?i)\bcontinue internal planning using latest context and determine next execution step\b[^\.\!\?]*[\.\!\?]?\s*",
        ]
        for pattern in leak_patterns:
            cleaned = re.sub(pattern, "", cleaned).strip()
        return cleaned

    def _requires_negotiation_continuity_rewrite(self, text: str, *, context: dict[str, Any]) -> bool:
        conversation_mode = str(context.get("conversation_mode", "collections")).strip().lower()
        payment_posture = str(context.get("customer_payment_posture", "unknown")).strip().lower()
        hardship_context = context.get("hardship_context") if isinstance(context.get("hardship_context"), dict) else {}
        active_dialogue_owner = str(context.get("active_dialogue_owner", "collections")).strip().lower()
        hardship_active = conversation_mode == "hardship_negotiation" or bool(hardship_context.get("hardship_detected", False))
        arrangement_active = payment_posture == "needs_arrangement"
        if not hardship_active and not arrangement_active:
            return False
        if active_dialogue_owner not in {"plan_proposal", "promise_capture", "collections"}:
            return False
        lowered = str(text or "").lower()
        return (
            not lowered
            or self._contains_internal_processing(text)
            or self._looks_like_root_menu(text)
        )

    def _render_negotiation_stage_followup(self, *, customer_name: str, context: dict[str, Any]) -> str:
        del customer_name
        directive = self._resolve_response_directive(
            state=context.get("state") if isinstance(context.get("state"), dict) else {},
            proposal=context.get("proposal") if isinstance(context.get("proposal"), dict) else {},
            context=context,
        )
        return self._fallback_from_directive(
            directive=directive,
            context=context,
            response_target="customer",
        )

    def _fallback_render_from_proposal(self, *, proposal: dict[str, Any]) -> str:
        context = proposal.get("context") if isinstance(proposal.get("context"), dict) else {}
        memory_state = {
            "active_customer_name": context.get("customer_name", "Customer"),
            "active_case_id": context.get("case_id", "COLL-1001"),
            "active_overdue_amount": context.get("overdue_amount", 0.0),
            "conversation_mode": context.get("conversation_mode", "collections"),
            "negotiation_stage": context.get("negotiation_stage", "none"),
            "customer_payment_posture": context.get("customer_payment_posture", "unknown"),
            "hardship_context": context.get("hardship_context", {}),
            "response_mode": context.get("response_mode", "informational"),
            "active_dialogue_owner": context.get("active_dialogue_owner", "collections"),
            "identity_verified": context.get("identity_verified", False),
            "active_verification_required_fields": context.get("required_fields", ["dob", "phone"]),
            "verification_missing_fields": context.get("verification_missing_fields", []),
            "verification_verified_fields": context.get("verification_verified_fields", []),
            "verification_entities": context.get("verification_entities", {}),
        }
        facts = {
            "customer_name": str(memory_state.get("active_customer_name", "Customer")).strip() or "Customer",
            "case_id": str(memory_state.get("active_case_id", "COLL-1001")).strip() or "COLL-1001",
            "overdue_amount": float(memory_state.get("active_overdue_amount", 0.0) or 0.0),
        }
        verification_context = {
            "identity_verified": bool(memory_state.get("identity_verified", False)),
            "required_fields": list(memory_state.get("active_verification_required_fields", []))
            if isinstance(memory_state.get("active_verification_required_fields"), list)
            else ["dob", "phone"],
            "verification_entities": dict(memory_state.get("verification_entities", {})),
            "verification_missing_fields": list(memory_state.get("verification_missing_fields", []))
            if isinstance(memory_state.get("verification_missing_fields"), list)
            else [],
        }
        directive = self._resolve_response_directive(
            state={"memory": None},
            proposal=proposal,
            context={
                "memory_state": memory_state,
                "response_target": str(proposal.get("target", "customer")).strip().lower() or "customer",
                "conversation_mode": str(context.get("conversation_mode", "collections")).strip(),
                "negotiation_stage": str(context.get("negotiation_stage", "none")).strip(),
                "customer_payment_posture": str(context.get("customer_payment_posture", "unknown")).strip(),
                "hardship_context": context.get("hardship_context", {}),
                "response_mode": str(context.get("response_mode", "informational")).strip(),
                "active_dialogue_owner": str(context.get("active_dialogue_owner", "collections")).strip(),
            },
        )
        return self._fallback_from_directive(
            directive=directive,
            context={
                "memory_state": memory_state,
                "facts": facts,
                "verification_context": verification_context,
                "response_target": str(proposal.get("target", "customer")).strip().lower() or "customer",
            },
            response_target=str(proposal.get("target", "customer")).strip().lower() or "customer",
        )

    def _fallback_from_directive(
        self,
        *,
        directive: dict[str, Any],
        context: dict[str, Any],
        response_target: str,
    ) -> str:
        objective = str(directive.get("conversation_objective", "close_conversation")).strip().lower()
        response_target = str(response_target).strip().lower() or "customer"
        if response_target == "self":
            return "Continue internal planning using the current directive and determine the next execution step."
        if response_target == "discount_planning_agent":
            return "Prepare specialist handoff payload and wait for discount recommendation."

        facts = context.get("facts") if isinstance(context.get("facts"), dict) else {}
        verification_context = (
            context.get("verification_context")
            if isinstance(context.get("verification_context"), dict)
            else {}
        )
        memory_state = context.get("memory_state") if isinstance(context.get("memory_state"), dict) else {}
        customer_name = str(facts.get("customer_name", memory_state.get("active_customer_name", "Customer"))).strip() or "Customer"
        case_id = str(facts.get("case_id", memory_state.get("active_case_id", "COLL-1001"))).strip() or "COLL-1001"
        overdue_amount = float(facts.get("overdue_amount", memory_state.get("active_overdue_amount", 0.0)) or 0.0)
        missing_fields = verification_context.get("missing_fields_human")
        if not missing_fields:
            labels = verification_context.get("missing_field_labels")
            if isinstance(labels, list):
                missing_fields = self._join_human_list([str(x) for x in labels if str(x).strip()])
        missing_fields = str(missing_fields or self.verification_default_missing_text).strip()
        response_mode = str(directive.get("response_mode", context.get("response_mode", "informational"))).strip().lower()
        hardship_context = context.get("hardship_context") if isinstance(context.get("hardship_context"), dict) else {}
        hardship_active = bool(hardship_context.get("hardship_detected", False)) or str(context.get("conversation_mode", "")).strip().lower() == "hardship_negotiation"
        negotiation_stage = str(context.get("negotiation_stage", "none")).strip().lower()

        if objective == "collect_verification":
            prefix = self.verification_hardship_prefix if hardship_active else ""
            return (
                f"Hello {customer_name}, this is Alex from the bank's collections team. "
                f"{prefix}Before I share details, please confirm {missing_fields}."
            ).strip()
        if objective == "explain_dues":
            return (
                f"Thank you {customer_name}. Your overdue amount is INR {overdue_amount:.2f}. "
                "What would you like to do next to bring the account current?"
            ).strip()
        if objective == "assess_affordability":
            if response_mode == "empathetic" or hardship_active:
                return (
                    f"I am sorry to hear that, {customer_name}. "
                    "To explore a manageable arrangement, what monthly amount would realistically work for you right now?"
                ).strip()
            return "To explore a manageable arrangement, what monthly amount would realistically work for you right now?"
        if objective == "present_arrangement_options":
            return "Let us work toward a practical repayment option. What installment amount would you be comfortable committing to?"
        if objective == "negotiate_installment":
            return "What installment amount would feel manageable for you at the moment?"
        if objective == "confirm_commitment":
            return "Thank you. What amount and payment date can you confidently commit to for the next step?"
        if objective == "capture_promise":
            return "Please confirm the amount and payment date you can commit to so I can capture your promise."
        if objective == "handoff_to_offer_agent":
            return "Prepare specialist handoff payload and wait for discount recommendation."
        if objective == "close_conversation":
            return "Thank you. I am closing this conversation now."

        if hardship_active and negotiation_stage in {"discovering_hardship", "assessing_capacity"}:
            return "What monthly amount would realistically work for you right now?"
        if hardship_active and negotiation_stage in {"evaluating_options", "negotiating_plan", "awaiting_customer_decision"}:
            return "What installment amount would you be comfortable committing to?"
        return "Please confirm how you would like to proceed with your dues."

    def _validate_response_against_directive(
        self,
        *,
        text: str,
        directive: dict[str, Any],
        context: dict[str, Any],
    ) -> str | None:
        rendered = self._apply_minimal_safety_cleanup(text=text, context=context, directive=directive)
        if not rendered:
            return None
        forbidden = {
            str(item).strip().lower()
            for item in (directive.get("forbidden_dialogue_actions") or [])
            if str(item).strip()
        }
        verification_context = (
            context.get("verification_context")
            if isinstance(context.get("verification_context"), dict)
            else {}
        )
        identity_verified = bool(verification_context.get("identity_verified", False))
        if "mention_internal_processing" in forbidden and self._contains_internal_processing(rendered):
            return None
        if "restart_collections_menu" in forbidden and self._looks_like_root_menu(rendered):
            return None
        if "ask_pay_now_or_arrangement" in forbidden and self._looks_like_root_menu(rendered):
            return None
        if "disclose_dues_before_verification" in forbidden and not identity_verified:
            lowered = rendered.lower()
            if any(token in lowered for token in ["inr ", "overdue", "dues", "amount", "emi", "late fee"]):
                return None

        required = [
            str(item).strip().lower()
            for item in (directive.get("required_response_elements") or [])
            if str(item).strip()
        ]
        if required and not self._meets_required_response_elements(rendered, required=required, context=context):
            return None
        return rendered

    def _apply_minimal_safety_cleanup(
        self,
        *,
        text: str,
        context: dict[str, Any],
        directive: dict[str, Any],
    ) -> str:
        del context, directive
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"```", "", cleaned)
        cleaned = self._strip_orchestration_leakage(cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def _contains_internal_processing(text: str) -> bool:
        lowered = str(text or "").lower()
        return any(
            token in lowered
            for token in [
                "internal processing",
                "backend",
                "workflow",
                "tool call",
                "node",
                "prompt",
                "orchestration",
            ]
        )

    @staticmethod
    def _looks_like_root_menu(text: str) -> bool:
        lowered = str(text or "").lower()
        patterns = [
            r"pay now.*arrangement",
            r"arrangement.*follow[- ]?up",
            r"request an arrangement",
            r"schedule a follow[- ]?up",
            r"what would you like to do",
            r"which would you prefer",
            r"please choose one next step",
            r"please confirm how you would like to proceed",
            r"would you like to.*pay",
        ]
        return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns)

    def _meets_required_response_elements(
        self,
        rendered: str,
        *,
        required: list[str],
        context: dict[str, Any],
    ) -> bool:
        lowered = rendered.lower()
        response_mode = str(context.get("response_mode", "")).strip().lower()
        hardship_active = bool(
            (context.get("hardship_context") or {}).get("hardship_detected", False)
        ) or str(context.get("conversation_mode", "")).strip().lower() == "hardship_negotiation"
        checks = {
            "acknowledge_hardship": any(token in lowered for token in ["sorry", "understand", "appreciate", "difficult"]),
            "mention_due_amount": any(token in lowered for token in ["inr", "overdue", "due", "dues"]),
            "ask_affordable_amount": any(token in lowered for token in ["monthly amount", "manageable", "what amount", "realistically work"]),
            "ask_verification": "please confirm" in lowered or "confirm your" in lowered,
            "ask_commitment_date": any(token in lowered for token in ["payment date", "date can you", "when can you", "commit to"]),
            "discuss_arrangement": any(token in lowered for token in ["arrangement", "installment", "repayment", "plan"]),
            "present_offer": any(token in lowered for token in ["offer", "plan", "option", "arrangement"]),
            "ask_next_step": any(
                token in lowered
                for token in [
                    "what next",
                    "what would you like",
                    "would you like",
                    "please confirm",
                    "?",
                ]
            ),
            "confirm_amount_or_date": any(token in lowered for token in ["amount", "payment date", "date", "commit"]),
            "handoff_payload": context.get("response_target") == "discount_planning_agent",
            "closure": any(token in lowered for token in ["closing this conversation", "thank you"]),
        }
        if response_mode == "empathetic" or hardship_active:
            checks.setdefault("acknowledge_hardship", True)
        for item in required:
            if not checks.get(item, True):
                return False
        return True

    @staticmethod
    def _render_plan_outline(plan_outline: str) -> str:
        text = plan_outline.strip()
        if not text:
            return "Please confirm how you would like to proceed with your dues."

        normalized = re.sub(r"^(?:Proposed\s+plan|Plan)\s+for\s+[^:]+:\s*", "", text, flags=re.IGNORECASE).strip()
        executed_match = re.match(r"^Executed\s+([a-zA-Z0-9_]+)\s*:\s*(.*)$", text)
        if executed_match:
            details = executed_match.group(2).strip()
            if details:
                return (
                    f"I checked this for you: {details}. "
                    "Please tell me whether you want to pay now, request a revised arrangement, or schedule follow-up."
                )
            return "I completed the required verification step. Please tell me your preferred next action."

        if not normalized:
            normalized = text

        # Convert planning language into delivery language for end-user/agent handoff.
        normalized = normalized[0].upper() + normalized[1:] if len(normalized) > 1 else normalized.upper()
        if normalized.endswith("."):
            normalized = normalized[:-1]
        return (
            f"{normalized}. "
            "Please confirm your preferred next step."
        )

    @staticmethod
    def _verification_field_label(field_name: str) -> str:
        key = str(field_name).strip().lower()
        mapping = {
            "dob": "your date of birth (YYYY-MM-DD)",
            "phone": "your registered phone number",
        }
        return mapping.get(key, key.replace("_", " "))

    @staticmethod
    def _join_human_list(items: list[str]) -> str:
        values = [str(item).strip() for item in items if str(item).strip()]
        if not values:
            return "the required verification details"
        if len(values) == 1:
            return values[0]
        if len(values) == 2:
            return f"{values[0]} and {values[1]}"
        return f"{', '.join(values[:-1])}, and {values[-1]}"

    def _enforce_verification_field_scope(
        self,
        *,
        text: str,
        verification_context: dict[str, Any],
        customer_name: str,
    ) -> str | None:
        """Reject verification asks outside configured missing fields (email/pan/zip drift)."""
        lowered = str(text or "").lower()
        if not lowered:
            return None
        missing_labels = verification_context.get("missing_field_labels")
        missing_labels_list = (
            [str(x).strip().lower() for x in missing_labels if str(x).strip()]
            if isinstance(missing_labels, list)
            else []
        )
        required_fields = verification_context.get("required_fields")
        required_list = (
            [str(x).strip().lower() for x in required_fields if str(x).strip()]
            if isinstance(required_fields, list)
            else []
        )
        # Known sensitive fields that can appear due model drift.
        disallowed_tokens = {
            "email": "email" not in required_list and "registered email address" not in missing_labels_list,
            "pan": "last4_pan" not in required_list and "last 4 characters of your pan" not in missing_labels_list,
            "zip": "zip" not in required_list and "registered zip/pincode" not in missing_labels_list,
            "pincode": "zip" not in required_list and "registered zip/pincode" not in missing_labels_list,
        }
        if any(flag and token in lowered for token, flag in disallowed_tokens.items()):
            return self._render_verification_first_message(customer_name=customer_name, guard=verification_context)
        return None

    @staticmethod
    def _enforce_commitment_stage_scope(
        *,
        text: str,
        current_plan_node_id: str,
        user_input: str,
        verification_context: dict[str, Any],
    ) -> str:
        lowered = str(text or "").lower()
        if not lowered:
            return text
        node_id = str(current_plan_node_id or "").strip().lower()
        in_commitment_stage = node_id in {
            "collect_payment_intent",
            "resolve_outcome",
            "capture_commitment_follow_up",
            "capture_commitment",
        }
        if not in_commitment_stage:
            return text
        input_lower = str(user_input or "").lower()
        requested_email_link = any(
            token in input_lower
            for token in ["send link to email", "email me the link", "payment link via email", "email link"]
        )
        if ("registered email" in lowered or "email address" in lowered) and not requested_email_link:
            return (
                "Please choose one next step: pay now, request a repayment arrangement, "
                "or schedule a follow-up date."
            )
        return text

    @staticmethod
    def _enforce_post_verification_scope(
        *,
        text: str,
        current_plan_node_id: str,
        verification_context: dict[str, Any],
        customer_name: str,
        case_id: str,
        overdue_amount: float,
        user_input: str,
    ) -> str:
        lowered = str(text or "").lower()
        if not lowered:
            return text
        identity_verified = bool(verification_context.get("identity_verified", False))
        required_fields = verification_context.get("required_fields")
        required_list = (
            [str(x).strip().lower() for x in required_fields if str(x).strip()]
            if isinstance(required_fields, list)
            else []
        )
        asks_extra_email = (
            ("registered email" in lowered or "email address" in lowered)
            and "email" not in required_list
        )
        asks_any_verification = any(
            token in lowered
            for token in [
                "confirm your date of birth",
                "confirm your registered phone",
                "confirm your identity",
                "verify your identity",
            ]
        )
        if not identity_verified:
            return text
        if asks_extra_email or asks_any_verification:
            input_lower = str(user_input or "").lower()
            requested_email_link = any(
                token in input_lower
                for token in ["send link to email", "email me the link", "payment link via email", "email link"]
            )
            if requested_email_link and "email" in lowered and not asks_any_verification:
                return text
            node_id = str(current_plan_node_id or "").strip().lower()
            if node_id == "explain_dues":
                return (
                    f"Thank you for completing verification, {customer_name}. "
                    f"For case {case_id}, your overdue amount is INR {overdue_amount:.2f}. "
                    "Would you like to pay now, request an arrangement, or schedule a follow-up?"
                )
            return (
                "Please choose one next step: pay now, request a repayment arrangement, "
                "or schedule a follow-up date."
            )
        return text

    @staticmethod
    def _resolve_plan_node_label(conversation_plan: dict[str, Any], node_id: str) -> str:
        if not isinstance(conversation_plan, dict) or not node_id:
            return ""
        nodes = conversation_plan.get("nodes")
        if not isinstance(nodes, list):
            return ""
        for node in nodes:
            if not isinstance(node, dict):
                continue
            if str(node.get("id", "")).strip() == node_id:
                return str(node.get("label", node_id)).strip() or node_id
        return ""

    @staticmethod
    def _resolve_case_facts(*, state: AgentState, proposal: dict[str, Any]) -> dict[str, Any]:
        memory = state.get("memory")
        memory_state = dict(getattr(memory, "state", {})) if memory is not None else {}
        customer_name = str(memory_state.get("active_customer_name", "Customer")).strip() or "Customer"
        case_id = str(memory_state.get("active_case_id", "COLL-1001")).strip() or "COLL-1001"
        overdue_amount = float(memory_state.get("active_overdue_amount", 0.0) or 0.0)
        emi_amount = float(memory_state.get("active_emi_amount", 0.0) or 0.0)
        late_fee = float(memory_state.get("active_late_fee", 0.0) or 0.0)
        dpd = int(memory_state.get("active_dpd", 0) or 0)

        context = proposal.get("context") if isinstance(proposal.get("context"), dict) else {}
        if context:
            customer_name = str(context.get("customer_name", customer_name)).strip() or customer_name
            case_id = str(context.get("case_id", case_id)).strip() or case_id
            overdue_amount = float(context.get("overdue_amount", overdue_amount) or overdue_amount)

        opening_context = proposal.get("opening_context") if isinstance(proposal.get("opening_context"), dict) else {}
        if opening_context:
            customer_name = str(opening_context.get("customer_name", customer_name)).strip() or customer_name
            case_id = str(opening_context.get("case_id", case_id)).strip() or case_id
            overdue_amount = float(opening_context.get("overdue_amount", overdue_amount) or overdue_amount)

        case_snapshot = proposal.get("case_snapshot") if isinstance(proposal.get("case_snapshot"), dict) else {}
        if case_snapshot:
            customer_name = str(case_snapshot.get("customer_name", customer_name)).strip() or customer_name
            case_id = str(case_snapshot.get("case_id", case_id)).strip() or case_id
            overdue_amount = float(case_snapshot.get("overdue_amount", overdue_amount) or overdue_amount)
            emi_amount = float(case_snapshot.get("emi_amount", emi_amount) or emi_amount)
            late_fee = float(case_snapshot.get("late_fee", late_fee) or late_fee)
            dpd = int(case_snapshot.get("dpd", dpd) or dpd)

        payment_context = proposal.get("payment_context") if isinstance(proposal.get("payment_context"), dict) else {}
        if payment_context:
            customer_name = str(payment_context.get("customer_name", customer_name)).strip() or customer_name
            case_id = str(payment_context.get("case_id", case_id)).strip() or case_id
            overdue_amount = float(payment_context.get("overdue_amount", overdue_amount) or overdue_amount)

        return {
            "customer_name": customer_name,
            "case_id": case_id,
            "overdue_amount": overdue_amount,
            "emi_amount": emi_amount,
            "late_fee": late_fee,
            "dpd": dpd,
        }

    @staticmethod
    def _render_template(template: str, values: dict[str, Any]) -> str:
        rendered = template
        for key, value in values.items():
            rendered = rendered.replace(f"{{{key}}}", str(value))
        return rendered

    @staticmethod
    def _post_process_rendered_response(
        *,
        text: str,
        customer_name: str,
        overdue_amount: float,
        is_opening_turn: bool,
        ensure_intro: bool = False,
    ) -> str:
        rendered = text.strip()

        # Normalize robotic verification phrasing from model variants.
        rendered = re.sub(
            r"(?i)\bi\s+(?:see|noticed)\s+you(?:'ve| have)\s+already\s+confirmed[^.!?]*[.!?]\s*",
            "Thank you. ",
            rendered,
        )

        # Replace placeholder fragments with deterministic values.
        rendered = re.sub(
            r"\[(?:insert\s+)?amount\]",
            f"INR {overdue_amount:.2f}",
            rendered,
            flags=re.IGNORECASE,
        )
        rendered = re.sub(
            r"\[(?:customer\s+)?name\]",
            customer_name,
            rendered,
            flags=re.IGNORECASE,
        )

        # Avoid repeating opener greeting on follow-up turns.
        if not is_opening_turn:
            rendered = re.sub(r'^["\']?\s*(hello|hi)\s+[^,]{1,60},\s*', "", rendered, flags=re.IGNORECASE).strip()
            rendered = rendered.strip(' "\'')
        elif ensure_intro and "alex" not in rendered.lower():
            rendered = (
                f"Hello {customer_name}, this is Alex from collections. "
                f"{rendered}"
            ).strip()

        return rendered

    @staticmethod
    def _is_provider_rate_limit_error(error_text: str) -> bool:
        lowered = str(error_text or "").lower()
        return (
            "rate limit" in lowered
            or "rate_limit_exceeded" in lowered
            or "error code: 429" in lowered
            or "tokens per day" in lowered
            or "tpm" in lowered
        )

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
    def _compact_observation(observation: Any) -> dict[str, Any]:
        if not isinstance(observation, dict):
            return {}
        phase = observation.get("tool_phase") if isinstance(observation.get("tool_phase"), dict) else observation
        if not isinstance(phase, dict):
            return {}
        output = phase.get("output") if isinstance(phase.get("output"), dict) else {}
        return {
            "tool_name": str(phase.get("tool_name", "")).strip(),
            "status": str(output.get("status", "")).strip(),
            "needs_additional_action": bool(output.get("needs_additional_action", False)),
            "keys": sorted([str(k) for k in output.keys()])[:12],
        }

    @staticmethod
    def _compact_conversation_plan(plan: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(plan, dict):
            return {}
        nodes_raw = plan.get("nodes") if isinstance(plan.get("nodes"), list) else []
        edges_raw = plan.get("edges") if isinstance(plan.get("edges"), list) else []
        markers = plan.get("step_markers") if isinstance(plan.get("step_markers"), dict) else {}
        nodes = []
        for node in nodes_raw[:10]:
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
        edges = []
        for edge in edges_raw[:16]:
            if not isinstance(edge, dict):
                continue
            edges.append(
                {
                    "from": str(edge.get("from", "")).strip(),
                    "to": str(edge.get("to", "")).strip(),
                    "condition": str(edge.get("condition", "")).strip(),
                }
            )
        marker_view = {}
        for key, raw in markers.items():
            if len(marker_view) >= 12:
                break
            if not isinstance(raw, dict):
                continue
            marker_view[str(key)] = str(raw.get("state", "pending")).strip()
        return {
            "plan_id": str(plan.get("plan_id", "")).strip(),
            "version": int(plan.get("version", 1) or 1),
            "status": str(plan.get("status", "active")).strip(),
            "current_node_id": str(plan.get("current_node_id", "")).strip(),
            "next_node_ids": [str(x).strip() for x in (plan.get("next_node_ids") or []) if str(x).strip()][:6],
            "nodes": nodes,
            "edges": edges,
            "step_markers": marker_view,
        }

    @staticmethod
    def _compact_plan_proposal(*, proposal: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(proposal, dict):
            return {}
        context = proposal.get("context") if isinstance(proposal.get("context"), dict) else {}
        next_actions = proposal.get("next_actions") if isinstance(proposal.get("next_actions"), list) else []
        return {
            "target": str(proposal.get("target", "")).strip(),
            "intent": str(proposal.get("intent", "")).strip(),
            "plan_outline": str(proposal.get("plan_outline", "")).strip(),
            "draft_response": str(proposal.get("draft_response", "")).strip(),
            "next_actions": [str(x).strip() for x in next_actions[:8] if str(x).strip()],
            "context": {
                "case_id": str(context.get("case_id", "")).strip(),
                "customer_name": str(context.get("customer_name", "")).strip(),
                "overdue_amount": context.get("overdue_amount"),
                "observed_tool": str(context.get("observed_tool", "")).strip(),
            },
        }
