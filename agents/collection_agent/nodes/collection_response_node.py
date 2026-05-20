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
            "response_render_debug": None,
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
        update["response_render_debug"] = self.last_render_debug.get("response_render_debug")
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
        render_debug = self._build_response_render_debug(
            directive=directive,
            response_target=str(render_context.get("response_target", "customer")).strip().lower() or "customer",
        )
        response_target = str(render_context.get("response_target", "customer")).strip().lower() or "customer"
        if response_target == "discount_planning_agent":
            render_debug["renderer_fallback_used"] = True
            self.last_render_debug["response_render_debug"] = render_debug
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
                validation = self._validate_response_against_directive(
                    text=llm_response,
                    directive=directive,
                    context=render_context,
                )
                render_debug["policy_filters_applied"] = validation["policy_filters_applied"]
                render_debug["forbidden_actions_blocked"] = validation["forbidden_actions_blocked"]
                if validation["text"]:
                    render_debug["renderer_fallback_used"] = False
                    self.last_render_debug["response_render_debug"] = render_debug
                    return self._apply_minimal_safety_cleanup(
                        text=validation["text"],
                        context=render_context,
                        directive=directive,
                    )
                self.last_render_debug["fallback_reason"] = "directive_validation_failed"
            else:
                render_debug["policy_filters_applied"] = ["llm_render_attempt"]
            if self.strict_llm_mode:
                fallback_reason = str(self.last_render_debug.get("fallback_reason", "")).strip()
                if fallback_reason == "provider_rate_limit":
                    raise RuntimeError(
                        "CollectionResponseNode rate-limited by provider while strict_llm_mode is enabled. "
                        f"Underlying error: {self.last_render_debug.get('llm_error', 'unknown')}"
                    )
        render_debug["renderer_fallback_used"] = True
        self.last_render_debug["response_render_debug"] = render_debug
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
        verification_context = self._build_verification_context(
            state=state,
            memory_state=memory_state,
            proposal=proposal,
        )
        return {
            "memory_state": memory_state,
            "user_input": user_input,
            "response_target": response_target,
            "conversation_plan": conversation_plan,
            "facts": facts,
            "current_plan_node_id": current_plan_node_id,
            "verification_context": verification_context,
            "negotiation_stage": str(state.get("negotiation_stage", memory_state.get("negotiation_stage", "none"))).strip(),
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
        directive = self._normalize_response_directive(raw_directive)
        if directive is None:
            directive = self._safe_renderer_fallback_directive(
                state=state,
                proposal=proposal,
                context=context,
            )
        return directive

    @staticmethod
    def _normalize_response_directive(raw_directive: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(raw_directive, dict) or not raw_directive:
            return None
        try:
            payload = _ResponseDirectivePayload.model_validate(raw_directive)
        except Exception:
            return None
        directive = payload.model_dump(mode="json")
        if not str(directive.get("conversation_objective", "")).strip():
            return None
        if not str(directive.get("dialogue_action", "")).strip():
            return None
        directive["required_response_elements"] = [
            str(item).strip()
            for item in directive.get("required_response_elements", [])
            if str(item).strip()
        ]
        directive["forbidden_dialogue_actions"] = [
            str(item).strip()
            for item in directive.get("forbidden_dialogue_actions", [])
            if str(item).strip()
        ]
        directive["allowed_dialogue_actions"] = [
            str(item).strip()
            for item in directive.get("allowed_dialogue_actions", [])
            if str(item).strip()
        ]
        directive.setdefault("customer_facing_goal", None)
        directive.setdefault("handoff_target", None)
        return directive

    def _safe_renderer_fallback_directive(
        self,
        *,
        state: AgentState,
        proposal: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        response_target = str(context.get("response_target", "customer")).strip().lower() or "customer"
        verification_context = (
            context.get("verification_context")
            if isinstance(context.get("verification_context"), dict)
            else self._build_verification_context(
                state=state,
                memory_state=context.get("memory_state") if isinstance(context.get("memory_state"), dict) else {},
                proposal=proposal,
            )
        )
        if response_target == "discount_planning_agent":
            return {
                "conversation_objective": "handoff_to_offer_agent",
                "dialogue_action": "handoff",
                "response_mode": "firm",
                "required_response_elements": ["handoff_payload"],
                "forbidden_dialogue_actions": ["restart_collections_menu", "mention_internal_processing"],
                "allowed_dialogue_actions": ["handoff"],
                "customer_facing_goal": "Prepare a specialist handoff without changing customer dialogue direction.",
                "handoff_target": "discount_planning_agent",
            }
        if not bool(verification_context.get("identity_verified", False)):
            return {
                "conversation_objective": "collect_verification",
                "dialogue_action": "ask_verification",
                "response_mode": "compliance",
                "required_response_elements": ["ask_only_missing_verification_fields"],
                "forbidden_dialogue_actions": [
                    "disclose_dues_before_verification",
                    "restart_collections_menu",
                    "mention_internal_processing",
                ],
                "allowed_dialogue_actions": ["ask_verification"],
                "customer_facing_goal": "Ask only for the missing verification details needed to continue securely.",
                "handoff_target": None,
            }
        return {
            "conversation_objective": "close_conversation",
            "dialogue_action": "ask_next_step",
            "response_mode": "informational",
            "required_response_elements": [],
            "forbidden_dialogue_actions": ["restart_collections_menu", "mention_internal_processing"],
            "allowed_dialogue_actions": ["ask_next_step"],
            "customer_facing_goal": "Please let me know how you would like to proceed.",
            "handoff_target": None,
        }

    @staticmethod
    def _select_template_name(*, directive: dict[str, Any], response_target: str) -> str:
        if response_target == "discount_planning_agent":
            return "handoff_payload"
        objective = str(directive.get("conversation_objective", "")).strip().lower()
        action = str(directive.get("dialogue_action", "")).strip().lower()
        if action == "ask_verification" or objective == "collect_verification":
            return "verification_request"
        if action == "ask_affordable_amount" or objective == "assess_affordability":
            return "affordability_question"
        if action in {"present_offer", "discuss_arrangement"} or objective in {
            "present_arrangement_options",
            "negotiate_installment",
        }:
            return "arrangement_discussion"
        if action in {"ask_commitment_date", "confirm_payment_intent"} or objective in {
            "confirm_commitment",
            "capture_promise",
        }:
            return "commitment_confirmation"
        if objective == "explain_dues" or action == "present_due_amount":
            return "dues_explanation"
        if objective == "handoff_to_offer_agent" or action == "handoff":
            return "handoff_payload"
        if objective == "close_conversation":
            return "safe_follow_up"
        return "safe_follow_up"

    def _build_response_render_debug(
        self,
        *,
        directive: dict[str, Any],
        response_target: str,
    ) -> dict[str, Any]:
        return {
            "conversation_objective": str(directive.get("conversation_objective", "")).strip(),
            "dialogue_action": str(directive.get("dialogue_action", "")).strip(),
            "response_mode": str(directive.get("response_mode", "")).strip(),
            "template_selected": self._select_template_name(
                directive=directive,
                response_target=response_target,
            ),
            "policy_filters_applied": [],
            "forbidden_actions_blocked": [],
            "renderer_fallback_used": False,
        }

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
                "response_directive_json": self._json_compact(response_directive, max_chars=800),
                "conversation_objective": str(response_directive.get("conversation_objective", "")).strip(),
                "dialogue_action": str(response_directive.get("dialogue_action", "")).strip(),
                "response_mode": str(response_directive.get("response_mode", "")).strip(),
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


    def _fallback_from_directive(
        self,
        *,
        directive: dict[str, Any],
        context: dict[str, Any],
        response_target: str,
    ) -> str:
        objective = str(directive.get("conversation_objective", "close_conversation")).strip().lower()
        dialogue_action = str(directive.get("dialogue_action", "")).strip().lower()
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
        empathy_prefix = "I am sorry to hear that. " if response_mode == "empathetic" else ""

        if objective == "collect_verification" or dialogue_action == "ask_verification":
            prefix = self.verification_hardship_prefix if response_mode == "empathetic" else ""
            return (
                f"Hello {customer_name}, this is Alex from the bank's collections team. "
                f"{prefix}Before I share details, please confirm {missing_fields}."
            ).strip()
        if objective == "explain_dues" or dialogue_action == "present_due_amount":
            return (
                f"Thank you {customer_name}. Your overdue amount is INR {overdue_amount:.2f}. "
                "What would you like to do next to bring the account current?"
            ).strip()
        if objective == "assess_affordability" or dialogue_action == "ask_affordable_amount":
            return (
                f"{empathy_prefix}To explore a manageable arrangement, what monthly amount would realistically work for you right now?"
            ).strip()
        if objective == "present_arrangement_options" or dialogue_action == "present_offer":
            return "Let us work toward a practical repayment option. What installment amount would you be comfortable committing to?"
        if objective == "negotiate_installment" or dialogue_action == "discuss_arrangement":
            return "What installment amount would feel manageable for you at the moment?"
        if objective == "confirm_commitment" or dialogue_action == "ask_commitment_date":
            return "Thank you. What amount and payment date can you confidently commit to for the next step?"
        if objective == "capture_promise":
            return "Please confirm the amount and payment date you can commit to so I can capture your promise."
        if objective == "handoff_to_offer_agent":
            return "Prepare specialist handoff payload and wait for discount recommendation."
        if objective == "close_conversation":
            goal = str(directive.get("customer_facing_goal", "")).strip()
            return goal or "Please let me know how you would like to proceed."

        goal = str(directive.get("customer_facing_goal", "")).strip()
        return goal or "Please let me know how you would like to proceed."

    def _validate_response_against_directive(
        self,
        *,
        text: str,
        directive: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        rendered = self._apply_minimal_safety_cleanup(text=text, context=context, directive=directive)
        result = {
            "text": None,
            "policy_filters_applied": [
                "minimal_safety_cleanup",
                "forbidden_dialogue_actions",
                "required_response_elements",
                "dialogue_action_alignment",
                "stage_consistency",
            ],
            "forbidden_actions_blocked": [],
        }
        if not rendered:
            return result
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
            result["forbidden_actions_blocked"].append("mention_internal_processing")
        if "restart_collections_menu" in forbidden and self._looks_like_root_menu(rendered):
            result["forbidden_actions_blocked"].append("restart_collections_menu")
        if "ask_pay_now_or_arrangement" in forbidden and self._looks_like_root_menu(rendered):
            result["forbidden_actions_blocked"].append("ask_pay_now_or_arrangement")
        if "disclose_dues_before_verification" in forbidden and not identity_verified:
            lowered = rendered.lower()
            if any(token in lowered for token in ["inr ", "overdue", "dues", "amount", "emi", "late fee"]):
                result["forbidden_actions_blocked"].append("disclose_dues_before_verification")
        required = [
            str(item).strip().lower()
            for item in (directive.get("required_response_elements") or [])
            if str(item).strip()
        ]
        if required and not self._meets_required_response_elements(rendered, required=required, context=context):
            result["forbidden_actions_blocked"].append("missing_required_response_elements")
        if not self._matches_dialogue_action(rendered=rendered, directive=directive, context=context):
            result["forbidden_actions_blocked"].append("dialogue_action_mismatch")
        if self._contradicts_stage_or_objective(rendered=rendered, directive=directive, context=context):
            result["forbidden_actions_blocked"].append("stage_contradiction")
        if result["forbidden_actions_blocked"]:
            return result
        result["text"] = rendered
        return result

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
        # Broad phrase checks are a final safety filter only. The directive is
        # still the authoritative source of conversation policy.
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
        for item in required:
            if not checks.get(item, True):
                return False
        return True

    def _matches_dialogue_action(
        self,
        *,
        rendered: str,
        directive: dict[str, Any],
        context: dict[str, Any],
    ) -> bool:
        response_target = str(context.get("response_target", "customer")).strip().lower() or "customer"
        expected_template = self._select_template_name(
            directive=directive,
            response_target=response_target,
        )
        actual_template = self._infer_rendered_template(rendered=rendered, response_target=response_target)
        if expected_template == "safe_follow_up":
            return actual_template in {"safe_follow_up", "dues_explanation"}
        if actual_template is None:
            return False
        return actual_template == expected_template

    def _contradicts_stage_or_objective(
        self,
        *,
        rendered: str,
        directive: dict[str, Any],
        context: dict[str, Any],
    ) -> bool:
        objective = str(directive.get("conversation_objective", "")).strip().lower()
        stage = str(context.get("negotiation_stage", "none")).strip().lower()
        response_target = str(context.get("response_target", "customer")).strip().lower() or "customer"
        actual_template = self._infer_rendered_template(rendered=rendered, response_target=response_target)
        if actual_template is None:
            return False
        if stage in {"discovering_hardship", "assessing_capacity"} and actual_template == "commitment_confirmation":
            return True
        if objective == "assess_affordability" and actual_template in {"commitment_confirmation", "dues_explanation"}:
            return True
        if objective in {"present_arrangement_options", "negotiate_installment"} and actual_template == "dues_explanation":
            return True
        return False

    @staticmethod
    def _infer_rendered_template(*, rendered: str, response_target: str) -> str | None:
        response_target = str(response_target).strip().lower() or "customer"
        if response_target == "discount_planning_agent":
            return "handoff_payload"
        lowered = str(rendered or "").lower()
        if not lowered:
            return None
        if any(token in lowered for token in ["prepare specialist handoff payload", "discount recommendation"]):
            return "handoff_payload"
        if "please confirm" in lowered or "confirm your" in lowered:
            if any(token in lowered for token in ["date of birth", "phone number", "registered phone"]):
                return "verification_request"
        if any(token in lowered for token in ["payment date", "when can you", "commit to", "can you confidently commit"]):
            return "commitment_confirmation"
        if any(token in lowered for token in ["monthly amount", "what amount would", "manageable arrangement", "realistically work for you"]):
            return "affordability_question"
        if any(token in lowered for token in ["installment", "repayment option", "repayment plan", "arrangement", "comfortable committing"]):
            return "arrangement_discussion"
        if any(token in lowered for token in ["overdue amount", "dues", "amount is inr", "inr "]):
            return "dues_explanation"
        if "next step" in lowered or "how you would like to proceed" in lowered or "what would you like to do next" in lowered:
            return "safe_follow_up"
        if "?" in lowered:
            return "safe_follow_up"
        return None

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
