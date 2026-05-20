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


class _CompiledResponseDirectivePayload(BaseModel):
    template_id: str = "safe_follow_up"
    response_target: str = "customer"
    tone: str = "informational"
    render_variables: dict[str, Any] = Field(default_factory=dict)
    response_constraints: dict[str, Any] = Field(default_factory=dict)
    fallback_template_id: str = "safe_follow_up"


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
            response_target=str(directive.get("response_target", render_context.get("response_target", "customer"))).strip().lower()
            or "customer",
        )
        response_target = str(directive.get("response_target", render_context.get("response_target", "customer"))).strip().lower() or "customer"
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
            "facts": facts,
            "verification_context": verification_context,
            "observations": state.get("observations") if isinstance(state.get("observations"), list) else [],
            "observation": state.get("observation"),
            "plan_proposal": proposal,
            "conversation_plan": conversation_plan,
            "current_plan_node_id": current_plan_node_id,
        }

    def _resolve_response_directive(
        self,
        *,
        state: AgentState,
        proposal: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        raw_compiled_directive = proposal.get("compiled_response_directive")
        if isinstance(raw_compiled_directive, dict):
            directive = self._normalize_compiled_response_directive(raw_compiled_directive)
            if directive is not None:
                return directive

        raw_legacy_directive = proposal.get("response_directive")
        if not isinstance(raw_legacy_directive, dict):
            raw_legacy_directive = {
                "conversation_objective": proposal.get("conversation_objective"),
                "dialogue_action": proposal.get("dialogue_action"),
                "response_mode": proposal.get("response_mode"),
                "customer_facing_goal": proposal.get("customer_facing_goal"),
                "handoff_target": proposal.get("handoff_target"),
                "draft_response": proposal.get("draft_response"),
            }
        raw_legacy_directive = {key: value for key, value in raw_legacy_directive.items() if value is not None}
        directive = self._compile_legacy_response_directive(
            raw_directive=raw_legacy_directive,
            proposal=proposal,
            context=context,
        )
        if directive is not None:
            return directive
        return self._safe_renderer_fallback_directive(
            state=state,
            proposal=proposal,
            context=context,
        )

    @staticmethod
    def _normalize_compiled_response_directive(raw_directive: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(raw_directive, dict) or not raw_directive:
            return None
        try:
            payload = _CompiledResponseDirectivePayload.model_validate(raw_directive)
        except Exception:
            return None
        directive = payload.model_dump(mode="json")
        directive["template_id"] = str(directive.get("template_id", "")).strip()
        directive["response_target"] = str(directive.get("response_target", "customer")).strip().lower() or "customer"
        directive["tone"] = str(directive.get("tone", "informational")).strip().lower() or "informational"
        directive["fallback_template_id"] = str(directive.get("fallback_template_id", "")).strip() or directive["template_id"]
        if directive["template_id"] == "":
            return None
        return directive

    def _compile_legacy_response_directive(
        self,
        *,
        raw_directive: dict[str, Any],
        proposal: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not isinstance(raw_directive, dict) or not raw_directive:
            return None
        template_id = str(raw_directive.get("template_id", "")).strip() or self._template_id_from_policy_fields(raw_directive)
        if not template_id:
            return None
        response_target = str(
            raw_directive.get("response_target", proposal.get("target", context.get("response_target", "customer")))
        ).strip().lower() or "customer"
        tone = str(raw_directive.get("tone", raw_directive.get("response_mode", "informational"))).strip().lower() or "informational"
        fallback_template_id = str(raw_directive.get("fallback_template_id", "")).strip() or template_id
        render_variables = self._build_render_variables(
            raw_directive=raw_directive,
            proposal=proposal,
            context=context,
        )
        response_constraints = self._build_render_constraints(
            response_target=response_target,
            context=context,
        )
        return self._normalize_compiled_response_directive(
            {
                "template_id": template_id,
                "response_target": response_target,
                "tone": tone,
                "render_variables": render_variables,
                "response_constraints": response_constraints,
                "fallback_template_id": fallback_template_id,
            }
        )

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
            return self._normalize_compiled_response_directive(
                {
                    "template_id": "handoff_payload",
                    "response_target": "discount_planning_agent",
                    "tone": "firm",
                    "render_variables": {
                        "message_hint": "Prepare specialist handoff payload and wait for discount recommendation.",
                    },
                    "response_constraints": self._build_render_constraints(
                        response_target="discount_planning_agent",
                        context=context,
                    ),
                    "fallback_template_id": "handoff_payload",
                }
            ) or {}
        if not bool(verification_context.get("identity_verified", False)):
            return self._normalize_compiled_response_directive(
                {
                    "template_id": "verification_request",
                    "response_target": response_target,
                    "tone": "compliance",
                    "render_variables": self._build_render_variables(
                        raw_directive={},
                        proposal=proposal,
                        context=context,
                    ),
                    "response_constraints": self._build_render_constraints(
                        response_target=response_target,
                        context=context,
                    ),
                    "fallback_template_id": "verification_request",
                }
            ) or {}
        return self._normalize_compiled_response_directive(
            {
                "template_id": "safe_follow_up",
                "response_target": response_target,
                "tone": "informational",
                "render_variables": {
                    "customer_facing_goal": "Please let me know how you would like to proceed.",
                },
                "response_constraints": self._build_render_constraints(
                    response_target=response_target,
                    context=context,
                ),
                "fallback_template_id": "safe_follow_up",
            }
        ) or {}

    @staticmethod
    def _template_id_from_policy_fields(raw_directive: dict[str, Any]) -> str:
        objective = str(raw_directive.get("conversation_objective", "")).strip().lower()
        action = str(raw_directive.get("dialogue_action", "")).strip().lower()
        if action == "ask_verification" or objective == "collect_verification":
            return "verification_request"
        if action == "ask_affordable_amount" or objective == "assess_affordability":
            return "capacity_question"
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
        return "safe_follow_up"

    def _build_response_render_debug(
        self,
        *,
        directive: dict[str, Any],
        response_target: str,
    ) -> dict[str, Any]:
        return {
            "template_selected": str(directive.get("template_id", "")).strip(),
            "fallback_template_id": str(directive.get("fallback_template_id", "")).strip(),
            "response_mode": str(directive.get("tone", "")).strip(),
            "response_target": response_target,
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
        facts = context.get("facts") if isinstance(context.get("facts"), dict) else self._resolve_case_facts(state=state, proposal=proposal)
        prior_agent_response = str(memory_state.get("last_agent_response", "")).strip()
        verification_context_merged = self._build_verification_context(state=state, memory_state=memory_state, proposal=proposal)
        response_directive = (
            dict(response_directive)
            if isinstance(response_directive, dict)
            else self._resolve_response_directive(state=state, proposal=proposal, context=context)
        )
        prior_agent_response_short = self._truncate_text(prior_agent_response, 280)
        compact_observation = self._compact_observation(observation)

        system_prompt = (f"{self.system_prompt or ''}\n{self.render_system_prompt or ''}").strip()
        user_prompt = self._render_template(
            self.render_user_prompt,
            {
                "user_input": user_input,
                "response_target": str(response_directive.get("response_target", response_target)).strip().lower() or response_target,
                "prior_agent_response": prior_agent_response_short,
                "template_id": str(response_directive.get("template_id", "")).strip(),
                "tone": str(response_directive.get("tone", "")).strip(),
                "fallback_template_id": str(response_directive.get("fallback_template_id", "")).strip(),
                "render_variables_json": self._json_compact(
                    response_directive.get("render_variables", {}),
                    max_chars=1200,
                ),
                "response_constraints_json": self._json_compact(
                    response_directive.get("response_constraints", {}),
                    max_chars=700,
                ),
                "verification_context_json": self._json_compact(verification_context_merged, max_chars=600),
                "observation_json": self._json_compact(compact_observation, max_chars=700),
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
        label_map = {
            "dob": "your date of birth (YYYY-MM-DD)",
            "phone": "your registered phone number",
            "email": "your registered email address",
            "zip": "your registered zip/pincode",
            "last4_pan": "the last 4 characters of your PAN",
        }
        missing_field_labels = [label_map.get(field, field.replace("_", " ")) for field in missing_fields]
        identity_verified = bool(state.get("identity_verified", memory_state.get("identity_verified", False)))
        verification_incomplete = (not identity_verified) or bool(missing_fields)
        return {
            "identity_verified": identity_verified,
            "required_fields": required_fields,
            "verification_entities": verification_entities,
            "verification_missing_fields": missing_fields,
            "missing_field_labels": missing_field_labels,
            "missing_fields_human": self._join_human_list(missing_field_labels),
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
        response_target = str(response_target).strip().lower() or "customer"
        if response_target == "self":
            return "Proceed with the prepared next step."
        if response_target == "discount_planning_agent":
            return "Prepare specialist handoff payload and wait for discount recommendation."

        template_id = str(directive.get("template_id", "safe_follow_up")).strip() or "safe_follow_up"
        render_variables = dict(directive.get("render_variables", {})) if isinstance(directive.get("render_variables"), dict) else {}
        tone = str(directive.get("tone", "informational")).strip().lower() or "informational"
        customer_name = str(render_variables.get("customer_name", "Customer")).strip() or "Customer"
        missing_fields = str(render_variables.get("missing_fields", self.verification_default_missing_text)).strip()
        overdue_amount_text = str(render_variables.get("overdue_amount_text", "0.00")).strip() or "0.00"
        customer_facing_goal = str(render_variables.get("customer_facing_goal", "")).strip()
        message_hint = str(render_variables.get("message_hint", "")).strip()
        opening_turn = bool(render_variables.get("opening_turn", False))

        if template_id == "verification_request":
            prefix = self.verification_hardship_prefix if tone == "empathetic" else ""
            if opening_turn:
                return (
                    f"Hello {customer_name}, this is Alex from the bank's collections team. "
                    f"{prefix}Before I share details, please confirm {missing_fields}."
                ).strip()
            return f"{prefix}Please confirm {missing_fields}.".strip()
        if template_id == "dues_explanation":
            return (
                f"Thank you {customer_name}. Your overdue amount is INR {overdue_amount_text}. "
                "What would you like to do next to bring the account current?"
            ).strip()
        if template_id == "capacity_question":
            prefix = "I am sorry to hear that. " if tone == "empathetic" else ""
            return (
                f"{prefix}To explore a manageable arrangement, what monthly amount would realistically work for you right now?"
            ).strip()
        if template_id == "arrangement_discussion":
            return customer_facing_goal or "Let us work toward a practical repayment option. What installment amount would be manageable for you?"
        if template_id == "commitment_confirmation":
            return customer_facing_goal or "Thank you. What amount and payment date can you confidently commit to for the next step?"
        if template_id == "handoff_payload":
            return message_hint or "Prepare specialist handoff payload and wait for discount recommendation."
        return customer_facing_goal or "Please let me know how you would like to proceed."

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
                "placeholder_cleanup",
                "internal_terminology_cleanup",
                "verification_amount_guard",
            ],
            "forbidden_actions_blocked": [],
        }
        if not rendered:
            result["forbidden_actions_blocked"].append("empty_response")
            return result
        if self._contains_unresolved_placeholders(rendered):
            result["forbidden_actions_blocked"].append("unresolved_placeholders")
        verification_context = context.get("verification_context") if isinstance(context.get("verification_context"), dict) else {}
        constraints = directive.get("response_constraints") if isinstance(directive.get("response_constraints"), dict) else {}
        if bool(constraints.get("no_dues_before_verification", False)):
            lowered = rendered.lower()
            if any(token in lowered for token in ["inr ", "overdue", "dues", "amount", "emi", "late fee"]):
                result["forbidden_actions_blocked"].append("disclose_dues_before_verification")
        if bool(constraints.get("avoid_internal_terms", True)) and self._contains_internal_processing(rendered):
            result["forbidden_actions_blocked"].append("mention_internal_processing")
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
        cleaned = re.sub(r"\[(?:insert\s+)?[^\]]+\]", "", cleaned, flags=re.IGNORECASE)
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
    def _contains_unresolved_placeholders(text: str) -> bool:
        return bool(re.search(r"\{[^}]+\}|\[[^\]]+\]", str(text or "")))

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

    def _build_render_variables(
        self,
        *,
        raw_directive: dict[str, Any],
        proposal: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        facts = context.get("facts") if isinstance(context.get("facts"), dict) else {}
        memory_state = context.get("memory_state") if isinstance(context.get("memory_state"), dict) else {}
        verification_context = context.get("verification_context") if isinstance(context.get("verification_context"), dict) else {}
        customer_name = str(facts.get("customer_name", memory_state.get("active_customer_name", "Customer"))).strip() or "Customer"
        overdue_amount = float(facts.get("overdue_amount", memory_state.get("active_overdue_amount", 0.0)) or 0.0)
        missing_fields = verification_context.get("missing_fields_human")
        if not missing_fields:
            missing_fields = self._join_human_list(verification_context.get("missing_field_labels", []))
        turn_index = int(memory_state.get("turn_index", 0) or 0)
        return {
            "customer_name": customer_name,
            "case_id": str(facts.get("case_id", memory_state.get("active_case_id", "COLL-1001"))).strip() or "COLL-1001",
            "overdue_amount_text": f"{overdue_amount:.2f}",
            "missing_fields": str(missing_fields or self.verification_default_missing_text).strip(),
            "customer_facing_goal": str(raw_directive.get("customer_facing_goal", "")).strip(),
            "message_hint": str(raw_directive.get("draft_response", proposal.get("draft_response", ""))).strip(),
            "opening_turn": turn_index <= 0,
        }

    def _build_render_constraints(
        self,
        *,
        response_target: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        verification_context = context.get("verification_context") if isinstance(context.get("verification_context"), dict) else {}
        return {
            "avoid_internal_terms": True,
            "avoid_placeholders": True,
            "avoid_verbatim_repeat": True,
            "ask_one_question": response_target == "customer",
            "no_dues_before_verification": not bool(verification_context.get("identity_verified", False)),
            "verification_incomplete": bool(verification_context.get("verification_incomplete", False)),
        }

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
