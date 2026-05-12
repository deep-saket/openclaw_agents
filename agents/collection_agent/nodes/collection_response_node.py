"""Collection-specific response node with target routing."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from agents.collection_agent.llm_structured import StructuredOutputRunner
from src.nodes.response_node import ResponseNode
from src.nodes.types import AgentState, NodeUpdate


class _ResponsePayload(BaseModel):
    message: str
    response_target: str = "customer"


@dataclass(slots=True)
class CollectionResponseNode(ResponseNode):
    """Emits response text and a response target for next-hop routing."""

    default_target: str = "customer"

    def execute(self, state: AgentState) -> NodeUpdate:
        plan = state.get("plan_proposal") if isinstance(state.get("plan_proposal"), dict) else {}
        if plan:
            update: NodeUpdate = {"response": self._render_from_proposal(state=state, proposal=plan)}
            plan_target = str(plan.get("target", "")).strip().lower()
            if plan_target:
                update["response_target"] = plan_target
        else:
            update = ResponseNode.execute(self, state)

        target = str(update.get("response_target", state.get("response_target", self.default_target))).strip().lower()
        if target == "discount_planning_agent":
            target = "self"
        if target not in {"customer", "self"}:
            target = self.default_target
        update["response_target"] = target
        return update

    def route(self, state: AgentState) -> str:
        target = str(state.get("response_target", self.default_target)).strip().lower()
        if target == "discount_planning_agent":
            return "self"
        if target not in {"customer", "self"}:
            return self.default_target
        return target

    def _render_from_proposal(self, *, state: AgentState, proposal: dict[str, Any]) -> str:
        memory = state.get("memory")
        memory_state = dict(getattr(memory, "state", {})) if memory is not None else {}
        response_target = str(proposal.get("target", state.get("response_target", "customer"))).strip().lower() or "customer"
        conversation_plan = (
            proposal.get("conversation_plan")
            if isinstance(proposal.get("conversation_plan"), dict)
            else (state.get("conversation_plan") if isinstance(state.get("conversation_plan"), dict) else {})
        )
        facts = self._resolve_case_facts(state=state, proposal=proposal)
        verification_guard_context = self._build_verification_guard_context(
            state=state,
            memory_state=memory_state,
            response_target=response_target,
            conversation_plan=conversation_plan,
            customer_name=str(facts.get("customer_name", "Customer")).strip() or "Customer",
        )

        if self.llm is not None:
            llm_response = self._llm_render_from_proposal(
                state=state,
                proposal=proposal,
                verification_guard_context=verification_guard_context,
            )
            if llm_response:
                return llm_response
        return self._fallback_render_from_proposal(proposal=proposal)

    def _llm_render_from_proposal(
        self,
        *,
        state: AgentState,
        proposal: dict[str, Any],
        verification_guard_context: dict[str, Any] | None = None,
    ) -> str | None:
        user_input = str(state.get("user_input", ""))
        observation = state.get("observation")
        memory = state.get("memory")
        memory_state = dict(getattr(memory, "state", {})) if memory is not None else {}
        response_target = str(proposal.get("target", state.get("response_target", "customer"))).strip().lower() or "customer"
        conversation_plan = (
            proposal.get("conversation_plan")
            if isinstance(proposal.get("conversation_plan"), dict)
            else (state.get("conversation_plan") if isinstance(state.get("conversation_plan"), dict) else {})
        )
        current_plan_node_id = str(conversation_plan.get("current_node_id", "")).strip() if isinstance(conversation_plan, dict) else ""
        current_plan_node_label = self._resolve_plan_node_label(conversation_plan, current_plan_node_id)
        facts = self._resolve_case_facts(state=state, proposal=proposal)
        customer_name = facts["customer_name"]
        case_id = facts["case_id"]
        overdue_amount = facts["overdue_amount"]
        prior_agent_response = str(memory_state.get("last_agent_response", "")).strip()
        turn_index = int(memory_state.get("turn_index", state.get("turn_index", 0)) or 0)
        is_opening_turn = turn_index <= 0
        verification_context = verification_guard_context if isinstance(verification_guard_context, dict) else {}

        system_prompt = (
            f"{self.system_prompt or ''}\n"
            "Your agent name is Alex.\n"
            "You are a bank collections assistant. Convert plan_proposal into the correct target-directed message.\n"
            "Allowed response_target values: customer, self.\n"
            "Never mention internal terms such as plan_proposal, plan, node, tool, or workflow.\n"
            "Use clear, polite, concise language and ask one concrete next-step question when needed.\n"
            "Never output placeholders such as [insert amount], [name], or [link].\n"
            "If amount context is provided, always include that numeric amount.\n"
            "Only use greeting/opening call introduction when is_opening_turn=true; avoid re-greeting on follow-up turns.\n"
            "When opening customer conversation, introduce yourself as Alex from collections.\n"
            "When response_target=customer: produce end-customer message only.\n"
            "When response_target=self: produce concise internal execution directive for next planner pass.\n"
            "If verification context indicates verification is incomplete, do not disclose dues/policy-sensitive details.\n"
            "If verification is incomplete, ask only for the missing fields listed in verification context.\n"
            "Return strict JSON only: {\"message\": \"...\", \"response_target\": \"customer|self\"}."
        ).strip()
        user_prompt = (
            f"User input: {user_input}\n"
            f"Customer name: {customer_name}\n"
            f"Case id: {case_id}\n"
            f"Overdue amount INR: {overdue_amount:.2f}\n"
            f"Response target: {response_target}\n"
            f"is_opening_turn: {json.dumps(is_opening_turn)}\n"
            f"Previous agent response (if any): {prior_agent_response}\n"
            f"Current plan step id: {current_plan_node_id}\n"
            f"Current plan step label: {current_plan_node_label}\n"
            f"Plan proposal: {json.dumps(proposal, ensure_ascii=True)}\n"
            f"Conversation plan graph: {json.dumps(conversation_plan, ensure_ascii=True)}\n"
            f"Verification context: {json.dumps(verification_context, ensure_ascii=True)}\n"
            f"Observation: {json.dumps(observation, ensure_ascii=True, default=str)}\n"
            "Generate only structured JSON output."
        )
        try:
            payload = StructuredOutputRunner(self.llm, max_retries=2).run(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=_ResponsePayload,
            )
        except Exception:
            return None
        response = str(payload.message).strip()
        response_target_payload = str(payload.response_target).strip().lower()
        if response_target_payload in {"customer", "self"}:
            proposal["target"] = response_target_payload
        if not response:
            return None
        return self._post_process_rendered_response(
            text=response,
            customer_name=customer_name,
            overdue_amount=overdue_amount,
            is_opening_turn=is_opening_turn,
            ensure_intro=bool(is_opening_turn and response_target == "customer"),
        )

    def _build_verification_guard_context(
        self,
        *,
        state: AgentState,
        memory_state: dict[str, Any],
        response_target: str,
        conversation_plan: dict[str, Any],
        customer_name: str,
    ) -> dict[str, Any] | None:
        if response_target != "customer":
            return None
        current_node_id = str(conversation_plan.get("current_node_id", "")).strip()
        if current_node_id != "verify_identity":
            return None
        if bool(memory_state.get("identity_verified", False)):
            return None

        collected = memory_state.get("verification_collected") if isinstance(memory_state.get("verification_collected"), dict) else {}
        required = memory_state.get("active_verification_required_fields")
        required_fields = [str(x).strip() for x in required if str(x).strip()] if isinstance(required, list) else []

        missing_labels: list[str] = []
        name_confirmed = bool(collected.get("name_confirmed"))
        if not name_confirmed:
            missing_labels.append("your full name")
        for field in required_fields:
            if field in {"name", "name_confirmed"}:
                continue
            if collected.get(field):
                continue
            missing_labels.append(self._verification_field_label(field))

        return {
            "verification_incomplete": True,
            "name_confirmed": bool(name_confirmed),
            "customer_name": customer_name,
            "required_fields": required_fields,
            "missing_field_labels": missing_labels,
            "missing_fields_human": self._join_human_list(missing_labels) if missing_labels else "",
            "guidance": (
                "Ask only for missing verification fields in a natural conversational way. "
                "Do not disclose dues or policy-sensitive details until verification completes."
            ),
            "current_plan_node_id": current_node_id,
            "collected_fields": collected,
        }

    def _fallback_render_from_proposal(self, *, proposal: dict[str, Any]) -> str:
        target = str(proposal.get("target", "customer")).strip().lower() or "customer"
        if target == "self":
            return "Continue internal planning using latest context and determine next execution step."

        intent = str(proposal.get("intent", "")).strip().lower()
        if intent == "generic_plan":
            draft = str(proposal.get("draft_response", "")).strip()
            if draft and not draft.lower().startswith("proposed plan for "):
                return draft

            context = proposal.get("context") if isinstance(proposal.get("context"), dict) else {}
            name = str(context.get("customer_name", "Customer")).strip() or "Customer"
            case_id = str(context.get("case_id", "COLL-1001")).strip() or "COLL-1001"
            overdue = float(context.get("overdue_amount", 0.0) or 0.0)
            observed_tool = str(context.get("observed_tool", "")).strip()
            plan_outline = str(proposal.get("plan_outline", "")).strip()
            next_actions = proposal.get("next_actions") if isinstance(proposal.get("next_actions"), list) else []

            if observed_tool == "case_fetch":
                return (
                    f"Hello {name}, this is Alex from collections. For case {case_id}, the current overdue amount is INR {overdue:.2f}. "
                    "Would you like to pay now, request an arrangement, or schedule a follow-up?"
                )

            if any(str(item).strip() == "complete_payment_flow" for item in next_actions):
                return (
                    f"Thank you {name}. Your current due amount is INR {overdue:.2f}. "
                    "I can generate a secure payment link right now. Would you like it via SMS or email?"
                )

            if plan_outline:
                return self._render_plan_outline(plan_outline)
            return "Please confirm how you would like to proceed with your dues."

        if intent == "outbound_opening":
            ctx = proposal.get("opening_context") if isinstance(proposal.get("opening_context"), dict) else {}
            name = str(ctx.get("customer_name", "Customer")).strip() or "Customer"
            case_id = str(ctx.get("case_id", "COLL-1001"))
            overdue = float(ctx.get("overdue_amount", 0.0))
            prior_signal = str(ctx.get("prior_signal", "")).strip()
            extra = f" {prior_signal}" if prior_signal else ""
            return (
                f"Hello {name}, this is Alex from the collections team calling regarding case {case_id}. "
                f"Our records show an overdue amount of INR {overdue:.2f}. "
                "I can help you clear dues now or discuss a suitable repayment arrangement."
                f"{extra} Are you available to proceed with payment today?"
            )

        if intent == "help_options":
            ctx = proposal.get("help_context") if isinstance(proposal.get("help_context"), dict) else {}
            name = str(ctx.get("customer_name", "Customer")).strip() or "Customer"
            case_id = str(ctx.get("case_id", "COLL-1001"))
            return (
                f"Sure {name}, I can help you in three ways for case {case_id}: "
                "1) pay dues now, 2) set a repayment arrangement if full payment is difficult, "
                "or 3) schedule a follow-up date. "
                "If you want, I can first verify your identity and then share exact due details."
            )

        if intent == "conversation_termination":
            return "Thank you. I am closing this conversation now."

        if intent == "loop_guard":
            return (
                "Please confirm one concrete next step: pay now, request a revised arrangement, or schedule follow-up."
            )

        if intent == "discount_recommendation":
            rec = (
                proposal.get("discount_recommendation", {}).get("recommended_offer", {})
                if isinstance(proposal.get("discount_recommendation"), dict)
                else {}
            )
            monthly = rec.get("monthly_emi")
            tenure = rec.get("tenure_months")
            if monthly is not None and tenure is not None:
                return (
                    f"I can offer a revised plan at INR {float(monthly):.2f} per month for {int(tenure)} months. "
                    "If this works for you, I will capture your promise-to-pay and schedule follow-up."
                )
            return "I have prepared revised discount and EMI options. Please confirm if you want to proceed."

        if intent == "case_not_found":
            return "I could not find an active dues case right now. Please confirm your case ID or customer ID."

        if intent == "case_snapshot":
            snap = proposal.get("case_snapshot") if isinstance(proposal.get("case_snapshot"), dict) else {}
            customer_name = str(snap.get("customer_name", "Customer")).strip() or "Customer"
            case_id = str(snap.get("case_id", "COLL-1001"))
            overdue = float(snap.get("overdue_amount", 0.0))
            emi = float(snap.get("emi_amount", 0.0))
            late = float(snap.get("late_fee", 0.0))
            dpd = int(snap.get("dpd", 0))
            return (
                f"Hello {customer_name}, this is Alex from the collections desk regarding case {case_id}. "
                f"Overdue amount is INR {overdue:.2f}, EMI is INR {emi:.2f}, late fee is INR {late:.2f}, "
                f"and the account is {dpd} days past due. "
                "Would you like to pay now, request a repayment arrangement, or schedule a follow-up?"
            )

        if intent == "plan_offer":
            offer = proposal.get("plan_offer") if isinstance(proposal.get("plan_offer"), dict) else {}
            months = offer.get("months")
            monthly = offer.get("monthly_amount")
            first_due = offer.get("first_due_date")
            if months is not None and monthly is not None:
                return (
                    f"I can offer a {int(months)}-month plan at INR {float(monthly):.2f} per month. "
                    f"First due date is {first_due}. Does this work for you?"
                )
            return "I can share a repayment plan option now. Would you like me to proceed?"

        if intent == "payment_collection_prompt":
            ctx = proposal.get("payment_context") if isinstance(proposal.get("payment_context"), dict) else {}
            name = str(ctx.get("customer_name", "Customer")).strip() or "Customer"
            amount = float(ctx.get("overdue_amount", 0.0) or 0.0)
            return (
                f"Thank you {name}. Your current due amount is INR {amount:.2f}. "
                "I can generate a secure payment link right now. "
                "Would you like the link via SMS or email?"
            )

        draft = str(proposal.get("draft_response", "")).strip()
        if draft:
            return draft

        plan_outline = str(proposal.get("plan_outline", "")).strip()
        if plan_outline:
            return self._render_plan_outline(plan_outline)

        return "Please confirm how you would like to proceed with your dues."

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
            "last4_pan": "the last 4 characters of your PAN",
            "zip": "your registered ZIP/pincode",
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

        payment_context = proposal.get("payment_context") if isinstance(proposal.get("payment_context"), dict) else {}
        if payment_context:
            customer_name = str(payment_context.get("customer_name", customer_name)).strip() or customer_name
            case_id = str(payment_context.get("case_id", case_id)).strip() or case_id
            overdue_amount = float(payment_context.get("overdue_amount", overdue_amount) or overdue_amount)

        return {
            "customer_name": customer_name,
            "case_id": case_id,
            "overdue_amount": overdue_amount,
        }

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
