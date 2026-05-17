"""Collection planner built on shared PlannerNode contract."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel, Field

from agents.collection_agent.llm_structured import StructuredOutputRunner
from agents.collection_agent.tools.common import followup_decision_from_observation, observation_to_response
from src.nodes.planner_node import PlannerNode


class _PlannerDecisionPayload(BaseModel):
    intent: str = Field(default="unknown")
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    respond_text: str | None = None
    mode_update: str | None = None
    reason: str | None = None


@dataclass(slots=True)
class CollectionPlanner(PlannerNode):
    """LLM-assisted planner with deterministic fallback for collection demo flows."""

    llm: Any | None = None
    intent_system_prompt: str | None = None
    intent_user_prompt: str | None = None
    require_llm: bool = True
    allow_rule_fallback: bool = False

    def plan(
        self,
        *,
        user_input: str,
        memory: Any | None = None,
        observation: dict[str, Any] | None = None,
        memory_context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        available_tools: list[Any] | None = None,
    ) -> Any:
        del memory_context, system_prompt, user_prompt
        text = user_input.strip()
        lowered = text.lower()
        state = dict(getattr(memory, "state", {})) if memory is not None else {}
        mode = str(state.get("mode", "strict_collections"))

        if self.require_llm and self.llm is None:
            if self.allow_rule_fallback:
                return self._rule_fallback(user_input=text, lowered=lowered, mode=mode, memory=memory, state=state)
            return self._respond(
                "I am temporarily unable to run advanced planning right now. "
                "Please share your case ID, or tell me if you want to pay now, request assistance, or schedule follow-up."
            )

        if observation:
            obs = (
                observation.get("tool_phase")
                if isinstance(observation, dict) and isinstance(observation.get("tool_phase"), dict)
                else observation
            )
            followup = followup_decision_from_observation(obs if isinstance(obs, dict) else {})
            if followup is not None:
                return followup
            return self._response_from_observation(obs if isinstance(obs, dict) else {}, mode, memory)

        llm_decision = self._plan_with_llm(
            user_input=text,
            lowered=lowered,
            mode=mode,
            memory=memory,
            state=state,
            available_tools=available_tools,
        )
        if llm_decision is not None:
            return llm_decision

        if not self.allow_rule_fallback:
            return self._respond(
                "I am temporarily unable to complete that planning step right now. "
                "Please confirm one next step: pay now, request arrangement, or schedule follow-up."
            )

        return self._rule_fallback(user_input=text, lowered=lowered, mode=mode, memory=memory, state=state)

    def _plan_with_llm(
        self,
        *,
        user_input: str,
        lowered: str,
        mode: str,
        memory: Any | None,
        state: dict[str, Any],
        available_tools: list[Any] | None,
    ) -> Any | None:
        if self.llm is None:
            return None
        try:
            system_prompt = self.intent_system_prompt or (
                "You are a collections intent and action planner. Return JSON only."
            )
            user_prompt = self._render_intent_prompt(
                template=self.intent_user_prompt,
                user_input=user_input,
                mode=mode,
                state=state,
                available_tools=available_tools,
            )
            payload = StructuredOutputRunner(self.llm, max_retries=2).run(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=_PlannerDecisionPayload,
            ).model_dump(mode="json")

            intent = str(payload.get("intent", "")).strip().lower()
            tool_name = payload.get("tool_name")
            respond_text = payload.get("respond_text")
            mode_update = payload.get("mode_update")

            if isinstance(mode_update, str) and mode_update in {"strict_collections", "hardship_negotiation"} and memory is not None:
                memory.set_state(mode=mode_update)

            if intent == "off_topic":
                return self._respond(
                    "I am a collections assistant and can only help with your payment account, dues, or repayment plan questions."
                )

            if intent == "hardship_request" and memory is not None:
                extracted = self._extract_args(user_input)
                memory.set_state(
                    mode="hardship_negotiation",
                    hardship_reason="job_loss",
                    active_case_id=str(extracted.get("case_id", state.get("active_case_id", "COLL-1001"))),
                )
                args = self._norm("offer_eligibility", extracted)
                return self._tool("offer_eligibility", args)

            if intent == "plan_accept" and mode == "hardship_negotiation":
                current_plan = state.get("current_plan") or {}
                args = {
                    "case_id": str(current_plan.get("case_id", state.get("active_case_id", "COLL-1001"))),
                    "promised_date": str(
                        current_plan.get("first_due_date", (datetime.now(UTC) + timedelta(days=7)).date().isoformat())
                    ),
                    "promised_amount": float(current_plan.get("monthly_amount", 1000.0)),
                    "channel": str(state.get("active_channel", "voice")),
                }
                return self._tool("promise_capture", args)

            if isinstance(tool_name, str) and tool_name.strip():
                name = tool_name.strip()
                llm_args = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
                extracted_args = self._extract_args(user_input)
                merged_args = {**extracted_args, **llm_args}
                return self._tool(name, self._norm(name, merged_args))

            if isinstance(respond_text, str) and respond_text.strip():
                return self._respond(respond_text.strip())

            mapped = self._intent_to_tool(intent=intent, lowered=lowered, user_input=user_input, memory=memory)
            if mapped is not None:
                return mapped
        except Exception:
            return None
        return None

    def _intent_to_tool(self, *, intent: str, lowered: str, user_input: str, memory: Any | None) -> Any | None:
        del lowered
        args = self._extract_args(user_input)
        if memory is not None:
            args.setdefault("case_id", str(memory.state.get("active_case_id", "COLL-1001")))

        mapping = {
            "verify_identity": "customer_verify",
            "policy_lookup": "loan_policy_lookup",
            "offer_check": "offer_eligibility",
            "payment_link": "payment_link_create",
            "promise_capture": "promise_capture",
            "human_escalation": "human_escalation",
        }
        tool_name = mapping.get(intent)
        if tool_name is None:
            return None
        return self._tool(tool_name, self._norm(tool_name, args))

    def _render_intent_prompt(
        self,
        *,
        template: str | None,
        user_input: str,
        mode: str,
        state: dict[str, Any],
        available_tools: list[Any] | None,
    ) -> str:
        default_template = (
            "User input: {user_input}\n"
            "Mode: {mode}\n"
            "State: {state}\n"
            "Available tools: {available_tools}\n"
            "Return strict JSON with keys: intent, tool_name, arguments, respond_text, mode_update, reason."
        )
        tmpl = template or default_template
        values = {
            "user_input": user_input,
            "mode": mode,
            "state": json.dumps(state, default=str, ensure_ascii=True),
            "available_tools": json.dumps(available_tools or [], default=str, ensure_ascii=True),
        }
        rendered = tmpl
        for key, value in values.items():
            rendered = rendered.replace(f"{{{key}}}", value)
        return rendered

    def _rule_fallback(self, *, user_input: str, lowered: str, mode: str, memory: Any | None, state: dict[str, Any]) -> Any:
        if self._is_off_topic(lowered):
            return self._respond(
                "I am a collections assistant and can only help with your payment account, dues, or repayment plan questions."
            )

        if self._is_hardship_signal(lowered):
            if memory is not None:
                args = self._extract_args(user_input)
                memory.set_state(
                    mode="hardship_negotiation",
                    hardship_reason="job_loss",
                    active_case_id=str(args.get("case_id", memory.state.get("active_case_id", "COLL-1001"))),
                )
            return self._tool("offer_eligibility", self._norm("offer_eligibility", self._extract_args(user_input)))

        if mode == "hardship_negotiation" and self._is_plan_acceptance(lowered):
            current_plan = state.get("current_plan") or {}
            args = {
                "case_id": str(current_plan.get("case_id", state.get("active_case_id", "COLL-1001"))),
                "promised_date": str(
                    current_plan.get("first_due_date", (datetime.now(UTC) + timedelta(days=7)).date().isoformat())
                ),
                "promised_amount": float(current_plan.get("monthly_amount", 1000.0)),
                "channel": str(state.get("active_channel", "voice")),
            }
            return self._tool("promise_capture", args)

        if "verify" in lowered or "zip" in lowered or "dob" in lowered:
            return self._tool("customer_verify", self._norm("customer_verify", self._extract_args(user_input)))

        if "policy" in lowered:
            return self._tool("loan_policy_lookup", self._norm("loan_policy_lookup", self._extract_args(user_input)))

        return self._respond(
            "Please choose one: verify customer, request assistance, or create payment link."
        )

    def _response_from_observation(self, observation: dict[str, Any], mode: str, memory: Any | None) -> Any:
        tool_name = str(observation.get("tool_name", ""))
        output = observation.get("output") if isinstance(observation.get("output"), dict) else {}

        if tool_name == "channel_switch":
            if memory is not None:
                memory.set_state(active_channel=output.get("to_channel", "voice"))
            msg = (
                f"Switch confirmed from {output.get('from_channel')} to {output.get('to_channel')}. "
                f"Context carried: {output.get('carried_context_summary')}"
            )
            return self._respond(msg)

        if tool_name == "plan_propose":
            msg = (
                f"I can offer a {output.get('months')}-month plan at {output.get('monthly_amount')} per month. "
                f"First due date is {output.get('first_due_date')}. Does this work for you?"
            )
            return self._respond(msg)

        if tool_name == "pay_by_phone_collect":
            status = str(output.get("status", ""))
            if status == "success":
                return self._respond(f"Payment collected successfully. Receipt: {output.get('receipt_reference')}.")
            if status == "partial":
                return self._respond(
                    f"Partial payment collected: {output.get('collected_amount')}. We can set a plan for remaining dues."
                )
            return self._respond("Payment failed in demo flow. Would you like to try a payment plan?")

        if tool_name == "followup_schedule":
            return self._tool(
                "disposition_update",
                {
                    "case_id": output.get("case_id", "COLL-1001"),
                    "disposition_code": "plan_accepted_followup_scheduled",
                    "notes": "Demo flow: plan accepted and follow-up scheduled.",
                },
            )

        if tool_name == "disposition_update":
            return self._respond("Your plan is confirmed and logged. A follow-up reminder has been scheduled.")

        if mode == "strict_collections" and tool_name == "contact_attempt":
            return self._respond(
                "Reminder sent. If you can pay now, I can process pay-by-phone. If you need assistance, say 'assist'."
            )

        return self._respond(observation_to_response(observation))

    @staticmethod
    def _tool(tool_name: str, arguments: dict[str, Any]) -> Any:
        return SimpleNamespace(
            thought=f"Executing {tool_name}.",
            tool_call=SimpleNamespace(tool_name=tool_name, arguments=arguments),
            respond_directly=False,
            response_text=None,
            done=False,
        )

    @staticmethod
    def _respond(text: str) -> Any:
        return SimpleNamespace(
            thought="Responding directly.",
            tool_call=None,
            respond_directly=True,
            response_text=text,
            done=True,
        )

    @staticmethod
    def _is_hardship_signal(text: str) -> bool:
        keys = [
            "assist",
            "lost my job",
            "health issue",
            "hardship",
            "income reduction",
            "cannot pay",
            "can't pay",
            "discount",
            "waiver",
            "lower emi",
            "reduce emi",
            "settlement",
        ]
        return any(key in text for key in keys)

    @staticmethod
    def _is_plan_acceptance(text: str) -> bool:
        keys = ["that will work", "works for me", "accept", "yes", "okay"]
        return any(key in text for key in keys)

    @staticmethod
    def _is_off_topic(text: str) -> bool:
        keys = ["super bowl", "netflix", "young sheldon", "weather", "movie"]
        return any(key in text for key in keys)

    @staticmethod
    def _extract_args(text: str) -> dict[str, Any]:
        pairs = re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^\s,]+)", text)
        args: dict[str, Any] = {}
        for key, value in pairs:
            clean = value.strip().strip("'\"")
            low = clean.lower()
            if low in {"true", "false"}:
                args[key] = low == "true"
            elif re.fullmatch(r"-?\d+", clean):
                args[key] = int(clean)
            elif re.fullmatch(r"-?\d+\.\d+", clean):
                args[key] = float(clean)
            else:
                args[key] = clean

        case_match = re.search(r"(COLL-\d+|CASE-\d+)", text, re.IGNORECASE)
        if case_match and "case_id" not in args:
            args["case_id"] = case_match.group(1).upper()

        amount_match = re.search(r"(?:\$|inr\s*)?(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        if amount_match and "amount" not in args:
            args["amount"] = float(amount_match.group(1))
        return args

    @staticmethod
    def _norm(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(args)
        now = datetime.now(UTC)
        normalized.setdefault("case_id", "COLL-1001")
        if tool_name == "case_fetch":
            normalized.setdefault("limit", 20)
        elif tool_name == "contact_attempt":
            normalized.setdefault("channel", "sms")
            normalized.setdefault("reached", False)
        elif tool_name == "customer_verify":
            normalized.setdefault("challenge_answers", {})
        elif tool_name == "dues_explain_build":
            normalized.setdefault("locale", "en-IN")
        elif tool_name == "offer_eligibility":
            normalized.setdefault("hardship_flag", True)
        elif tool_name == "pay_by_phone_collect":
            normalized.setdefault("amount", 6000.0)
            normalized.setdefault("consent_confirmed", True)
            normalized.setdefault("simulate_status", "success")
        elif tool_name == "channel_switch":
            normalized.setdefault("from_channel", "sms")
            normalized.setdefault("to_channel", "voice")
        elif tool_name == "promise_capture":
            normalized.setdefault("promised_date", (now + timedelta(days=7)).date().isoformat())
            normalized.setdefault("promised_amount", 1000.0)
            normalized.setdefault("channel", "voice")
        return normalized
