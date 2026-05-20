"""Negotiation cognition node for collection agent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from agents.collection_agent.llm_structured import StructuredOutputRunner
from src.nodes.base import BaseGraphNode
from src.nodes.types import AgentState, NodeUpdate


class _HardshipContextPayload(BaseModel):
    hardship_detected: bool = False
    hardship_reason: str | None = None
    confidence: float = 0.0


class _NegotiationPayload(BaseModel):
    conversation_mode: str = "collections"
    negotiation_stage: str = "none"
    customer_payment_posture: str = "unknown"
    hardship_context: _HardshipContextPayload = Field(default_factory=_HardshipContextPayload)
    response_mode: str = "informational"
    active_dialogue_owner: str = "collections"
    reason: str | None = None


@dataclass(slots=True)
class NegotiationClassificationNode(BaseGraphNode):
    """Classifies persistent negotiation state before planning begins."""

    llm: Any | None = None
    system_prompt: str = ""
    user_prompt: str = ""
    strict_llm_mode: bool = True

    _ALLOWED_CONVERSATION_MODES = {
        "collections",
        "hardship_negotiation",
        "promise_capture",
        "verification",
        "escalation",
    }
    _ALLOWED_NEGOTIATION_STAGES = {
        "none",
        "discovering_hardship",
        "assessing_capacity",
        "evaluating_options",
        "negotiating_plan",
        "confirming_commitment",
        "awaiting_customer_decision",
    }
    _ALLOWED_PAYMENT_POSTURES = {
        "unknown",
        "can_pay_full",
        "needs_arrangement",
        "cannot_pay_now",
        "avoiding_commitment",
    }
    _ALLOWED_RESPONSE_MODES = {
        "informational",
        "empathetic",
        "negotiation",
        "compliance",
        "firm",
    }
    _ALLOWED_DIALOGUE_OWNERS = {
        "verification",
        "collections",
        "plan_proposal",
        "promise_capture",
    }

    def execute(self, state: AgentState) -> NodeUpdate:
        self._record_llm_usage(state, node_name="negotiation_classification")
        memory = state.get("memory")
        memory_state = dict(getattr(memory, "state", {})) if memory is not None else {}
        identity_verified = bool(state.get("identity_verified", memory_state.get("identity_verified", False)))
        prior = self._prior_state(state=state, memory_state=memory_state, identity_verified=identity_verified)

        recent_conversation = self._recent_conversation(state=state, memory_state=memory_state)
        prompt_debug = {
            "prompt": None,
            "system_prompt": self.system_prompt or None,
            "llm_response": None,
            "llm_error": None,
        }
        payload: _NegotiationPayload | None = None

        if self.llm is not None and self.system_prompt.strip() and self.user_prompt.strip():
            try:
                user_prompt = self._render_prompt(
                    self.user_prompt,
                    {
                        "user_input": str(state.get("user_input", "")),
                        "recent_conversation_json": json.dumps(recent_conversation, ensure_ascii=True),
                        "memory_state_json": json.dumps(self._compact_memory_state(memory_state), ensure_ascii=True, default=str),
                        "existing_negotiation_state_json": json.dumps(prior, ensure_ascii=True, default=str),
                        "extracted_entities_json": json.dumps(state.get("extracted_entities", {}), ensure_ascii=True, default=str),
                        "extracted_entities_turn_json": json.dumps(
                            state.get("extracted_entities_turn", {}),
                            ensure_ascii=True,
                            default=str,
                        ),
                        "verification_state_json": json.dumps(
                            {
                                "identity_verified": identity_verified,
                                "verification_missing_fields": state.get("verification_missing_fields", []),
                                "verification_verified_fields": state.get("verification_verified_fields", []),
                            },
                            ensure_ascii=True,
                            default=str,
                        ),
                    },
                )
                prompt_debug["prompt"] = user_prompt
                payload = StructuredOutputRunner(self.llm, max_retries=2).run(
                    system_prompt=self.system_prompt,
                    user_prompt=user_prompt,
                    schema=_NegotiationPayload,
                )
                prompt_debug["llm_response"] = payload.model_dump(mode="json")
            except Exception as exc:
                prompt_debug["llm_error"] = str(exc)
                if self.strict_llm_mode:
                    raise

        merged = self._merge_with_prior_state(
            prior=prior,
            payload=(payload.model_dump(mode="json") if payload is not None else None),
            user_input=str(state.get("user_input", "")),
            extracted_entities_turn=(
                dict(state.get("extracted_entities_turn", {}))
                if isinstance(state.get("extracted_entities_turn"), dict)
                else {}
            ),
            identity_verified=identity_verified,
        )

        update: NodeUpdate = {
            "negotiation_classification": {
                "conversation_mode": merged["conversation_mode"],
                "negotiation_stage": merged["negotiation_stage"],
                "customer_payment_posture": merged["customer_payment_posture"],
                "hardship_context": dict(merged["hardship_context"]),
                "response_mode": merged["response_mode"],
                "active_dialogue_owner": merged["active_dialogue_owner"],
            },
            "conversation_mode": merged["conversation_mode"],
            "negotiation_stage": merged["negotiation_stage"],
            "customer_payment_posture": merged["customer_payment_posture"],
            "hardship_context": dict(merged["hardship_context"]),
            "response_mode": merged["response_mode"],
            "active_dialogue_owner": merged["active_dialogue_owner"],
            "prompt": prompt_debug.get("prompt"),
            "system_prompt": prompt_debug.get("system_prompt"),
            "llm_response": prompt_debug.get("llm_response"),
            "llm_error": prompt_debug.get("llm_error"),
        }
        if memory is not None:
            memory.set_state(
                conversation_mode=merged["conversation_mode"],
                negotiation_stage=merged["negotiation_stage"],
                customer_payment_posture=merged["customer_payment_posture"],
                hardship_context=dict(merged["hardship_context"]),
                response_mode=merged["response_mode"],
                active_dialogue_owner=merged["active_dialogue_owner"],
                mode=("hardship_negotiation" if merged["conversation_mode"] == "hardship_negotiation" else "strict_collections"),
                hardship_reason=(
                    str(merged["hardship_context"].get("hardship_reason", "")).strip()
                    if bool(merged["hardship_context"].get("hardship_detected", False))
                    else None
                ),
            )
        return update

    def _merge_with_prior_state(
        self,
        *,
        prior: dict[str, Any],
        payload: dict[str, Any] | None,
        user_input: str,
        extracted_entities_turn: dict[str, Any],
        identity_verified: bool,
    ) -> dict[str, Any]:
        fallback = self._fallback_classification(
            prior=prior,
            user_input=user_input,
            extracted_entities_turn=extracted_entities_turn,
            identity_verified=identity_verified,
        )
        raw = payload if isinstance(payload, dict) else fallback
        hardship_payload = raw.get("hardship_context") if isinstance(raw.get("hardship_context"), dict) else {}

        merged = {
            "conversation_mode": self._normalize_choice(
                raw.get("conversation_mode"),
                allowed=self._ALLOWED_CONVERSATION_MODES,
                default=str(prior.get("conversation_mode", "collections")),
            ),
            "negotiation_stage": self._normalize_choice(
                raw.get("negotiation_stage"),
                allowed=self._ALLOWED_NEGOTIATION_STAGES,
                default=str(prior.get("negotiation_stage", "none")),
            ),
            "customer_payment_posture": self._normalize_choice(
                raw.get("customer_payment_posture"),
                allowed=self._ALLOWED_PAYMENT_POSTURES,
                default=str(prior.get("customer_payment_posture", "unknown")),
            ),
            "hardship_context": {
                "hardship_detected": bool(hardship_payload.get("hardship_detected", False)),
                "hardship_reason": self._normalize_optional_text(hardship_payload.get("hardship_reason")),
                "confidence": self._normalize_confidence(hardship_payload.get("confidence")),
            },
            "response_mode": self._normalize_choice(
                raw.get("response_mode"),
                allowed=self._ALLOWED_RESPONSE_MODES,
                default=str(prior.get("response_mode", "informational")),
            ),
            "active_dialogue_owner": self._normalize_choice(
                raw.get("active_dialogue_owner"),
                allowed=self._ALLOWED_DIALOGUE_OWNERS,
                default=str(prior.get("active_dialogue_owner", "collections")),
            ),
        }

        prior_hardship = (
            dict(prior.get("hardship_context", {}))
            if isinstance(prior.get("hardship_context"), dict)
            else {}
        )
        if bool(prior_hardship.get("hardship_detected", False)) and not bool(merged["hardship_context"]["hardship_detected"]):
            merged["hardship_context"] = {
                "hardship_detected": True,
                "hardship_reason": self._normalize_optional_text(
                    prior_hardship.get("hardship_reason") or merged["hardship_context"]["hardship_reason"]
                ),
                "confidence": max(
                    self._normalize_confidence(prior_hardship.get("confidence")),
                    float(merged["hardship_context"]["confidence"]),
                ),
            }

        if bool(merged["hardship_context"]["hardship_detected"]):
            merged["conversation_mode"] = "hardship_negotiation"
            if merged["negotiation_stage"] in {"none", ""}:
                merged["negotiation_stage"] = str(prior.get("negotiation_stage", "discovering_hardship"))
                if merged["negotiation_stage"] == "none":
                    merged["negotiation_stage"] = "discovering_hardship"
            if merged["customer_payment_posture"] == "unknown":
                merged["customer_payment_posture"] = "needs_arrangement"
            if merged["response_mode"] == "informational":
                merged["response_mode"] = "empathetic"
            merged["active_dialogue_owner"] = "plan_proposal"
        elif not identity_verified:
            merged["conversation_mode"] = "verification"
            merged["response_mode"] = "compliance"
            if merged["active_dialogue_owner"] not in {"plan_proposal"}:
                merged["active_dialogue_owner"] = "verification"
            if merged["negotiation_stage"] == "none":
                merged["negotiation_stage"] = str(prior.get("negotiation_stage", "none"))
        else:
            if merged["conversation_mode"] == "verification":
                merged["conversation_mode"] = "collections"
            if merged["active_dialogue_owner"] == "verification":
                merged["active_dialogue_owner"] = "collections"

        if merged["conversation_mode"] == "promise_capture":
            merged["active_dialogue_owner"] = "promise_capture"
            merged["response_mode"] = "negotiation"
        elif merged["conversation_mode"] == "escalation":
            merged["response_mode"] = "firm"

        if (
            prior.get("conversation_mode") == "hardship_negotiation"
            and merged["conversation_mode"] == "collections"
            and bool(prior_hardship.get("hardship_detected", False))
        ):
            merged["conversation_mode"] = "hardship_negotiation"
            merged["active_dialogue_owner"] = "plan_proposal"
            if merged["response_mode"] == "informational":
                merged["response_mode"] = "negotiation"

        return merged

    def _fallback_classification(
        self,
        *,
        prior: dict[str, Any],
        user_input: str,
        extracted_entities_turn: dict[str, Any],
        identity_verified: bool,
    ) -> dict[str, Any]:
        lowered = str(user_input or "").lower()
        hardship_reason = self._detect_hardship_reason(lowered)
        hardship_detected = hardship_reason is not None or bool(
            prior.get("hardship_context", {}).get("hardship_detected", False)
            if isinstance(prior.get("hardship_context"), dict)
            else False
        )

        posture = str(prior.get("customer_payment_posture", "unknown"))
        if any(token in lowered for token in ["pay full", "pay in full", "clear all dues", "settle now"]):
            posture = "can_pay_full"
        elif any(token in lowered for token in ["later", "not now", "next month", "salary not credited"]):
            posture = "cannot_pay_now"
        elif hardship_detected or any(token in lowered for token in ["installment", "arrangement", "plan", "emi"]):
            posture = "needs_arrangement"
        elif any(token in lowered for token in ["not interested", "stop calling", "will see", "can't commit"]):
            posture = "avoiding_commitment"

        stage = str(prior.get("negotiation_stage", "none"))
        amount_present = any(
            str(key).strip().lower() in {"amount", "emi_amount", "promised_amount"}
            and str(value).strip()
            for key, value in extracted_entities_turn.items()
        )
        if hardship_detected and stage == "none":
            stage = "discovering_hardship"
        if hardship_detected and amount_present:
            stage = "negotiating_plan"
        elif hardship_detected and any(token in lowered for token in ["installment", "emi", "monthly", "how much can pay"]):
            stage = "assessing_capacity"
        elif posture == "can_pay_full":
            stage = "confirming_commitment"

        if hardship_detected:
            return {
                "conversation_mode": "hardship_negotiation",
                "negotiation_stage": stage or "discovering_hardship",
                "customer_payment_posture": ("needs_arrangement" if posture == "unknown" else posture),
                "hardship_context": {
                    "hardship_detected": True,
                    "hardship_reason": hardship_reason
                    or self._normalize_optional_text(
                        prior.get("hardship_context", {}).get("hardship_reason")
                        if isinstance(prior.get("hardship_context"), dict)
                        else None
                    ),
                    "confidence": 0.92 if hardship_reason else 0.75,
                },
                "response_mode": "empathetic",
                "active_dialogue_owner": "plan_proposal",
            }

        if not identity_verified:
            return {
                "conversation_mode": "verification",
                "negotiation_stage": stage or "none",
                "customer_payment_posture": posture,
                "hardship_context": {
                    "hardship_detected": False,
                    "hardship_reason": None,
                    "confidence": 0.0,
                },
                "response_mode": "compliance",
                "active_dialogue_owner": "verification",
            }

        return {
            "conversation_mode": "collections",
            "negotiation_stage": stage or "none",
            "customer_payment_posture": posture,
            "hardship_context": {
                "hardship_detected": False,
                "hardship_reason": None,
                "confidence": 0.0,
            },
            "response_mode": ("negotiation" if posture == "needs_arrangement" else "informational"),
            "active_dialogue_owner": ("plan_proposal" if posture == "needs_arrangement" else "collections"),
        }

    @staticmethod
    def _prior_state(*, state: AgentState, memory_state: dict[str, Any], identity_verified: bool) -> dict[str, Any]:
        hardship_context = (
            dict(state.get("hardship_context", {}))
            if isinstance(state.get("hardship_context"), dict)
            else (
                dict(memory_state.get("hardship_context", {}))
                if isinstance(memory_state.get("hardship_context"), dict)
                else {}
            )
        )
        default_mode = "verification" if not identity_verified else "collections"
        return {
            "conversation_mode": str(
                state.get("conversation_mode", memory_state.get("conversation_mode", default_mode))
            ).strip()
            or default_mode,
            "negotiation_stage": str(
                state.get("negotiation_stage", memory_state.get("negotiation_stage", "none"))
            ).strip()
            or "none",
            "customer_payment_posture": str(
                state.get("customer_payment_posture", memory_state.get("customer_payment_posture", "unknown"))
            ).strip()
            or "unknown",
            "hardship_context": {
                "hardship_detected": bool(hardship_context.get("hardship_detected", False)),
                "hardship_reason": str(hardship_context.get("hardship_reason", "")).strip() or None,
                "confidence": float(hardship_context.get("confidence", 0.0) or 0.0),
            },
            "response_mode": str(state.get("response_mode", memory_state.get("response_mode", "informational"))).strip()
            or "informational",
            "active_dialogue_owner": str(
                state.get("active_dialogue_owner", memory_state.get("active_dialogue_owner", default_mode))
            ).strip()
            or default_mode,
        }

    @staticmethod
    def _recent_conversation(*, state: AgentState, memory_state: dict[str, Any]) -> list[dict[str, str]]:
        conversation_history = (
            state.get("conversation_history")
            if isinstance(state.get("conversation_history"), list)
            else (
                memory_state.get("conversation_history")
                if isinstance(memory_state.get("conversation_history"), list)
                else []
            )
        )
        recent: list[dict[str, str]] = []
        for item in conversation_history[-8:]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if role and content:
                recent.append({"role": role, "content": content})
        return recent

    @staticmethod
    def _compact_memory_state(memory_state: dict[str, Any]) -> dict[str, Any]:
        keep = {
            "mode",
            "conversation_mode",
            "negotiation_stage",
            "customer_payment_posture",
            "hardship_context",
            "response_mode",
            "active_dialogue_owner",
            "identity_verified",
            "verification_missing_fields",
            "verification_verified_fields",
            "active_case_id",
            "active_user_id",
            "active_customer_name",
            "last_agent_response",
        }
        return {key: memory_state.get(key) for key in keep if key in memory_state}

    @staticmethod
    def _render_prompt(template: str, values: dict[str, Any]) -> str:
        rendered = str(template or "")
        for key, value in values.items():
            rendered = rendered.replace(f"{{{key}}}", str(value))
        return rendered

    @staticmethod
    def _normalize_choice(value: Any, *, allowed: set[str], default: str) -> str:
        text = str(value or "").strip().lower()
        return text if text in allowed else default

    @staticmethod
    def _normalize_optional_text(value: Any) -> str | None:
        text = str(value or "").strip().lower()
        return text or None

    @staticmethod
    def _normalize_confidence(value: Any) -> float:
        try:
            confidence = float(value)
        except Exception:
            confidence = 0.0
        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _detect_hardship_reason(lowered_text: str) -> str | None:
        detectors = {
            "job_loss": ["lost my job", "lost job", "job loss", "unemployed", "laid off"],
            "salary_delay": ["salary delay", "salary not credited", "salary late"],
            "medical_emergency": ["medical emergency", "hospital", "treatment", "medical issue"],
            "family_emergency": ["family emergency", "family issue", "family problem"],
            "reduced_income": ["reduced income", "income loss", "salary cut", "less income"],
            "business_loss": ["business losses", "business loss", "loss in business"],
            "cashflow_issue": ["cashflow", "cash flow", "financial issues", "temporary issue", "temporary cashflow"],
        }
        for reason, phrases in detectors.items():
            if any(phrase in lowered_text for phrase in phrases):
                return reason
        return None
