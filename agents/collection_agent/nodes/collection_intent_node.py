"""Collection-specific intent node with namespaced state output keys."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from pydantic import BaseModel, Field

from agents.collection_agent.llm_structured import StructuredOutputRunner
from src.nodes.intent_node import IntentNode
from src.nodes.types import AgentState, NodeUpdate


class _IntentPayload(BaseModel):
    intent: str
    confidence: float = Field(default=0.0)
    reason: str | None = None


@dataclass(slots=True)
class CollectionIntentNode(IntentNode):
    """Writes and routes intent payloads using a dedicated state key."""

    output_key: str = "intent"
    allow_deterministic_fallback: bool = False
    allow_rate_limit_fallback: bool = True
    enable_relevance_guard: bool = True

    def classify(
        self,
        *,
        user_input: str,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        intent_labels: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        labels = intent_labels if intent_labels is not None else self.intent_labels
        rendered_system_prompt = system_prompt if system_prompt is not None else self.system_prompt
        rendered_user_prompt = self._render_user_prompt(
            user_prompt=user_prompt if user_prompt is not None else (self.user_prompt or "{user_input}"),
            user_input=user_input,
            intent_labels=labels,
        )
        context_block = self._build_intent_context_block(context or {})
        if context_block:
            rendered_user_prompt = f"{rendered_user_prompt}\n\n{context_block}"
        debug_payload: dict[str, Any] = {
            "prompt": rendered_user_prompt,
            "system_prompt": rendered_system_prompt or None,
            "llm_response": None,
            "llm_error": None,
            "llm_status": "not_called",
        }

        if self.llm is None:
            deterministic = self._deterministic_classify(user_input=user_input, labels=labels)
            debug_payload["llm_status"] = "fallback_no_llm"
            deterministic["_debug"] = debug_payload
            return deterministic

        try:
            runner = StructuredOutputRunner(self.llm, max_retries=2)
            payload = runner.run(
                system_prompt=rendered_system_prompt or "",
                user_prompt=rendered_user_prompt,
                schema=_IntentPayload,
            )
            debug_payload["llm_status"] = "used_structured_output"
            debug_payload["llm_response"] = payload.model_dump(mode="json")
            intent_name = self._normalize_intent_name(payload.intent)
            if labels and intent_name not in {self._normalize_intent_name(v) for v in labels}:
                raise ValueError(f"Intent `{payload.intent}` is not in allowed labels: {labels}")
            return {
                "intent": intent_name or self.default_intent,
                "confidence": float(payload.confidence),
                "reason": payload.reason,
                "_debug": debug_payload,
            }
        except Exception as exc:
            error_text = str(exc)
            debug_payload["llm_error"] = error_text
            is_rate_limit = self._is_provider_rate_limit_error(error_text)
            if self.allow_rate_limit_fallback and is_rate_limit:
                debug_payload["llm_status"] = "fallback_rate_limit_default"
                return {
                    "intent": self.default_intent,
                    "confidence": 0.0,
                    "reason": "LLM provider rate limit hit; using default route-safe intent fallback.",
                    "_debug": {
                        **debug_payload,
                        "fallback_reason": "provider_rate_limit",
                    },
                }
            if is_rate_limit and not self.allow_rate_limit_fallback:
                raise RuntimeError(
                    "CollectionIntentNode was rate-limited by provider and rate-limit fallback is disabled. "
                    f"Underlying error: {exc}"
                ) from exc
            if not self.allow_deterministic_fallback:
                raise RuntimeError(
                    "CollectionIntentNode failed to produce structured LLM intent output. "
                    f"Fallback disabled. Error: {exc}"
                ) from exc
            deterministic = self._deterministic_classify(user_input=user_input, labels=labels)
            debug_payload["llm_status"] = "fallback_deterministic_after_error"
            deterministic["_debug"] = debug_payload
            return deterministic

    def execute(self, state: AgentState) -> NodeUpdate:
        self._record_llm_usage(state, node_name="intent")
        if self.llm is None and not self.allow_deterministic_fallback:
            raise RuntimeError(
                "CollectionIntentNode requires an active LLM. "
                "Deterministic intent fallback is disabled for collection agent."
            )
        context = self._build_context_for_intent(state)
        pre_rule = self._apply_node_specific_pre_rule(state=state, context=context)
        if pre_rule is None:
            pre_rule = {}
        if not isinstance(pre_rule, dict):
            raise TypeError("_apply_node_specific_pre_rule must return dict or None.")

        if pre_rule.get("context_updates") and isinstance(pre_rule.get("context_updates"), dict):
            context = {**context, **dict(pre_rule["context_updates"])}

        if pre_rule.get("skip_llm", False) and isinstance(pre_rule.get("intent"), dict):
            intent = dict(pre_rule["intent"])
            intent["_debug"] = {
                "prompt": None,
                "system_prompt": None,
                "llm_response": None,
                "llm_error": None,
                "llm_status": "skipped_by_pre_rule",
                "pre_rule_reason": pre_rule.get("reason"),
            }
        else:
            intent = self.classify(
                user_input=state["user_input"],
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
                context=context,
            )
        intent = self._apply_node_specific_intent_override(intent=intent, state=state, context=context)
        pre_guard_debug = intent.get("_debug") if isinstance(intent, dict) else {}
        intent = self._apply_relevance_guard(
            intent=intent,
            user_input=str(state.get("user_input", "")),
            context=context,
        )
        if isinstance(intent, dict) and "_debug" not in intent and isinstance(pre_guard_debug, dict):
            intent["_debug"] = pre_guard_debug
        debug_payload = intent.pop("_debug", {}) if isinstance(intent, dict) else {}
        update: NodeUpdate = {
            self.output_key: intent,
            # Compatibility channel for shared routing assumptions in existing
            # graph/runtime reducers. This can be removed after full migration.
            "intent": intent,
            "prompt": debug_payload.get("prompt"),
            "system_prompt": debug_payload.get("system_prompt"),
            "llm_response": debug_payload.get("llm_response"),
            "llm_error": debug_payload.get("llm_error"),
            "llm_status": debug_payload.get("llm_status"),
        }
        if isinstance(debug_payload, dict) and debug_payload.get("fallback_reason"):
            update["fallback_reason"] = debug_payload.get("fallback_reason")
        intent_name = self._normalize_intent_name(intent.get("intent") if isinstance(intent, dict) else None)
        mapped_response = self._lookup_mapped_value(self.response_map, intent_name)
        if mapped_response is not None:
            update["response"] = mapped_response
        return update

    def route(self, state: AgentState) -> str:
        intent_payload = state.get(self.output_key)
        if intent_payload is None:
            intent_payload = state.get("intent")
        intent_name = None
        if isinstance(intent_payload, dict):
            intent_name = intent_payload.get("intent")
        normalized = self._normalize_intent_name(intent_name)
        route = self._lookup_mapped_value(self.route_map, normalized)
        return route if route is not None else self.default_route

    @staticmethod
    def _build_intent_context_block(context: dict[str, Any]) -> str:
        if not context:
            return ""
        items: list[str] = []
        for key in sorted(context.keys()):
            value = context.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            items.append(f"- {key}: {text}")
        if not items:
            return ""
        return "Conversation context:\n" + "\n".join(items)

    def _get_memory_state(self, state: AgentState) -> dict[str, Any]:
        memory = state.get("memory")
        return dict(getattr(memory, "state", {})) if memory is not None else {}

    def _apply_node_specific_pre_rule(
        self,
        *,
        state: AgentState,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Optional node-specific rule stage executed after context gathering and before LLM classify.

        Return schema:
        {
          "skip_llm": bool,
          "reason": str,
          "intent": {"intent": "...", "confidence": 0.x, "reason": "..."},
          "context_updates": {...}
        }
        """
        return None

    def _apply_node_specific_intent_override(
        self,
        *,
        intent: dict[str, Any],
        state: AgentState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        return intent

    def _apply_relevance_guard(
        self,
        *,
        intent: dict[str, Any],
        user_input: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.enable_relevance_guard:
            return intent
        if self.output_key != "relevance_intent":
            return intent
        current_intent = self._normalize_intent_name(intent.get("intent"))
        if current_intent in {"relevant", "empty"}:
            return intent
        if not self._is_active_collections_context(context):
            return intent
        if not self._looks_like_identity_or_verification_reply(user_input):
            return intent
        return {
            "intent": "relevant",
            "confidence": max(float(intent.get("confidence", 0.0) or 0.0), 0.88),
            "reason": "Context-aware relevance guard: identity/verification reply during active collections flow.",
        }

    @staticmethod
    def _is_active_collections_context(context: dict[str, Any]) -> bool:
        has_case = bool(str(context.get("active_case_id", "")).strip())
        last_response = str(context.get("last_agent_response", "")).lower()
        verify_hints = (
            "confirm your identity",
            "confirm your full name",
            "last 4 digits",
            "account number",
            "overdue amount",
            "payment today",
            "collections team",
        )
        has_verify_prompt = any(hint in last_response for hint in verify_hints)
        return has_case or has_verify_prompt

    @staticmethod
    def _looks_like_identity_or_verification_reply(text: str) -> bool:
        normalized = str(text or "").strip().lower()
        if not normalized:
            return False
        patterns = (
            r"\bmy name is\b",
            r"\bi am\b",
            r"\bthis is\b",
            r"\bmy full name\b",
            r"\blast\s*4\b",
            r"\baccount\b",
            r"\bending\b",
            r"\bxxxx\b",
            r"\b\d{4}\b",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _is_provider_rate_limit_error(error_text: str) -> bool:
        lowered = str(error_text or "").lower()
        return (
            "rate limit" in lowered
            or "rate_limit_exceeded" in lowered
            or "error code: 429" in lowered
            or "tokens per day" in lowered
            or "tpm" in lowered
            or "requests per minute" in lowered
        )
