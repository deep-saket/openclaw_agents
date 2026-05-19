"""Relevance intent node for collection agent."""

from __future__ import annotations

from typing import Any

from agents.collection_agent.nodes.collection_intent_node import CollectionIntentNode
from src.nodes.types import AgentState


class RelevanceIntentNode(CollectionIntentNode):
    """Intent node dedicated to relevance classification.

    State Keys Read:
    - `user_input`
    - `message_source`
    - `conversation_phase`
    - `conversation_history`
    - `case_id`
    - `user_id`
    - `memory` (reads `memory.state` fields like `active_case_id`, `last_agent_response`, `turn_index`)

    State Keys Write:
    - `relevance_intent`
    - `intent` (compatibility mirror)
    - `prompt`
    - `system_prompt`
    - `llm_response`
    - `llm_error`
    - `llm_status`
    - `fallback_reason` (optional)
    - `response` (optional for irrelevant/empty mapped responses)
    """
    max_history_chars: int = 900
    max_history_turns: int = 20

    def __init__(
        self,
        *,
        llm: Any | None,
        allow_deterministic_fallback: bool,
        strict_llm_mode: bool,
        system_prompt: str,
        user_prompt: str,
    ) -> None:
        super().__init__(
            llm=llm,
            allow_deterministic_fallback=allow_deterministic_fallback,
            allow_rate_limit_fallback=False,
            enable_relevance_guard=not strict_llm_mode,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_key="relevance_intent",
            intent_labels=["relevant", "irrelevant", "empty"],
            default_intent="irrelevant",
            default_confidence=0.3,
            route_map={
                "relevant": "relevant",
                "irrelevant": "irrelevant",
                "empty": "empty",
                "unknown": "irrelevant",
            },
            default_route="irrelevant",
            empty_input_intent="empty",
            fallback_keyword_map={},
            response_map={
                "empty": "No input was provided. Please share a collections-related query such as dues, EMI, payment, verification, or repayment plan.",
                "irrelevant": "This request is outside collections scope. I can only help with loan dues, EMI, payments, verification, hardship plans, and follow-ups.",
                "unknown": "This request is outside collections scope. I can only help with loan dues, EMI, payments, verification, hardship plans, and follow-ups.",
            },
        )

    def _build_context_for_intent(self, state: AgentState) -> dict[str, Any]:
        memory_state = self._get_memory_state(state)
        conversation_history = (
            state.get("conversation_history")
            if isinstance(state.get("conversation_history"), list)
            else (memory_state.get("conversation_history") if isinstance(memory_state.get("conversation_history"), list) else [])
        )
        compact_history_text, history_truncated = self._compact_conversation_history(conversation_history)
        last_agent_response = str(memory_state.get("last_agent_response", "")).strip()
        active_case_id = str(state.get("case_id") or memory_state.get("active_case_id", "")).strip()
        active_user_id = str(state.get("user_id") or memory_state.get("active_user_id", "")).strip()

        return {
            "message_source": state.get("message_source") or memory_state.get("last_message_source"),
            "conversation_phase": state.get("conversation_phase"),
            "active_case_id": active_case_id,
            "active_user_id": active_user_id,
            "identity_verified": bool(memory_state.get("identity_verified", False)),
            "last_agent_response": last_agent_response,
            "previous_node": state.get("previous_node"),
            "conversation_history_compact": compact_history_text,
            "conversation_history_truncated": history_truncated,
            "conversation_history_turns": min(len(conversation_history), self.max_history_turns),
            "turn_index": memory_state.get("turn_index"),
            "active_collections_session": bool(active_case_id or active_user_id),
        }

    def _compact_conversation_history(self, history: list[dict[str, Any]]) -> tuple[str, bool]:
        if not history:
            return "", False
        # Keep earliest turns first (from beginning) and truncate by character budget.
        lines: list[str] = []
        for row in history[: self.max_history_turns]:
            if not isinstance(row, dict):
                continue
            role = str(row.get("role", "")).strip().lower() or "unknown"
            content = str(row.get("content", "")).strip()
            if not content:
                continue
            lines.append(f"{role}: {content}")
        joined = "\n".join(lines)
        if len(joined) <= self.max_history_chars:
            return joined, False
        return joined[: self.max_history_chars].rstrip() + " ...[truncated]", True

    def _apply_node_specific_pre_rule(
        self,
        *,
        state: AgentState,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        # No relevance pre-rule; allow classifier + relevance guard flow.
        return None

    def _apply_node_specific_intent_override(
        self,
        *,
        intent: dict[str, Any],
        state: AgentState,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        # No post-override at relevance-node level.
        return intent
