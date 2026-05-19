"""Collection memory-helper specific react node."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from src.nodes.react_node import ReactNode
from src.nodes.types import AgentState


@dataclass(slots=True)
class CollectionMemoryHelperReactNode(ReactNode):
    """Deterministic memory-helper orchestration using React hooks."""

    def _apply_pre_llm_override(self, *, state: AgentState, context: dict[str, Any]) -> dict[str, Any] | None:
        observation = state.get("observation")
        if isinstance(observation, dict) and observation:
            output = observation.get("output") if isinstance(observation.get("output"), dict) else {}
            return {
                "skip_llm": True,
                "reason": "observation_complete",
                "decision": SimpleNamespace(
                    thought="Memory update complete.",
                    tool_call=None,
                    respond_directly=True,
                    response_text=f"Memory helper updated stores. Extracted key points: {output.get('extracted_key_points', [])}",
                    done=True,
                ),
            }

        payload = self._parse_payload(str(state.get("user_input", "")))
        return {
            "skip_llm": True,
            "reason": "memory_update_dispatch",
            "decision": SimpleNamespace(
                thought="Updating global and user key-event memory stores.",
                tool_call=SimpleNamespace(tool_name="update_key_event_memory", arguments=payload),
                respond_directly=False,
                response_text=None,
                done=False,
            ),
        }

    @staticmethod
    def _parse_payload(user_input: str) -> dict[str, Any]:
        try:
            payload = json.loads(user_input)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
        return {
            "session_id": "unknown",
            "user_id": None,
            "trigger": {"reason": "unparsed_payload"},
            "conversation_messages": [{"role": "user", "content": user_input}],
            "conversation_state": {},
        }

