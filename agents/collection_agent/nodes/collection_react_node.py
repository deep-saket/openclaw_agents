"""Collection-specific non-verification React node."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from src.nodes.react_node import ReactNode
from src.nodes.types import AgentState
from src.tools.registry import ToolRegistry


@dataclass(slots=True)
class CollectionReactNode(ReactNode):
    """Collection-oriented React behavior for non-verification tooling.

    State Keys Read:
    - `user_input`
    - `steps`
    - `conversation_history`
    - `observations` (session-level ordered tool/node observations)
    - `observation` (compatibility mirror of latest observation)
    - `extracted_entities`
    - `extracted_entities_turn`
    - `verification_entities`
    - `memory` (reads `memory.state` for tool history and entity context)

    State Keys Write:
    - standard React outputs from base node:
      `decision`, `steps`, `prompt`, `system_prompt`, `llm_response`,
      `llm_error`, `llm_status`, `tool_calls`, `pending_tool_calls`
    """

    tool_registry: ToolRegistry | None = None

    def _build_context_for_react(self, state: AgentState) -> dict[str, Any]:
        memory = state.get("memory")
        memory_state = dict(getattr(memory, "state", {})) if memory is not None else {}

        observations = list(state.get("observations", [])) if isinstance(state.get("observations"), list) else []
        if not observations and isinstance(state.get("observation"), dict):
            current_observation = state.get("observation")
            if isinstance(current_observation, dict) and isinstance(current_observation.get("tool_phase"), dict):
                current_observation = current_observation.get("tool_phase")
            if isinstance(current_observation, dict) and current_observation:
                observations = [current_observation]
        observations = [item for item in observations if isinstance(item, dict)]

        extracted_entities = state.get("extracted_entities")
        if not isinstance(extracted_entities, dict):
            extracted_entities = (
                dict(memory_state.get("extracted_entities", {}))
                if isinstance(memory_state.get("extracted_entities"), dict)
                else {}
            )

        extracted_entities_turn = state.get("extracted_entities_turn")
        if not isinstance(extracted_entities_turn, dict):
            extracted_entities_turn = (
                dict(memory_state.get("extracted_entities_turn", {}))
                if isinstance(memory_state.get("extracted_entities_turn"), dict)
                else {}
            )

        verification_entities = state.get("verification_entities")
        if not isinstance(verification_entities, dict):
            verification_entities = (
                dict(memory_state.get("verification_entities", {}))
                if isinstance(memory_state.get("verification_entities"), dict)
                else {}
            )

        conversation_history = state.get("conversation_history")
        if not isinstance(conversation_history, list):
            conversation_history = (
                list(memory_state.get("conversation_history", []))
                if isinstance(memory_state.get("conversation_history"), list)
                else []
            )
        recent_conversation: list[dict[str, str]] = []
        for item in conversation_history[-8:]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if not role or not content:
                continue
            recent_conversation.append(
                {
                    "role": role,
                    "content": (content[:280] + " ...[truncated]") if len(content) > 280 else content,
                }
            )

        return {
            "available_tools": self.available_tools if self.available_tools is not None else state.get("available_tools"),
            "user_input": state.get("user_input"),
            "observations": observations,
            "observation": observations[-1] if observations else None,
            "recent_conversation": recent_conversation,
            "steps": state.get("steps", 0),
            "extracted_entities": extracted_entities,
            "extracted_entities_turn": extracted_entities_turn,
            "verification_entities": verification_entities,
        }

    def _apply_post_llm_override(self, *, state: AgentState, context: dict[str, Any], decision: Any) -> Any:
        del state, context
        decision = self._sanitize_tool_decision(decision)
        if not bool(getattr(decision, "no_tools_required", False)):
            return decision
        return SimpleNamespace(
            thought=str(getattr(decision, "thought", "") or "No tool execution is required."),
            tool_call=None,
            tool_calls=[],
            respond_directly=bool(getattr(decision, "respond_directly", False)),
            response_text=getattr(decision, "response_text", None),
            done=True,
            no_tools_required=True,
        )

    def _sanitize_tool_decision(self, decision: Any) -> Any:
        proposed_calls = self._decision_tool_calls(decision) or []
        if not proposed_calls:
            return decision
        valid_calls: list[dict[str, Any]] = []
        for item in proposed_calls:
            normalized = self._validate_tool_call(item)
            if normalized is not None:
                valid_calls.append(normalized)
        if not valid_calls:
            return SimpleNamespace(
                thought=str(getattr(decision, "thought", "") or "No valid tool execution is required."),
                tool_call=None,
                tool_calls=[],
                respond_directly=bool(getattr(decision, "respond_directly", False)),
                response_text=getattr(decision, "response_text", None),
                done=True,
                no_tools_required=True,
            )
        first = valid_calls[0]
        return SimpleNamespace(
            thought=str(getattr(decision, "thought", "") or f"Use {first['tool_name']}."),
            tool_call=SimpleNamespace(
                tool_name=str(first.get("tool_name", "")).strip(),
                arguments=first.get("arguments", {}) if isinstance(first.get("arguments"), dict) else {},
            ),
            tool_calls=valid_calls,
            respond_directly=False,
            response_text=None,
            done=False,
            no_tools_required=False,
        )

    def _validate_tool_call(self, item: dict[str, Any]) -> dict[str, Any] | None:
        if self.tool_registry is None or not isinstance(item, dict):
            return item if isinstance(item, dict) else None
        tool_name = str(item.get("tool_name", "")).strip()
        if not tool_name:
            return None
        arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        try:
            tool = self.tool_registry.get(tool_name)
            validated_input = tool.input_schema.model_validate(arguments)
        except Exception:
            return None
        return {
            "tool_name": tool_name,
            "arguments": validated_input.model_dump(mode="json"),
        }

    @staticmethod
    def _tool_decision(tool_name: str, arguments: dict[str, Any]) -> Any:
        return SimpleNamespace(
            thought=f"Collection pre-rule chose tool `{tool_name}`.",
            tool_call=SimpleNamespace(tool_name=tool_name, arguments=arguments),
            respond_directly=False,
            response_text=None,
            done=False,
        )
