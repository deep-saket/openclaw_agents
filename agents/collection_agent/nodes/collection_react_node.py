"""Collection-specific React node with hook-based overrides."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import yaml

from src.nodes.react_node import ReactNode
from src.nodes.types import AgentState
from src.tools.registry import ToolRegistry


@dataclass(slots=True)
class CollectionReactNode(ReactNode):
    """Collection-oriented react behavior without planner delegation.

    State Keys Read:
    - `user_input`
    - `steps`
    - `intent`
    - `conversation_history`
    - `observations` (session-level ordered tool/node observations)
    - `observation` (compatibility mirror of latest observation)
    - `extracted_entities`
    - `extracted_entities_turn`
    - `verification_entities`
    - `memory` (reads `memory.state` for tool history, verification state, case/user ids)

    State Keys Write:
    - standard React outputs from base node:
      `decision`, `steps`, `prompt`, `system_prompt`, `llm_response`,
      `llm_error`, `llm_status`, `tool_calls`, `pending_tool_calls`
    - verification-owned keys (via React lifecycle hooks):
      `verification_verified_fields`, `verification_missing_fields`,
      `verified_dob`, `verified_mobile`, `identity_verified`
    """

    tool_registry: ToolRegistry | None = None

    def _update_node_owned_keys_before(self, *, state: AgentState) -> dict[str, Any] | None:
        """Node-owned pre-execution state mutation.

        Applies verification state updates from the latest tool observation
        before routing/LLM planning for this React pass.
        """
        latest_observation = self._latest_observation_from_state(state=state)
        return self._apply_verification_state_from_observation(state=state, observation=latest_observation)

    def _update_node_owned_keys_after(self, *, state: AgentState, update: dict[str, Any]) -> dict[str, Any] | None:
        """Node-owned post-execution state mutation.

        CollectionReactNode currently does not mutate additional owned keys
        after decision construction; keep hook explicit for future ownership.
        """
        del state, update
        return None

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
        available_tools = self.available_tools if self.available_tools is not None else state.get("available_tools")
        available_tools = self._filter_available_tools_for_verification(
            available_tools=available_tools,
            verified_dob=bool(state.get("verified_dob", memory_state.get("verified_dob", False))),
            verified_mobile=bool(state.get("verified_mobile", memory_state.get("verified_mobile", False))),
        )

        # Return only the requested context keys.
        return {
            "available_tools": available_tools,
            "user_input": state.get("user_input"),
            "observations": observations,
            "recent_conversation": recent_conversation,
            "steps": state.get("steps", 0),
            "extracted_entities": extracted_entities,
            "extracted_entities_turn": extracted_entities_turn,
            "verification_entities": verification_entities,
        }

    @staticmethod
    def _filter_available_tools_for_verification(
        *,
        available_tools: Any,
        verified_dob: bool,
        verified_mobile: bool,
    ) -> Any:
        if not isinstance(available_tools, str):
            return available_tools
        if not verified_dob and not verified_mobile:
            return available_tools
        try:
            catalog = yaml.safe_load(available_tools)
        except Exception:
            return available_tools
        if not isinstance(catalog, dict):
            return available_tools
        tools = catalog.get("tools")
        if not isinstance(tools, list):
            return available_tools
        filtered_tools: list[Any] = []
        for tool in tools:
            if not isinstance(tool, dict):
                filtered_tools.append(tool)
                continue
            tool_name = str(tool.get("name", "")).strip()
            if verified_dob and tool_name == "verify_dob":
                continue
            if verified_mobile and tool_name == "verify_mobile":
                continue
            filtered_tools.append(tool)
        return yaml.safe_dump(
            {**catalog, "tools": filtered_tools},
            sort_keys=False,
            allow_unicode=False,
        ).strip()

    def _apply_pre_llm_override(self, *, state: AgentState, context: dict[str, Any]) -> dict[str, Any] | None:
        return self._apply_pre_llm_verification_override(state=state, context=context)

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

    def _apply_pre_llm_verification_override(
        self,
        *,
        state: AgentState,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        # Step 1: check whether verification is already complete.
        memory = state.get("memory")
        memory_state = dict(getattr(memory, "state", {})) if memory is not None else {}
        identity_verified = bool(state.get("identity_verified", memory_state.get("identity_verified", False)))
        if identity_verified:
            return None

        # Step 2: fetch required fields, verified fields, and derive pending fields.
        required_fields = self._required_verification_fields(memory_state=memory_state)
        state_verified_fields = state.get("verification_verified_fields")
        verified_fields = (
            [str(x).strip().lower() for x in state_verified_fields if str(x).strip()]
            if isinstance(state_verified_fields, list)
            else (
                [str(x).strip().lower() for x in memory_state.get("verification_verified_fields", []) if str(x).strip()]
                if isinstance(memory_state.get("verification_verified_fields"), list)
                else []
            )
        )
        pending_fields = (
            [str(x).strip().lower() for x in state.get("verification_missing_fields", []) if str(x).strip()]
            if isinstance(state.get("verification_missing_fields"), list)
            else []
        )
        if not pending_fields:
            verified_set = set(verified_fields)
            pending_fields = [field for field in required_fields if field not in verified_set]
        if not pending_fields:
            return None

        # Step 3: check only turn entities; if pending field appears, route to tool.
        turn_entities = context.get("extracted_entities_turn") if isinstance(context.get("extracted_entities_turn"), dict) else {}
        case_id = str(memory_state.get("active_case_id", "COLL-1001")).strip() or "COLL-1001"
        customer_id = str(memory_state.get("active_user_id", "")).strip()
        queued_calls: list[dict[str, Any]] = []
        for field in pending_fields:
            field_value = str(turn_entities.get(field, "")).strip()
            if not field_value:
                continue
            if field == "dob":
                queued_calls.append(
                    {
                        "tool_name": "verify_dob",
                        "arguments": {"case_id": case_id, "customer_id": customer_id, "dob": field_value},
                    }
                )
            elif field == "phone":
                queued_calls.append(
                    {
                        "tool_name": "verify_mobile",
                        "arguments": {"case_id": case_id, "customer_id": customer_id, "phone": field_value},
                    }
                )
        if not queued_calls:
            return None
        first_call = queued_calls[0]
        return {
            "skip_llm": True,
            "reason": "pre_llm_verification_override",
            "decision": self._tool_decision(
                str(first_call.get("tool_name", "")).strip(),
                first_call.get("arguments", {}) if isinstance(first_call.get("arguments"), dict) else {},
            ),
            "pending_tool_calls": queued_calls[1:],
        }

    @staticmethod
    def _latest_observation_from_state(*, state: AgentState) -> dict[str, Any] | None:
        observations = state.get("observations")
        if isinstance(observations, list):
            for item in reversed(observations):
                if isinstance(item, dict):
                    return item
        observation = state.get("observation")
        if isinstance(observation, dict):
            return observation
        return None

    @staticmethod
    def _latest_observation_from_context(*, context: dict[str, Any]) -> dict[str, Any] | None:
        observations = context.get("observations")
        if not isinstance(observations, list) or not observations:
            return None
        for item in reversed(observations):
            if isinstance(item, dict):
                return item
        return None

    def _apply_verification_state_from_observation(
        self,
        *,
        state: AgentState,
        observation: Any,
    ) -> dict[str, Any] | None:
        """Recomputes node-owned verification keys from memory + latest observation."""
        memory = state.get("memory")
        if memory is None:
            return None
        memory_state = dict(getattr(memory, "state", {}))
        required_fields = self._required_verification_fields(memory_state=memory_state)

        existing_verified_fields = (
            [str(x).strip().lower() for x in memory_state.get("verification_verified_fields", []) if str(x).strip()]
            if isinstance(memory_state.get("verification_verified_fields"), list)
            else []
        )
        verified_set = set(existing_verified_fields)

        # Backward compatibility: if list wasn't maintained previously, derive from boolean flags.
        if bool(memory_state.get("verified_dob", False)):
            verified_set.add("dob")
        if bool(memory_state.get("verified_mobile", False)):
            verified_set.add("phone")

        normalized_observation = observation if isinstance(observation, dict) else {}
        if isinstance(normalized_observation.get("tool_phase"), dict):
            normalized_observation = dict(normalized_observation.get("tool_phase", {}))
        obs_tool = str(normalized_observation.get("tool_name", "")).strip().lower()
        obs_output = (
            normalized_observation.get("output")
            if isinstance(normalized_observation.get("output"), dict)
            else {}
        )
        obs_status = str(obs_output.get("status", "")).strip().lower()

        # Apply latest verification tool outcome (if current observation is a verification tool).
        if obs_tool in {"verify_dob", "verify_mobile"}:
            field = "dob" if obs_tool == "verify_dob" else "phone"
            if obs_status == "verified":
                verified_set.add(field)
            elif obs_status in {"failed", "locked"}:
                verified_set.discard(field)

        verified_dob = "dob" in verified_set
        verified_mobile = "phone" in verified_set
        missing_fields = [item for item in required_fields if item not in verified_set]

        updates = {
            "verification_verified_fields": sorted(verified_set),
            "verification_missing_fields": sorted(missing_fields),
            "verified_dob": bool(verified_dob),
            "verified_mobile": bool(verified_mobile),
            "identity_verified": not bool(missing_fields),
        }
        memory.set_state(**updates)
        state.update(updates)
        return updates

    @staticmethod
    def _required_verification_fields(*, memory_state: dict[str, Any]) -> list[str]:
        required_fields = (
            [str(x).strip().lower() for x in memory_state.get("active_verification_required_fields", []) if str(x).strip()]
            if isinstance(memory_state.get("active_verification_required_fields"), list)
            else []
        )
        if not required_fields:
            required_fields = ["dob", "phone"]
        required_fields = [field for field in required_fields if field in {"dob", "phone"}]
        if not required_fields:
            return ["dob", "phone"]
        return required_fields

    @staticmethod
    def _tool_decision(tool_name: str, arguments: dict[str, Any]) -> Any:
        return SimpleNamespace(
            thought=f"Collection pre-rule chose tool `{tool_name}`.",
            tool_call=SimpleNamespace(tool_name=tool_name, arguments=arguments),
            respond_directly=False,
            response_text=None,
            done=False,
        )
