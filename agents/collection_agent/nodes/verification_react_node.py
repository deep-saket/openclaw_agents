"""Verification-specific React node for collection agent."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import yaml

from agents.collection_agent.nodes.collection_react_node import CollectionReactNode
from src.nodes.types import AgentState


@dataclass(slots=True)
class VerificationReactNode(CollectionReactNode):
    """Owns verification progression and verification-only tool planning."""

    def _update_node_owned_keys_before(self, *, state: AgentState) -> dict[str, Any] | None:
        observations = self._observations_from_state(state=state)
        return self._apply_verification_state_from_observations(state=state, observations=observations)

    def _build_context_for_react(self, state: AgentState) -> dict[str, Any]:
        context = super(VerificationReactNode, self)._build_context_for_react(state)
        memory = state.get("memory")
        memory_state = dict(getattr(memory, "state", {})) if memory is not None else {}
        context["available_tools"] = self._filter_available_tools_for_verification(
            available_tools=context.get("available_tools"),
            verified_dob=bool(state.get("verified_dob", memory_state.get("verified_dob", False))),
            verified_mobile=bool(state.get("verified_mobile", memory_state.get("verified_mobile", False))),
        )
        return context

    def _apply_pre_llm_override(self, *, state: AgentState, context: dict[str, Any]) -> dict[str, Any] | None:
        queued = super(VerificationReactNode, self)._apply_pre_llm_override(state=state, context=context)
        if queued is not None:
            return queued
        return self._apply_pre_llm_verification_override(state=state, context=context)

    def _apply_post_llm_override(self, *, state: AgentState, context: dict[str, Any], decision: Any) -> Any:
        decision = super(VerificationReactNode, self)._apply_post_llm_override(
            state=state,
            context=context,
            decision=decision,
        )
        proposed_calls = self._decision_tool_calls(decision) or []
        if not proposed_calls:
            return decision

        observations = context.get("observations") if isinstance(context.get("observations"), list) else []
        filtered_calls = [
            item
            for item in proposed_calls
            if not self._is_redundant_verification_tool_call(item=item, observations=observations)
        ]
        if filtered_calls == proposed_calls:
            return decision
        if not filtered_calls:
            return SimpleNamespace(
                thought=str(getattr(decision, "thought", "") or "No further verification tool execution is required."),
                tool_call=None,
                tool_calls=[],
                respond_directly=bool(getattr(decision, "respond_directly", False)),
                response_text=getattr(decision, "response_text", None),
                done=True,
                no_tools_required=True,
            )
        first = filtered_calls[0]
        return SimpleNamespace(
            thought=str(getattr(decision, "thought", "") or f"Use {first['tool_name']}."),
            tool_call=SimpleNamespace(
                tool_name=str(first.get("tool_name", "")).strip(),
                arguments=first.get("arguments", {}) if isinstance(first.get("arguments"), dict) else {},
            ),
            tool_calls=filtered_calls,
            respond_directly=False,
            response_text=None,
            done=False,
            no_tools_required=False,
        )

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

    def _apply_pre_llm_verification_override(
        self,
        *,
        state: AgentState,
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        memory = state.get("memory")
        memory_state = dict(getattr(memory, "state", {})) if memory is not None else {}
        identity_verified = bool(state.get("identity_verified", memory_state.get("identity_verified", False)))
        if identity_verified:
            return None

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

        turn_entities = context.get("extracted_entities_turn") if isinstance(context.get("extracted_entities_turn"), dict) else {}
        case_id = str(memory_state.get("active_case_id", "COLL-1001")).strip() or "COLL-1001"
        customer_id = str(memory_state.get("active_user_id", "")).strip()
        queued_calls: list[dict[str, Any]] = []
        for field in pending_fields:
            field_value = str(turn_entities.get(field, "")).strip()
            if not field_value:
                continue
            if self._field_has_terminal_verification_result(
                field=field,
                candidate_value=field_value,
                observations=context.get("observations") if isinstance(context.get("observations"), list) else [],
            ):
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
    def _is_redundant_verification_tool_call(*, item: dict[str, Any], observations: list[dict[str, Any]]) -> bool:
        tool_name = str(item.get("tool_name", "")).strip().lower()
        arguments = item.get("arguments") if isinstance(item.get("arguments"), dict) else {}
        if tool_name == "verify_dob":
            return VerificationReactNode._field_has_terminal_verification_result(
                field="dob",
                candidate_value=str(arguments.get("dob", "")).strip(),
                observations=observations,
            )
        if tool_name == "verify_mobile":
            return VerificationReactNode._field_has_terminal_verification_result(
                field="phone",
                candidate_value=str(arguments.get("phone", "")).strip(),
                observations=observations,
            )
        return False

    @staticmethod
    def _field_has_terminal_verification_result(
        *,
        field: str,
        candidate_value: str,
        observations: list[dict[str, Any]],
    ) -> bool:
        field_key = "dob" if field == "dob" else "phone"
        tool_name = "verify_dob" if field_key == "dob" else "verify_mobile"
        input_key = "dob" if field_key == "dob" else "phone"
        normalized_candidate = str(candidate_value).strip()
        if not normalized_candidate:
            return False

        for raw_observation in reversed(observations):
            observation = raw_observation if isinstance(raw_observation, dict) else {}
            if isinstance(observation.get("tool_phase"), dict):
                observation = dict(observation.get("tool_phase", {}))
            if str(observation.get("tool_name", "")).strip().lower() != tool_name:
                continue
            obs_input = observation.get("input") if isinstance(observation.get("input"), dict) else {}
            obs_value = str(obs_input.get(input_key, "")).strip()
            if obs_value != normalized_candidate:
                continue
            obs_output = observation.get("output") if isinstance(observation.get("output"), dict) else {}
            obs_status = str(obs_output.get("status", "")).strip().lower()
            return obs_status in {"verified", "failed", "locked"}
        return False

    def _apply_verification_state_from_observations(
        self,
        *,
        state: AgentState,
        observations: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        memory = state.get("memory")
        if memory is None:
            return None
        memory_state = dict(getattr(memory, "state", {}))
        required_fields = self._required_verification_fields(memory_state=memory_state)

        existing_verified_fields = self._existing_verified_fields(state=state, memory_state=memory_state)
        verified_set = set(existing_verified_fields)
        latest_status_by_field: dict[str, str] = {}

        unresolved_fields = {field for field in required_fields if field not in verified_set}
        if unresolved_fields:
            for raw_observation in observations:
                normalized_observation = raw_observation if isinstance(raw_observation, dict) else {}
                if isinstance(normalized_observation.get("tool_phase"), dict):
                    normalized_observation = dict(normalized_observation.get("tool_phase", {}))
                obs_tool = str(normalized_observation.get("tool_name", "")).strip().lower()
                if obs_tool not in {"verify_dob", "verify_mobile"}:
                    continue
                field = "dob" if obs_tool == "verify_dob" else "phone"
                if field not in unresolved_fields:
                    continue
                obs_output = (
                    normalized_observation.get("output")
                    if isinstance(normalized_observation.get("output"), dict)
                    else {}
                )
                obs_status = str(obs_output.get("status", "")).strip().lower()
                if obs_status in {"verified", "failed", "locked"}:
                    latest_status_by_field[field] = obs_status

            for field, status in latest_status_by_field.items():
                if status == "verified":
                    verified_set.add(field)
                elif status in {"failed", "locked"}:
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
    def _observations_from_state(*, state: AgentState) -> list[dict[str, Any]]:
        observations = state.get("observations")
        if isinstance(observations, list):
            return [item for item in observations if isinstance(item, dict)]
        observation = state.get("observation")
        if isinstance(observation, dict):
            return [observation]
        return []

    @staticmethod
    def _existing_verified_fields(*, state: AgentState, memory_state: dict[str, Any]) -> list[str]:
        verified_fields: list[str] = []
        state_verified = state.get("verification_verified_fields")
        if isinstance(state_verified, list):
            verified_fields.extend([str(x).strip().lower() for x in state_verified if str(x).strip()])
        elif isinstance(memory_state.get("verification_verified_fields"), list):
            verified_fields.extend(
                [str(x).strip().lower() for x in memory_state.get("verification_verified_fields", []) if str(x).strip()]
            )

        if bool(state.get("verified_dob", memory_state.get("verified_dob", False))):
            verified_fields.append("dob")
        if bool(state.get("verified_mobile", memory_state.get("verified_mobile", False))):
            verified_fields.append("phone")
        return sorted(set(verified_fields))

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
