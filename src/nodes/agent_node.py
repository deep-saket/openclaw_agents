"""Created: 2026-04-18

Purpose: Implements a reusable node that delegates execution to another agent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.nodes.base import BaseGraphNode
from src.nodes.types import AgentState, NodeUpdate


@dataclass(slots=True)
class AgentNode(BaseGraphNode):
    """Runs a nested agent as one step inside a parent graph.

    This node allows agent composition where an existing agent runtime is reused
    as a capability in another agent graph. The nested agent is treated as a
    black-box executable component behind a stable node contract.
    """

    agent: Any
    llm: Any | None = None
    input_template: str = "{user_input}"
    session_id_template: str = "{session_id}::delegate::{agent_name}"
    include_as_observation: bool = True
    include_as_response: bool = False

    def execute(self, state: AgentState) -> NodeUpdate:
        """Delegates the current step to the configured nested agent."""
        self._record_llm_usage(state, node_name="agent")

        delegate_name = self._agent_name()
        delegate_input = self._render_template(
            template=self.input_template,
            state=state,
            delegate_name=delegate_name,
            default_value=state.get("user_input", ""),
        )
        delegate_session_id = self._render_template(
            template=self.session_id_template,
            state=state,
            delegate_name=delegate_name,
            default_value=str(state.get("session_id", "delegate-session")),
        )

        result = self._run_delegate(delegate_input=delegate_input, delegate_session_id=delegate_session_id)

        memory = state.get("memory")
        if memory is not None:
                memory.set_state(last_delegated_agent=delegate_name)

        update: NodeUpdate = {}
        if self.include_as_observation:
            new_observation = {
                "tool_name": f"agent:{delegate_name}",
                "output": result,
                "session_id": delegate_session_id,
            }
            update["observation"] = new_observation
            existing_observations = state.get("observations")
            observations = list(existing_observations) if isinstance(existing_observations, list) else []
            observations.append(new_observation)
            update["observations"] = observations
        if self.include_as_response:
            update["response"] = str(result)
        return update

    def _run_delegate(self, *, delegate_input: str, delegate_session_id: str) -> Any:
        """Runs the nested agent with best-effort signature compatibility."""
        run_method = getattr(self.agent, "run")
        try:
            return run_method(delegate_input, session_id=delegate_session_id)
        except TypeError:
            return run_method(delegate_input)

    def _agent_name(self) -> str:
        """Returns a stable nested-agent display name."""
        configured = getattr(self.agent, "agent_name", None)
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
        return type(self.agent).__name__

    @staticmethod
    def _render_template(*, template: str, state: AgentState, delegate_name: str, default_value: str) -> str:
        """Renders a state-aware template with placeholder-safe behavior."""
        latest_observation = None
        observations = state.get("observations")
        if isinstance(observations, list):
            for item in reversed(observations):
                if isinstance(item, dict):
                    latest_observation = item
                    break
        if latest_observation is None:
            latest_observation = state.get("observation")
        values = {
            "session_id": AgentNode._stringify(state.get("session_id")),
            "user_input": AgentNode._stringify(state.get("user_input")),
            "response": AgentNode._stringify(state.get("response")),
            "observation": AgentNode._stringify(latest_observation),
            "observations": AgentNode._stringify(state.get("observations")),
            "memory_context": AgentNode._stringify(state.get("memory_context")),
            "agent_name": AgentNode._stringify(delegate_name),
        }
        rendered = template
        for key, value in values.items():
            placeholder = f"{{{key}}}"
            if placeholder not in rendered:
                continue
            if value is None:
                rendered = rendered.replace(placeholder, "")
            else:
                rendered = rendered.replace(placeholder, value)

        if re.search(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", rendered):
            return default_value
        normalized = rendered.strip()
        return normalized or default_value

    @staticmethod
    def _stringify(value: Any | None) -> str | None:
        """Converts values to prompt-safe text."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, default=str, ensure_ascii=True)
        except TypeError:
            return str(value)


__all__ = ["AgentNode"]
