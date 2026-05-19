"""Created: 2026-03-31

Purpose: Implements the reusable response node for shared agent graphs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.nodes.base import BaseGraphNode
from src.nodes.types import AgentState, NodeUpdate


@dataclass(slots=True)
class ResponseNode(BaseGraphNode):
    """Builds the final user-facing response text for the current turn."""

    llm: Any | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None
    default_response: str = "Done."

    def plan(
        self,
        *,
        user_input: str,
        observation: dict[str, Any] | None = None,
        response: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> str:
        """Builds the final response text for the current turn.

        Args:
            user_input: The original user request.
            observation: Optional observation produced by a prior tool or node.
            response: Optional precomputed response from the decision or state.
            system_prompt: Optional system prompt for LLM-based response
                formatting.
            user_prompt: Optional user prompt template.

        Returns:
            A final response string.
        """
        if response and self.llm is None:
            return response

        rendered_user_prompt = self._render_user_prompt(
            user_prompt=user_prompt
            if user_prompt is not None
            else (self.user_prompt or "{user_input}\n{observation}"),
            user_input=user_input,
            observation=observation,
        )

        if self.llm is None:
            if observation is not None:
                return rendered_user_prompt
            return self.default_response

        return self.llm.generate(system_prompt or self.system_prompt or "", rendered_user_prompt).strip() or self.default_response

    def execute(self, state: AgentState) -> NodeUpdate:
        """Computes the final response text from the planner decision and state.

        Args:
            state: The current shared graph state.

        Returns:
            A partial state update containing the textual response.
        """
        self._record_llm_usage(state, node_name="response")
        observations = list(state.get("observations", [])) if isinstance(state.get("observations"), list) else []
        if not observations and isinstance(state.get("observation"), dict):
            observations = [dict(state.get("observation", {}))]
        latest_observation = None
        for item in reversed(observations):
            if isinstance(item, dict):
                latest_observation = item
                break
        decision = state.get("decision")
        response = self.plan(
            user_input=state["user_input"],
            observation=latest_observation,
            response=(getattr(decision, "response_text", None) if decision is not None else None) or state.get("response"),
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
        )
        return {
            "response": response,
            "observation": latest_observation,
            "observations": observations,
        }

    @staticmethod
    def _render_user_prompt(
        *,
        user_prompt: str,
        user_input: str,
        observation: dict[str, Any] | None,
    ) -> str:
        """Renders response context into a user prompt template."""
        values = {
            "user_input": ResponseNode._stringify_context(user_input),
            "observation": ResponseNode._stringify_context(observation),
        }
        rendered_lines: list[str] = []
        for line in user_prompt.splitlines():
            rendered_line = line
            skip_line = False
            for key, value in values.items():
                placeholder = f"{{{key}}}"
                if placeholder not in rendered_line:
                    continue
                if value is None:
                    skip_line = True
                    break
                rendered_line = rendered_line.replace(placeholder, value)
            if skip_line:
                continue
            if re.search(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", rendered_line):
                continue
            rendered_lines.append(rendered_line)
        return "\n".join(rendered_lines).strip() or user_prompt

    @staticmethod
    def _stringify_context(value: Any | None) -> str | None:
        """Serializes response context into a prompt-safe string."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, default=str, ensure_ascii=True)
        except TypeError:
            return str(value)


RespondNode = ResponseNode
