"""Created: 2026-04-03

Purpose: Implements a reusable WhatsApp channel node for shared agent graphs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.nodes.base import BaseGraphNode
from src.nodes.types import AgentState, NodeUpdate
from src.interfaces.whatsapp import WhatsAppInterface


@dataclass(slots=True)
class WhatsAppNode(BaseGraphNode):
    """Sends an outbound WhatsApp message from inside the graph.

    This node is the channel-side counterpart to response generation. It lets a
    graph send a message and optionally pause the current turn until a human
    responds on WhatsApp. That makes it suitable for HITL checkpoints and other
    mid-graph messaging flows.
    """

    interface: WhatsAppInterface
    llm: Any | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None
    wait_for_reply: bool = False
    waiting_route: str = "wait"
    continue_route: str = "continue"

    def plan(
        self,
        *,
        user_input: str,
        response: str | None = None,
        observation: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> str:
        """Builds the outbound WhatsApp message body.

        Args:
            user_input: The original user request.
            response: Optional precomputed response text.
            observation: Optional observation produced earlier in the graph.
            system_prompt: Optional system prompt override.
            user_prompt: Optional user prompt template override.

        Returns:
            The text to send over WhatsApp.
        """

        rendered_user_prompt = self._render_user_prompt(
            user_prompt=user_prompt
            if user_prompt is not None
            else (self.user_prompt or "{response}"),
            user_input=user_input,
            response=response,
            observation=observation,
        )

        if self.llm is None:
            if response:
                return response
            return rendered_user_prompt

        return self.llm.generate(system_prompt or self.system_prompt or "", rendered_user_prompt).strip() or (
            response or rendered_user_prompt
        )

    def execute(self, state: AgentState) -> NodeUpdate:
        """Sends the current message over WhatsApp and updates graph state."""

        self._record_llm_usage(state, node_name="whatsapp")
        observation = None
        observations = state.get("observations")
        if isinstance(observations, list):
            for item in reversed(observations):
                if isinstance(item, dict):
                    observation = item
                    break
        if observation is None:
            observation = state.get("observation")
        session_id = state.get("session_id") or getattr(state.get("memory"), "session_id", None)
        if not session_id:
            raise ValueError("WhatsAppNode requires `session_id` in state or on working memory.")

        message = self.plan(
            user_input=state.get("user_input", ""),
            response=state.get("response"),
            observation=observation,
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
        )
        send_result = self.interface.send_message(session_id, message)

        memory = state.get("memory")
        if self.wait_for_reply and memory is not None:
            memory.set_state(awaiting_whatsapp_reply=True, last_whatsapp_message=message)

        channel_result = {
            "channel": "whatsapp",
            "session_id": session_id,
            "message": message,
            "wait_for_reply": self.wait_for_reply,
            "send_result": send_result,
        }
        return {
            "channel_result": channel_result,
            "waiting": self.wait_for_reply,
            "final": self.wait_for_reply,
            "route": self.waiting_route if self.wait_for_reply else self.continue_route,
        }

    def route(self, state: AgentState) -> str:
        """Routes either to a wait edge or the next live graph step."""

        return self.waiting_route if state.get("waiting") else self.continue_route

    @staticmethod
    def _render_user_prompt(
        *,
        user_prompt: str,
        user_input: str,
        response: str | None,
        observation: dict[str, Any] | None,
    ) -> str:
        """Renders a prompt template for the outbound WhatsApp message."""

        values = {
            "user_input": WhatsAppNode._stringify_context(user_input),
            "response": WhatsAppNode._stringify_context(response),
            "observation": WhatsAppNode._stringify_context(observation),
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
        """Serializes prompt context into a safe string value."""

        if value is None:
            return None
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, default=str, ensure_ascii=True)
        except TypeError:
            return str(value)
