"""Created: 2026-04-02

Purpose: Implements the reusable planner node for shared agent graphs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from src.nodes.base import BaseGraphNode
from src.nodes.types import AgentState, NodeUpdate


@dataclass(slots=True)
class PlannerNode(BaseGraphNode):
    """Runs the planner and decides whether the graph should act or respond.

    This is the neutral shared planning node for the platform. It performs the
    "decide the next step" part of an agent loop without assuming that the
    surrounding agent should be labeled as a ReAct agent.
    """

    llm: Any | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None
    available_tools: Any | None = None
    max_steps: int = 4
    last_plan_debug: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def plan(
        self,
        *,
        user_input: str,
        memory: Any | None = None,
        observation: dict[str, Any] | None = None,
        memory_context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        available_tools: list[Any] | None = None,
    ) -> Any:
        """Produces the next planner decision for this node.

        This method is the primary planner extension point for framework users.
        By default, the node can also act as a simple direct-response planner
        when an LLM and prompts are configured. That keeps the node usable out
        of the box for small graphs without forcing subclassing.

        Args:
            user_input: The latest user message.
            memory: Optional working memory.
            observation: Optional tool observation from the same turn.
            memory_context: Optional long-term memory retrieval context.
            system_prompt: Optional node-provided planner system prompt.
            user_prompt: Optional node-provided planner user prompt.
            available_tools: Optional tool metadata for planner use.

        Returns:
            A planner-defined decision object.
        """
        rendered_system_prompt = system_prompt if system_prompt is not None else self.system_prompt
        rendered_user_prompt = self._render_user_prompt(
            user_prompt=user_prompt if user_prompt is not None else (self.user_prompt or "{user_input}"),
            user_input=user_input,
            memory=memory,
            observation=observation,
            memory_context=memory_context,
            available_tools=available_tools,
        )
        self.last_plan_debug = {
            "prompt": rendered_user_prompt,
            "system_prompt": rendered_system_prompt or None,
            "llm_response": None,
            "llm_error": None,
        }
        if self.llm is None:
            response_text = rendered_user_prompt
        else:
            response_text = self.llm.generate(rendered_system_prompt or "", rendered_user_prompt).strip()
            self.last_plan_debug["llm_response"] = response_text
        return SimpleNamespace(
            thought="PlannerNode generated a direct response from its configured prompts.",
            tool_call=None,
            respond_directly=True,
            response_text=response_text,
            done=True,
        )

    def execute(self, state: AgentState) -> NodeUpdate:
        """Calls the planner for the current graph state.

        Args:
            state: The current shared graph state.

        Returns:
            A partial state update containing the planner decision and the
            incremented step count.
        """
        self._record_llm_usage(state, node_name="planner")
        decision = self.plan(
            user_input=state["user_input"],
            memory=state.get("memory"),
            observation=state.get("observation"),
            available_tools=self.available_tools if self.available_tools is not None else state.get("available_tools"),
            memory_context=state.get("memory_context"),
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
        )
        tool_calls = None
        tool_call = getattr(decision, "tool_call", None)
        if tool_call is not None:
            tool_calls = [
                {
                    "tool_name": str(getattr(tool_call, "tool_name", "")),
                    "arguments": getattr(tool_call, "arguments", {}) or {},
                }
            ]
        return {
            "decision": decision,
            "steps": state.get("steps", 0) + 1,
            "prompt": self.last_plan_debug.get("prompt"),
            "system_prompt": self.last_plan_debug.get("system_prompt"),
            "llm_response": self.last_plan_debug.get("llm_response"),
            "llm_error": self.last_plan_debug.get("llm_error"),
            "tool_calls": tool_calls,
        }

    def route(self, state: AgentState) -> str:
        """Chooses the next graph edge after planning.

        Args:
            state: The current shared graph state.

        Returns:
            The next node label: `act` when a tool should run, `respond` when a
            final textual reply should be generated, or `end` for termination.
        """
        if state.get("steps", 0) > self.max_steps:
            state["response"] = "I reached the tool limit for this turn. Please narrow the request."
            return "respond"
        decision = state["decision"]
        if getattr(decision, "respond_directly", False) or getattr(decision, "done", False):
            return "respond"
        if getattr(decision, "tool_call", None) is None:
            state["response"] = "I need a bit more detail to continue."
            return "respond"
        return "act"

    @staticmethod
    def _render_user_prompt(
        *,
        user_prompt: str,
        user_input: str,
        memory: Any | None,
        observation: dict[str, Any] | None,
        memory_context: dict[str, Any] | None,
        available_tools: list[Any] | None,
    ) -> str:
        """Renders optional planner context into the user prompt.

        The prompt is only enriched when it explicitly asks for a context value
        via one of these placeholders:

        - `{memory}`
        - `{observation}`
        - `{memory_context}`
        - `{available_tools}`

        Lines containing unresolved placeholders are removed so callers can keep
        one reusable template without manually branching on missing context.
        """
        values = {
            "user_input": PlannerNode._stringify_context(user_input),
            "memory": PlannerNode._stringify_context(memory),
            "observation": PlannerNode._stringify_context(observation),
            "memory_context": PlannerNode._stringify_context(memory_context),
            "available_tools": PlannerNode._stringify_context(available_tools),
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
        """Serializes planner context into a prompt-safe string."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, default=str, ensure_ascii=True)
        except TypeError:
            return str(value)
