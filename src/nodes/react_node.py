"""Created: 2026-03-31

Purpose: Provides a backward-compatible alias for the planner node.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from src.nodes.planner_node import PlannerNode


@dataclass(slots=True)
class ReactNode(PlannerNode):
    """Implements a self-sufficient ReAct-style planning node.

    `ReactNode` is the node-first planning surface for agents that need to
    choose tools and route based on LLM output. By default it can:

    - render node prompts
    - call the bound llm
    - parse a JSON decision
    - return either a tool call or a direct response

    A separate `planner` object remains optional for backward compatibility,
    but users can now rely on the node directly or subclass it and override
    `plan()` when they need custom behavior.
    """

    planner: Any | None = None

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
        """Produces a ReAct-style decision for the current graph step.

        The node uses its own llm and prompt configuration by default. When a
        legacy delegate planner is attached, it forwards the call for backward
        compatibility.
        """
        if self.planner is None:
            if self.llm is None:
                return super().plan(
                    user_input=user_input,
                    memory=memory,
                    observation=observation,
                    memory_context=memory_context,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    available_tools=available_tools,
                )
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
            raw = self.llm.generate(rendered_system_prompt or "", rendered_user_prompt).strip()
            self.last_plan_debug["llm_response"] = raw
            return self._parse_decision(raw)
        plan_kwargs = {
            "user_input": user_input,
            "memory": memory,
            "observation": observation,
            "memory_context": memory_context,
            "available_tools": available_tools,
        }
        if system_prompt is not None:
            plan_kwargs["system_prompt"] = system_prompt
        if user_prompt is not None:
            plan_kwargs["user_prompt"] = user_prompt
        try:
            return self.planner.plan(**plan_kwargs)
        except TypeError:
            plan_kwargs.pop("system_prompt", None)
            plan_kwargs.pop("user_prompt", None)
            try:
                return self.planner.plan(**plan_kwargs)
            except TypeError:
                plan_kwargs.pop("available_tools", None)
                return self.planner.plan(**plan_kwargs)

    @staticmethod
    def _parse_decision(raw: str) -> Any:
        """Parses a JSON decision returned by an llm.

        If the payload does not contain valid JSON, the node falls back to a
        direct response decision using the raw text.
        """
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return SimpleNamespace(
                thought="ReactNode fell back to a direct response because no JSON decision was returned.",
                tool_call=None,
                respond_directly=True,
                response_text=raw,
                done=True,
            )
        payload = json.loads(match.group(0))
        tool_name = payload.get("tool_name")
        if tool_name is None:
            return SimpleNamespace(
                thought=payload.get("thought", "Respond directly."),
                tool_call=None,
                respond_directly=bool(payload.get("respond_directly", True)),
                response_text=payload.get("response_text", raw),
                done=bool(payload.get("done", True)),
            )
        return SimpleNamespace(
            thought=payload.get("thought", f"Use {tool_name}."),
            tool_call=SimpleNamespace(
                tool_name=tool_name,
                arguments=payload.get("arguments", {}),
            ),
            respond_directly=False,
            response_text=None,
            done=False,
        )
