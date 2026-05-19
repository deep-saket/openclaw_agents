"""Created: 2026-03-31

Purpose: Implements the reusable reflection node for shared agent graphs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.memory.base import BaseMemoryStore
from src.memory.types import ReflectionMemory, ReflectionMemoryContent
from src.nodes.base import BaseGraphNode
from src.nodes.types import AgentState, NodeUpdate


@dataclass(slots=True)
class ReflectNode(BaseGraphNode):
    """Reflects on the current turn and optionally feeds critique back into the graph.

    The default contract always emits structured reflection state:

    - `reflection_feedback`
    - `reflection_complete`

    Additional behaviors remain configurable:

    - optional merged reflection feedback in `observation`
    - optional `memory_updates` for a later `MemoryNode`
    - optional durable reflection logging through `memory_store`
    """

    memory_store: BaseMemoryStore | None = None
    agent_name: str = "platform"
    llm: Any | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None
    complete_route: str = "complete"
    incomplete_route: str = "incomplete"
    merge_feedback_into_observation: bool = False
    emit_memory_update: bool = False
    default_reason: str = "Reflection completed."
    default_is_complete: bool = True

    def execute(self, state: AgentState) -> NodeUpdate:
        """Runs reflection for the current graph state."""
        self._record_llm_usage(state, node_name="reflect")
        observations = list(state.get("observations", [])) if isinstance(state.get("observations"), list) else []
        if not observations and isinstance(state.get("observation"), dict):
            observations = [dict(state.get("observation", {}))]
        observation = None
        for item in reversed(observations):
            if isinstance(item, dict):
                observation = item
                break
        decision = state.get("decision")
        self._store_reflection_log(observation=observation, decision=decision)

        reflection_result = self._reflect(
            user_input=state.get("user_input", ""),
            observation=observation,
            decision=decision,
        )
        feedback = (
            reflection_result.get("feedback")
            if isinstance(reflection_result.get("feedback"), dict)
            else {
                "reason": self.default_reason,
                "is_complete": self.default_is_complete,
            }
        )
        result: NodeUpdate = {
            "reflection_feedback": feedback,
            "reflection_complete": bool(feedback.get("is_complete", self.default_is_complete)),
            "prompt": reflection_result.get("prompt"),
            "system_prompt": reflection_result.get("system_prompt"),
            "llm_response": reflection_result.get("llm_response"),
            "llm_error": reflection_result.get("llm_error"),
            "observations": observations,
        }
        if self.merge_feedback_into_observation:
            merged_observation = {
                "tool_phase": observation,
                "reflection_feedback": feedback,
            }
            result["observation"] = merged_observation
            merged_observations = [item for item in observations if isinstance(item, dict)]
            merged_observations.append(merged_observation)
            result["observations"] = merged_observations
        if self.emit_memory_update:
            result["memory_updates"] = [self._build_memory_update(state, feedback)]
        return result

    def route(self, state: AgentState) -> str:
        """Routes based on whether reflection considers the turn complete."""
        return self.complete_route if state.get("reflection_complete", self.default_is_complete) else self.incomplete_route

    def _store_reflection_log(self, *, observation: dict[str, Any] | None, decision: Any | None) -> None:
        """Stores a durable reflection record when memory logging is configured."""
        if self.memory_store is None or not observation:
            return
        tool_name = observation.get("tool_name", "unknown")
        output = observation.get("output")
        content = ReflectionMemoryContent(
            reasoning=getattr(decision, "thought", None),
            summary=f"Observed output from tool `{tool_name}`.",
            improvement_suggestions=[],
            failure_analysis=None,
        )
        self.memory_store.add(
            ReflectionMemory(
                layer="warm",
                content=content.model_dump(mode="json"),
                metadata={
                    "agent": self.agent_name,
                    "tags": ["reflection", tool_name],
                    "source": "reflect_node",
                    "priority": "low",
                    "tool_name": tool_name,
                    "llm_name": self._llm_name(),
                    "observation_preview": self._preview_output(output),
                },
            )
        )

    def _reflect(self, *, user_input: str, observation: dict[str, Any] | None, decision: Any | None) -> dict[str, Any]:
        """Builds structured reflection output for the current graph state."""
        if self.llm is None:
            return {
                "feedback": {
                    "reason": self.default_reason if observation or decision else "No reflection context was available.",
                    "is_complete": self.default_is_complete,
                },
                "prompt": None,
                "system_prompt": self.system_prompt or "",
                "llm_response": None,
                "llm_error": None,
            }
        rendered_user_prompt = self._render_user_prompt(
            user_prompt=self.user_prompt
            or "User input:\n{user_input}\n\nObservation:\n{observation}\n\nDecision:\n{decision}\n\n"
            "Return JSON with `reason` and `is_complete`.",
            user_input=user_input,
            observation=observation,
            decision=decision,
        )
        system_prompt = self.system_prompt or ""
        try:
            raw = self.llm.generate(system_prompt, rendered_user_prompt).strip()
            return {
                "feedback": self._parse_payload(raw),
                "prompt": rendered_user_prompt,
                "system_prompt": system_prompt,
                "llm_response": raw,
                "llm_error": None,
            }
        except Exception as exc:
            return {
                "feedback": {
                    "reason": f"Reflect LLM call failed: {str(exc).strip() or 'unknown_error'}",
                    "is_complete": self.default_is_complete,
                },
                "prompt": rendered_user_prompt,
                "system_prompt": system_prompt,
                "llm_response": None,
                "llm_error": str(exc),
            }

    def _build_memory_update(self, state: AgentState, feedback: dict[str, Any]) -> dict[str, Any]:
        """Builds a reflection-memory update for a later MemoryNode."""
        observations = list(state.get("observations", [])) if isinstance(state.get("observations"), list) else []
        if not observations and isinstance(state.get("observation"), dict):
            observations = [dict(state.get("observation", {}))]
        latest_observation = None
        for item in reversed(observations):
            if isinstance(item, dict):
                latest_observation = item
                break
        return {
            "target": "reflection",
            "operation": "store",
            "layer": "warm",
            "content": {
                "summary": feedback.get("reason", self.default_reason),
                "reasoning": {
                    "user_input": state.get("user_input", ""),
                    "decision": self._stringify(state.get("decision")),
                    "observation": self._stringify(latest_observation),
                    "is_complete": bool(feedback.get("is_complete", self.default_is_complete)),
                },
            },
            "metadata": {
                "source": "reflect_node",
                "tags": ["reflection", self.agent_name],
                "is_complete": bool(feedback.get("is_complete", self.default_is_complete)),
            },
        }

    def _parse_payload(self, raw: str) -> dict[str, Any]:
        """Parses structured reflection output from the bound llm."""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match is None:
            return {
                "reason": raw or self.default_reason,
                "is_complete": self.default_is_complete,
            }
        try:
            payload = json.loads(match.group(0))
        except Exception:
            # Do not fail the graph on imperfect model JSON; recover heuristically.
            lowered = str(raw or "").lower()
            inferred_complete = self.default_is_complete
            if re.search(r'"?is_complete"?\s*:\s*false', lowered):
                inferred_complete = False
            elif re.search(r'"?is_complete"?\s*:\s*true', lowered):
                inferred_complete = True
            return {
                "reason": str(raw or self.default_reason).strip() or self.default_reason,
                "is_complete": inferred_complete,
            }
        return {
            "reason": str(payload.get("reason", "")).strip() or self.default_reason,
            "is_complete": bool(payload.get("is_complete", self.default_is_complete)),
        }

    @staticmethod
    def _render_user_prompt(*, user_prompt: str, user_input: str, observation: Any, decision: Any) -> str:
        """Renders reflection prompt text from the current state."""
        rendered = user_prompt
        values = {
            "user_input": ReflectNode._stringify(user_input),
            "observation": ReflectNode._stringify(observation),
            "decision": ReflectNode._stringify(decision),
        }
        for key, value in values.items():
            rendered = rendered.replace(f"{{{key}}}", value)
        return rendered

    @staticmethod
    def _stringify(value: Any) -> str:
        """Serializes arbitrary reflection context into prompt-safe text."""
        try:
            return json.dumps(value, default=str, ensure_ascii=True)
        except TypeError:
            return str(value)

    @staticmethod
    def _preview_output(output: Any) -> str:
        """Builds a short serializable preview of a tool output payload."""
        preview = str(output)
        return preview[:280]
