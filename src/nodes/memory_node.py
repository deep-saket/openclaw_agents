"""Created: 2026-04-02

Purpose: Implements the reusable memory update node for shared agent graphs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.nodes.base import BaseGraphNode
from src.nodes.types import AgentState, NodeUpdate
from src.memory.types import TypedMemoryRecord, resolve_memory_type


@dataclass(slots=True)
class MemoryNode(BaseGraphNode):
    """Applies decision-driven updates to working and long-term memory.

    This node is intended to be the single place in a graph where memory
    mutations occur. It can:

    - update working memory message history
    - update working memory state
    - persist typed long-term memories through the configured store

    The node consumes structured `memory_updates` from either graph state or
    the current decision object. This keeps planner and response nodes focused
    on reasoning while memory mutation happens in one explicit graph step.
    """

    llm: Any | None = None
    memories: list[Any] = field(default_factory=list)
    memory_store: Any | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None
    default_memory_plan: list[dict[str, Any]] = field(default_factory=list)

    def plan(
        self,
        *,
        user_input: str,
        response: str | None = None,
        observation: dict[str, Any] | None = None,
        decision: Any | None = None,
        memory: Any | None = None,
        memory_targets: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Builds a memory update plan for the current turn.

        When an LLM and prompts are configured, the node asks the model to
        choose which memory targets should be updated, and which layer each
        update should use. Without an LLM, the node falls back to a simple,
        conservative plan for every configured memory target.
        """
        if self.llm is None:
            return self._default_plan(user_input=user_input, response=response, observation=observation, memory=memory)

        rendered_user_prompt = self._render_user_prompt(
            user_prompt=user_prompt if user_prompt is not None else (self.user_prompt or "{user_input}"),
            user_input=user_input,
            response=response,
            observation=observation,
            decision=decision,
            memory_targets=self._memory_targets(memory_targets),
        )
        raw = self.llm.generate(system_prompt or self.system_prompt or "", rendered_user_prompt).strip()
        planned_updates = self._parse_updates(raw)
        return planned_updates or self._default_plan(
            user_input=user_input,
            response=response,
            observation=observation,
            memory=memory,
        )

    def execute(self, state: AgentState) -> NodeUpdate:
        """Applies memory mutations for the current graph turn."""
        self._record_llm_usage(state, node_name="memory")
        observation = self._latest_observation_from_state(state)
        memory = self._working_memory() or state.get("memory")
        stored_memories: list[Any] = []
        explicit_updates = self._collect_updates(state)
        if not explicit_updates:
            explicit_updates = self.plan(
                user_input=state.get("user_input", ""),
                response=state.get("response"),
                observation=observation,
                decision=state.get("decision"),
                memory=memory,
                memory_targets=state.get("memory_targets"),
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
            )
        updates = self._merge_updates(
            defaults=self._default_plan(
                user_input=state.get("user_input", ""),
                response=state.get("response"),
                observation=observation,
                memory=memory,
                memory_targets=state.get("memory_targets"),
            ),
            planned=explicit_updates,
        )
        for update in updates:
            target = str(update.get("target", "working"))
            managed_types = self._managed_types(state.get("memory_targets"))
            if managed_types and target not in managed_types:
                continue
            if target == "working":
                self._apply_working_update(memory, update)
                continue
            stored = self._apply_long_term_update(update)
            if stored is not None:
                stored_memories.append(stored)

        result: NodeUpdate = {}
        if memory is not None:
            result["memory"] = memory
        if stored_memories:
            result["stored_memories"] = stored_memories
        return result

    @staticmethod
    def _latest_observation_from_state(state: AgentState) -> dict[str, Any] | None:
        observations = state.get("observations")
        if isinstance(observations, list):
            for item in reversed(observations):
                if isinstance(item, dict):
                    return item
        observation = state.get("observation")
        return observation if isinstance(observation, dict) else None

    @staticmethod
    def _collect_updates(state: AgentState) -> list[dict[str, Any]]:
        """Collects explicit memory updates from state and decision objects."""
        updates: list[dict[str, Any]] = []
        explicit_updates = state.get("memory_updates")
        if isinstance(explicit_updates, list):
            updates.extend(update for update in explicit_updates if isinstance(update, dict))
        decision = state.get("decision")
        decision_updates = getattr(decision, "memory_updates", None)
        if isinstance(decision_updates, list):
            updates.extend(update for update in decision_updates if isinstance(update, dict))
        return updates

    def _default_plan(
        self,
        *,
        user_input: str,
        response: str | None,
        observation: dict[str, Any] | None,
        memory: Any | None,
        memory_targets: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Builds a blind/default memory update plan for all configured targets."""
        updates = [dict(update) for update in self.default_memory_plan]
        for target in self._memory_targets(memory_targets):
            target_type = str(target["type"])
            if target_type == "working":
                if user_input:
                    updates.append({"target": "working", "operation": "add_message", "role": "user", "content": user_input})
                if response:
                    updates.append({"target": "working", "operation": "add_message", "role": "agent", "content": response})
                if memory is not None and getattr(memory, "current_goal", None) != user_input:
                    updates.append({"target": "working", "operation": "set_state", "values": {"current_goal": user_input}})
                continue

            content = {
                "user_input": user_input,
                "response": response,
                "observation": observation,
            }
            layer = target.get("default_layer", "warm")
            metadata = {"source": "memory_node", "tags": ["auto_memory_update"]}
            if target_type == "semantic":
                content = {"summary": response or user_input}
                metadata["tags"] = ["auto_summary"]
            elif target_type == "reflection":
                content = {"summary": response or user_input, "reasoning": None}
            elif target_type == "error":
                content = {
                    "input": user_input,
                    "output": response,
                    "error_type": "tool_failure" if observation and observation.get("status") == "failed" else "reflection",
                    "root_cause": None,
                }
            elif target_type == "task":
                content = {"summary": response or user_input}
            updates.append(
                {
                    "target": target_type,
                    "operation": "store",
                    "layer": layer,
                    "content": content,
                    "metadata": metadata,
                }
            )
        return updates

    def _memory_targets(self, state_targets: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Normalizes constructor-provided memory targets into descriptors."""
        targets: list[dict[str, Any]] = []
        for memory in self.memories:
            if memory is None:
                continue
            if hasattr(memory, "session_id") and hasattr(memory, "recent_items") and hasattr(memory, "state"):
                targets.append({"type": "working", "instance": memory, "default_layer": "hot"})
                continue
            if isinstance(memory, type) and issubclass(memory, TypedMemoryRecord):
                targets.append(
                    {
                        "type": memory.model_fields["type"].default,
                        "memory_class": memory,
                        "default_layer": getattr(memory, "default_layer", "warm"),
                    }
                )
        for target in state_targets or []:
            if not isinstance(target, dict):
                continue
            if not target.get("enabled", True):
                continue
            target_type = str(target.get("type", "")).strip().lower()
            if not target_type:
                continue
            targets.append(
                {
                    "type": target_type,
                    "default_layer": str(target.get("layer", "warm")),
                    "scope": target.get("scope"),
                    "limit": target.get("limit"),
                    "metadata": dict(target.get("metadata", {})) if isinstance(target.get("metadata"), dict) else {},
                }
            )
        return targets

    def _managed_types(self, state_targets: list[dict[str, Any]] | None = None) -> set[str]:
        """Returns the set of memory types configured on this node."""
        return {str(target["type"]) for target in self._memory_targets(state_targets)}

    def _working_memory(self) -> Any | None:
        """Returns the configured working-memory target, if any."""
        for target in self._memory_targets():
            if target["type"] == "working":
                return target["instance"]
        return None

    @staticmethod
    def _apply_working_update(memory: Any | None, update: dict[str, Any]) -> None:
        """Applies a working-memory update to the current memory object."""
        if memory is None:
            return
        operation = str(update.get("operation", "set_state"))
        if operation == "set_state":
            values = update.get("values", {})
            if isinstance(values, dict):
                memory.set_state(**values)
            return
        if operation == "add_message":
            role = str(update.get("role", "agent"))
            content = str(update.get("content", ""))
            if role == "user":
                memory.add_user_message(content)
            else:
                memory.add_agent_message(content)

    def _apply_long_term_update(self, update: dict[str, Any]) -> Any | None:
        """Stores one typed long-term memory update when a store is available."""
        if self.memory_store is None:
            return None
        target = str(update.get("target", "episodic"))
        memory_cls = resolve_memory_type(target)
        record = memory_cls.model_validate(
            {
                "type": target,
                "layer": update.get("layer", "warm"),
                "scope": update.get("scope", "agent_local"),
                "agent_id": update.get("agent_id"),
                "content": update.get("content"),
                "content_text": update.get("content_text"),
                "content_json": update.get("content_json"),
                "source_type": update.get("source_type"),
                "source_id": update.get("source_id"),
                "tags": update.get("tags", []),
                "metadata": update.get("metadata", {}),
                "importance": update.get("importance"),
                "confidence": update.get("confidence"),
            }
        )
        layer = str(update.get("layer", getattr(record, "layer", "warm")))
        if layer == "hot" and hasattr(record, "store_hot"):
            return record.store_hot(self.memory_store)
        if layer == "cold" and hasattr(record, "store_cold"):
            return record.store_cold(self.memory_store)
        if hasattr(record, "store_warm"):
            return record.store_warm(self.memory_store)
        return self.memory_store.add(record)

    @staticmethod
    def _merge_updates(defaults: list[dict[str, Any]], planned: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merges default and planned updates with planned overrides by target.

        Working-memory defaults are always preserved because they capture the
        live session transcript. For long-term memories, planned updates replace
        the default update for the same target type.
        """
        planned_targets = {str(update.get("target")) for update in planned if str(update.get("target")) != "working"}
        merged = [
            update
            for update in defaults
            if str(update.get("target")) == "working" or str(update.get("target")) not in planned_targets
        ]
        merged.extend(planned)
        return merged

    @staticmethod
    def _parse_updates(raw: str) -> list[dict[str, Any]]:
        """Parses JSON memory updates from the LLM output."""
        candidate = raw.strip()
        match = re.search(r"(\{.*\}|\[.*\])", candidate, re.DOTALL)
        if match is not None:
            candidate = match.group(1)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, dict):
            updates = parsed.get("memory_updates", [])
            return [item for item in updates if isinstance(item, dict)]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        return []

    @staticmethod
    def _render_user_prompt(
        *,
        user_prompt: str,
        user_input: str,
        response: str | None,
        observation: dict[str, Any] | None,
        decision: Any | None,
        memory_targets: list[dict[str, Any]],
    ) -> str:
        """Renders memory-planning context into a prompt template."""
        values = {
            "user_input": user_input,
            "response": response,
            "observation": json.dumps(observation, default=str, ensure_ascii=True) if observation is not None else None,
            "decision": str(decision) if decision is not None else None,
            "memory_targets": json.dumps(
                [
                    {"type": target["type"], "default_layer": target.get("default_layer")}
                    for target in memory_targets
                ],
                ensure_ascii=True,
            )
            if memory_targets
            else None,
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
