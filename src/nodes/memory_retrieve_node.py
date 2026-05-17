"""Created: 2026-03-31

Purpose: Implements the reusable memory retrieval node for shared agent graphs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.nodes.base import BaseGraphNode
from src.nodes.types import AgentState, NodeUpdate
from src.memory.models import RetrievalContext
from src.memory.types import EpisodicMemory, ProceduralMemory, SemanticMemory, WorkingMemory
from src.tools.registry import ToolRegistry


@dataclass(slots=True)
class MemoryRetrieveNode(BaseGraphNode):
    """Builds the structured memory context for the current agent turn.

    This node gathers the four runtime memory inputs expected by the shared
    planner flow:

    - semantic memory retrieved from long-term memory
    - episodic memory retrieved from long-term memory
    - working memory derived from the active session state
    - procedural memory derived from the available tools and planner identity

    It is intentionally generic and does not embed MailMind-specific logic.
    """

    tool_registry: ToolRegistry
    llm: Any | None = None
    memory_retriever: Any | None = None
    memories: list[Any] | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None
    max_queries_per_target: int = 3
    last_plan_debug: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def plan(
        self,
        *,
        user_input: str,
        memory: Any | None = None,
        memory_context: dict[str, Any] | None = None,
        memory_targets: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Builds a memory retrieval plan for the current turn.

        If an LLM and prompts are configured, the model can choose which memory
        targets to retrieve and with what limits. Otherwise the node retrieves
        all configured memory targets using conservative defaults.
        """
        default_plan = self._default_plan(memory=memory, memory_targets=memory_targets)
        if self.llm is None:
            return default_plan
        rendered_user_prompt = self._render_user_prompt(
            user_prompt=user_prompt if user_prompt is not None else (self.user_prompt or "{user_input}"),
            user_input=user_input,
            memory=memory,
            memory_context=memory_context,
            memory_targets=self._memory_targets(memory_targets),
        )
        self.last_plan_debug = {
            "prompt": rendered_user_prompt,
            "system_prompt": (system_prompt or self.system_prompt or None),
            "llm_response": None,
            "llm_error": None,
        }
        raw = self.llm.generate(system_prompt or self.system_prompt or "", rendered_user_prompt).strip()
        self.last_plan_debug["llm_response"] = raw
        planned = self._parse_plan(raw)
        return planned or default_plan

    def execute(self, state: AgentState) -> NodeUpdate:
        """Builds and stores the memory context for the current turn.

        Args:
            state: The current shared graph state.

        Returns:
            A partial state update containing the assembled memory context.
        """
        self._record_llm_usage(state, node_name="memory_retrieve")
        user_input = state["user_input"]
        memory = state["memory"]
        memory_state = dict(getattr(memory, "state", {}))
        context = RetrievalContext(
            agent_id=str(memory_state.get("agent_id", "mailmind")),
            step_count=state.get("steps", 0),
            confidence=float(state.get("confidence", 1.0) or 1.0),
            last_error=bool(memory_state.get("last_error", False)),
        )
        retrieval_plan = self.plan(
            user_input=user_input,
            memory=memory,
            memory_context=state.get("memory_context"),
            memory_targets=state.get("memory_targets"),
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
        )
        assembled_context: dict[str, Any] = {}
        retrieval_events: list[dict[str, Any]] = []
        for item in retrieval_plan:
            target = str(item.get("target", "")).lower()
            limit = int(item.get("limit", 5))
            if target == "working":
                assembled_context["working"] = self._build_working_memory(memory, user_input)
                continue
            if target == "procedural":
                assembled_context["procedural"] = self._build_procedural_memory()
                continue
            if self.memory_retriever is None:
                assembled_context[target] = []
                continue
            filters = self._filters_for_target(item, context)
            stop_on_first_hit = bool(item.get("stop_on_first_hit", True))
            min_results = max(int(item.get("min_results", 1)), 1)
            per_target_results: dict[str, Any] = {}
            for query in self._target_queries(item=item, user_input=user_input):
                records = self._retrieve(
                    query,
                    filters=filters,
                    limit=limit,
                    context=context,
                )
                for record in records:
                    record_id = str(getattr(record, "id", ""))
                    if record_id and record_id not in per_target_results:
                        per_target_results[record_id] = record
                retrieval_events.append(
                    {
                        "target": target,
                        "query": query,
                        "filters": dict(filters),
                        "limit": limit,
                        "result_count": len(records),
                        "result_ids": [str(getattr(record, "id", "")) for record in records if getattr(record, "id", None)],
                    }
                )
                if stop_on_first_hit and len(per_target_results) >= min_results:
                    break
            assembled_context[target] = list(per_target_results.values())[:limit]
        return {
            "memory_context": assembled_context,
            "memory_retrievals": retrieval_events,
            "prompt": self.last_plan_debug.get("prompt"),
            "system_prompt": self.last_plan_debug.get("system_prompt"),
            "llm_response": self.last_plan_debug.get("llm_response"),
            "llm_error": self.last_plan_debug.get("llm_error"),
        }

    def _default_plan(self, *, memory: Any | None, memory_targets: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Builds the default retrieval plan for configured memory targets."""
        del memory
        return [
            {
                "target": target.get("type", ""),
                "limit": int(target.get("limit", 5)),
                "layer": target.get("layer"),
                "scope": target.get("scope"),
                "agent_id": target.get("agent_id"),
                "source_type": target.get("source_type"),
                "source_id": target.get("source_id"),
                "tags": target.get("tags"),
                "metadata": target.get("metadata"),
                "query": target.get("query"),
                "query_candidates": target.get("query_candidates"),
                "created_before": target.get("created_before"),
                "created_after": target.get("created_after"),
                "stop_on_first_hit": target.get("stop_on_first_hit", True),
                "min_results": int(target.get("min_results", 1)),
                "max_queries": int(target.get("max_queries", self.max_queries_per_target)),
            }
            for target in self._memory_targets(memory_targets)
        ]

    def _memory_targets(self, state_targets: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Normalizes constructor-provided memory targets into target names."""
        configured = self.memories if self.memories is not None else [SemanticMemory, EpisodicMemory, WorkingMemory, ProceduralMemory]
        targets: list[dict[str, Any]] = []
        for memory in configured:
            if memory is WorkingMemory or getattr(memory, "__name__", None) == "WorkingMemory":
                targets.append({"type": "working", "limit": 5})
                continue
            if memory is ProceduralMemory or getattr(memory, "__name__", None) == "ProceduralMemory":
                targets.append({"type": "procedural", "limit": 5})
                continue
            memory_type = getattr(getattr(memory, "model_fields", {}), "get", lambda *_: None)("type")
            if memory_type is not None:
                targets.append({"type": str(memory_type.default), "limit": 5})
                continue
            if hasattr(memory, "type"):
                targets.append({"type": str(getattr(memory, "type")), "limit": 5})
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
                    "limit": int(target.get("limit", 5)),
                    "layer": target.get("layer"),
                    "scope": target.get("scope"),
                    "agent_id": target.get("agent_id"),
                    "source_type": target.get("source_type"),
                    "source_id": target.get("source_id"),
                    "tags": target.get("tags"),
                    "metadata": dict(target.get("metadata", {})) if isinstance(target.get("metadata"), dict) else {},
                    "query": target.get("query"),
                    "query_candidates": target.get("query_candidates"),
                    "created_before": target.get("created_before"),
                    "created_after": target.get("created_after"),
                    "stop_on_first_hit": bool(target.get("stop_on_first_hit", True)),
                    "min_results": int(target.get("min_results", 1)),
                    "max_queries": int(target.get("max_queries", self.max_queries_per_target)),
                }
            )
        return targets

    @staticmethod
    def _build_working_memory(memory: Any | None, user_input: str) -> WorkingMemory:
        """Builds a working-memory snapshot from the current graph memory."""
        session_id = getattr(memory, "session_id", "unknown")
        recent_items = list(getattr(memory, "recent_items", []))
        if not recent_items and hasattr(memory, "history"):
            recent_items = [
                {"role": message.role, "content": message.content}
                for message in getattr(memory, "history", [])[-6:]
            ]
        return WorkingMemory(
            session_id=session_id,
            current_goal=user_input,
            state=dict(getattr(memory, "state", {})),
            recent_items=recent_items,
        )

    def _build_procedural_memory(self) -> ProceduralMemory:
        """Builds procedural memory from the configured planner, tools, and LLM."""
        return ProceduralMemory(
            tool_names=[tool.name for tool in self.tool_registry.list_tools()],
            planner_names=[],
            llm_names=[] if self._llm_name() is None else [self._llm_name()],
            prompt_names=[],
        )

    @staticmethod
    def _filters_for_target(target_plan: dict[str, Any], context: RetrievalContext) -> dict[str, object]:
        """Builds retrieval filters for one long-term memory target."""
        target = str(target_plan.get("target", "")).lower()
        filters: dict[str, object] = {"type": target}
        if target_plan.get("layer") is not None:
            filters["layer"] = target_plan["layer"]
        if target_plan.get("scope") is not None:
            filters["scope"] = target_plan["scope"]
        if target_plan.get("source_type") is not None:
            filters["source_type"] = target_plan["source_type"]
        if target_plan.get("source_id") is not None:
            filters["source_id"] = target_plan["source_id"]
        if target_plan.get("tags") is not None:
            filters["tags"] = target_plan["tags"]
        if target_plan.get("created_before") is not None:
            filters["created_before"] = target_plan["created_before"]
        if target_plan.get("created_after") is not None:
            filters["created_after"] = target_plan["created_after"]
        metadata = target_plan.get("metadata")
        if isinstance(metadata, dict) and metadata:
            filters["metadata"] = metadata
        explicit_agent_id = target_plan.get("agent_id")
        if explicit_agent_id is not None:
            filters["agent_id"] = explicit_agent_id
        elif context.agent_id is not None:
            filters["agent_id"] = context.agent_id
        return filters

    def _retrieve(
        self,
        query: str,
        *,
        filters: dict[str, object],
        limit: int,
        context: RetrievalContext,
    ) -> list[Any]:
        """Calls either a context-aware router or a legacy local retriever."""
        try:
            return self.memory_retriever.retrieve(query, filters=filters, limit=limit, context=context)
        except TypeError:
            return self.memory_retriever.retrieve(query, filters=filters, limit=limit)

    def _target_queries(self, *, item: dict[str, Any], user_input: str) -> list[str]:
        """Builds ordered, de-duplicated queries for one retrieval target."""
        query_values: list[str] = []
        raw_primary_query = item.get("query")
        primary_query = str(raw_primary_query).strip() if raw_primary_query is not None else ""
        if primary_query:
            query_values.append(primary_query)
        else:
            query_values.append(user_input)

        query_candidates = item.get("query_candidates")
        if isinstance(query_candidates, str) and query_candidates.strip():
            query_values.append(query_candidates.strip())
        elif isinstance(query_candidates, (list, tuple, set)):
            query_values.extend(
                str(candidate).strip()
                for candidate in query_candidates
                if candidate is not None and str(candidate).strip()
            )

        max_queries = max(1, int(item.get("max_queries", self.max_queries_per_target)))
        return self._dedupe_strings(query_values)[:max_queries]

    @staticmethod
    def _dedupe_strings(values: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for value in values:
            key = value.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(value.strip())
        return unique

    @staticmethod
    def _parse_plan(raw: str) -> list[dict[str, Any]]:
        """Parses an LLM-produced memory retrieval plan from JSON output."""
        candidate = raw.strip()
        match = re.search(r"(\{.*\}|\[.*\])", candidate, re.DOTALL)
        if match is not None:
            candidate = match.group(1)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, dict):
            items = parsed.get("memory_retrievals", [])
            return [item for item in items if isinstance(item, dict)]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        return []

    @staticmethod
    def _render_user_prompt(
        *,
        user_prompt: str,
        user_input: str,
        memory: Any | None,
        memory_context: dict[str, Any] | None,
        memory_targets: list[str],
    ) -> str:
        """Renders retrieval-planning context into the prompt template."""
        values = {
            "user_input": user_input,
            "memory": json.dumps(getattr(memory, "state", {}), default=str, ensure_ascii=True) if memory is not None else None,
            "memory_context": json.dumps(memory_context, default=str, ensure_ascii=True) if memory_context is not None else None,
            "memory_targets": json.dumps(memory_targets, ensure_ascii=True) if memory_targets else None,
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
