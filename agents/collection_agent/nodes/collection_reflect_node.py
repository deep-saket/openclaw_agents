"""Reflect node variant for collection demo loops."""

from __future__ import annotations

from dataclasses import dataclass

from src.nodes.reflect_node import ReflectNode


@dataclass(slots=True)
class CollectionReflectNode(ReflectNode):
    """Collection-specific routing with LLM-driven reflection judgment."""

    def route(self, state: dict[str, Any]) -> str:
        if state.get("reflection_complete", self.default_is_complete):
            return self.complete_route
        routing_context = state.get("routing_context") if isinstance(state.get("routing_context"), dict) else {}
        plan_origin = str(routing_context.get("plan_origin", "react"))
        if plan_origin in {"pre_plan_intent", "post_memory_plan_intent"}:
            return "retry_plan_proposal"
        return "retry_react"
