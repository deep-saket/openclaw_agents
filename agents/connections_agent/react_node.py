"""Connections-specific react node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.connections_agent.planner import ConnectionsRulePlanner
from src.nodes.react_node import ReactNode
from src.nodes.types import AgentState


@dataclass(slots=True)
class ConnectionsReactNode(ReactNode):
    """Runs the legacy deterministic rule engine through React hooks."""

    rule_engine: ConnectionsRulePlanner | None = None

    def _apply_pre_llm_override(self, *, state: AgentState, context: dict[str, Any]) -> dict[str, Any] | None:
        engine = self.rule_engine or ConnectionsRulePlanner()
        decision = engine.plan(
            user_input=str(state.get("user_input", "")),
            memory=state.get("memory"),
            observation=state.get("observation"),
            memory_context=state.get("memory_context"),
            available_tools=context.get("available_tools"),
        )
        return {
            "skip_llm": True,
            "reason": "connections_rule_engine",
            "decision": decision,
        }

