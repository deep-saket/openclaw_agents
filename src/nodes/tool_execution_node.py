"""Created: 2026-03-31

Purpose: Implements the reusable tool execution node for shared agent graphs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.nodes.base import BaseGraphNode
from src.nodes.types import AgentState, NodeUpdate
from src.memory.policies import MemoryPolicy
from src.memory.store import MemoryStore
from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry


@dataclass(slots=True)
class ToolExecutionNode(BaseGraphNode):
    """Executes a selected tool call and returns a normalized observation.

    This node is the standard "act" step in a graph. It assumes a previous
    planning node has already placed a tool-bearing decision into the shared
    state.

    The node supports two construction styles:

    - pass a ready-made `ToolExecutor` when the caller already has one
    - pass `registry` and `repository` directly for a simpler small-graph setup

    The second path keeps notebook and demo agents smaller while preserving the
    shared executor abstraction for larger applications.
    """

    executor: ToolExecutor | None = None
    registry: ToolRegistry | None = None
    repository: Any | None = None
    memory_store: MemoryStore | None = None
    memory_policy: MemoryPolicy | None = None
    llm: Any | None = None

    def execute(self, state: AgentState) -> NodeUpdate:
        """Runs the selected tool call from the planner decision.

        Args:
            state: The current shared graph state.

        Returns:
            A partial state update containing the normalized observation.
        """
        self._record_llm_usage(state, node_name="tool_execution")
        decision = state["decision"]
        tool_call = getattr(decision, "tool_call", None)
        assert tool_call is not None
        tool_result = self._get_executor().execute(tool_call.tool_name, tool_call.arguments)
        memory = state.get("memory")
        if memory is not None:
            memory.set_state(last_tool_used=tool_result["tool_name"])
        tool_arguments = dict(tool_call.arguments) if isinstance(tool_call.arguments, dict) else tool_call.arguments
        current_observation = {
            "tool_name": tool_result["tool_name"],
            "input": tool_arguments,
            "output": tool_result["output"],
        }
        existing_observations = state.get("observations")
        observations = list(existing_observations) if isinstance(existing_observations, list) else []
        observations.append(current_observation)
        return {
            "observation": current_observation,
            "observations": observations,
            "tool_calls": [
                {
                    "tool_name": str(tool_call.tool_name),
                    "arguments": tool_arguments,
                }
            ],
        }

    def _get_executor(self) -> ToolExecutor:
        """Returns the configured executor, creating one when needed.

        Returns:
            A reusable tool executor instance for this node.

        Raises:
            ValueError: If neither `executor` nor a registry was provided.
        """
        if self.executor is not None:
            return self.executor
        if self.registry is None:
            raise ValueError("ToolExecutionNode requires either an executor or a registry.")
        self.executor = ToolExecutor(
            registry=self.registry,
            repository=self.repository,
            memory_store=self.memory_store,
            memory_policy=self.memory_policy,
        )
        return self.executor


ToolNode = ToolExecutionNode
