"""Connections Agent built on the shared easy_agents graph runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from agents.connections_agent.nodes import ConnectionsReflectNode
from agents.connections_agent.prompts import (
    load_connections_agent_prompts,
    render_connections_tool_catalog_yaml,
)
from agents.connections_agent.react_node import ConnectionsReactNode
from agents.connections_agent.repository import ConnectionsRepository
from agents.connections_agent.tools import ConnectionsDataStore, build_offline_toolset
from src.agents.base_agent import BaseAgent
from src.memory.session_store import SessionStore
from src.memory.types import WorkingMemory
from src.nodes import AgentState, MemoryRetrieveNode, ResponseNode, ToolExecutionNode
from src.platform_logging.tracing import ExecutionTrace, trace_turn
from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry


@dataclass(slots=True)
class ConnectionsAgent(BaseAgent):
    """Runs the collections-specific graph with local JSON-backed tools."""

    repository: ConnectionsRepository
    data_store: ConnectionsDataStore
    llm: Any | None = None
    session_store: SessionStore | None = None
    tool_registry: ToolRegistry | None = None
    tool_executor: ToolExecutor | None = None
    logger: Any | None = None
    trace_sink: Any | None = None
    agent_name: str = "connections_agent"

    def __post_init__(self) -> None:
        prompt_bundle = load_connections_agent_prompts()
        react_prompts = prompt_bundle.get("react", {})
        reflect_prompts = prompt_bundle.get("reflect", {})
        response_prompts = prompt_bundle.get("response", {})

        BaseAgent.__init__(
            self,
            llm=self.llm,
            agent_name=self.agent_name,
            logger=self.logger,
            trace_sink=self.trace_sink,
        )
        self.session_store = self.session_store or SessionStore(self.repository)
        self.tool_registry = self.tool_registry or self._build_tool_registry()
        self.tool_executor = self.tool_executor or ToolExecutor(
            registry=self.tool_registry,
            repository=self.repository,
            memory_store=None,
            memory_policy=None,
        )
        self.memory_retrieve_node = MemoryRetrieveNode(
            tool_registry=self.tool_registry,
            memories=[WorkingMemory],
        )
        self.react_node = ConnectionsReactNode(
            llm=self.llm,
            system_prompt=str(react_prompts.get("system_prompt", "")),
            user_prompt=str(react_prompts.get("user_prompt", "{user_input}")),
            available_tools=render_connections_tool_catalog_yaml(),
            max_steps=6,
        )
        self.tool_execution_node = ToolExecutionNode(executor=self.tool_executor)
        self.reflect_node = ConnectionsReflectNode(
            llm=self.llm,
            system_prompt=str(reflect_prompts.get("system_prompt", "")),
            complete_route="complete",
            incomplete_route="incomplete",
            merge_feedback_into_observation=True,
            emit_memory_update=False,
        )
        self.response_node = ResponseNode(
            llm=None,
            system_prompt=str(response_prompts.get("system_prompt", "")),
            user_prompt=str(response_prompts.get("user_prompt", "{observation}")),
            default_response="No action selected.",
        )
        self.graph = self._build_graph()

    def _build_tool_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        for tool in build_offline_toolset(self.data_store):
            registry.register(tool)
        return registry

    def _build_graph(self) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("memory_retrieve", self.memory_retrieve_node.execute)
        graph.add_node("react", self.react_node.execute)
        graph.add_node("tool_execution", self.tool_execution_node.execute)
        graph.add_node("reflect", self.reflect_node.execute)
        graph.add_node("response", self.response_node.execute)

        graph.add_edge(START, "memory_retrieve")
        graph.add_edge("memory_retrieve", "react")
        graph.add_conditional_edges(
            "react",
            self.react_node.route,
            {
                "act": "tool_execution",
                "respond": "reflect",
                "end": END,
            },
        )
        graph.add_edge("tool_execution", "react")
        graph.add_conditional_edges(
            "reflect",
            self.reflect_node.route,
            {
                "incomplete": "react",
                "complete": "response",
            },
        )
        graph.add_edge("response", END)
        return graph.compile()

    def run(self, user_input: str, session_id: str | None = None) -> str:
        """Runs one turn through the Connections Agent graph."""
        session_key = session_id or "connections-session"
        memory = self.session_store.load(session_key)
        trace = ExecutionTrace(agent_name=self.agent_name, session_id=session_key, user_input=user_input)
        with trace_turn(trace, sink=self.trace_sink):
            state = self.graph.invoke(
                {
                    "session_id": session_key,
                    "turn_id": str(uuid4()),
                    "user_input": user_input,
                    "memory": memory,
                    "memory_targets": [{"type": "working", "enabled": True, "limit": 6}],
                    "observation": None,
                    "steps": 0,
                }
            )
            trace.finish(status="completed")
        return str(state.get("response", "No response generated."))

    @classmethod
    def from_local_files(cls, base_dir: Path | None = None) -> "ConnectionsAgent":
        """Creates the agent using local `data/` and `runtime/` directories."""
        root = base_dir or Path(__file__).resolve().parent
        repository = ConnectionsRepository(runtime_dir=root / "runtime")
        data_store = ConnectionsDataStore(base_dir=root)
        return cls(repository=repository, data_store=data_store)


__all__ = ["ConnectionsAgent"]
