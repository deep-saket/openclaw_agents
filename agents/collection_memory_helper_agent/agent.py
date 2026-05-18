"""Collection Memory Helper Agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.collection_memory_helper_agent.nodes import CollectionMemoryHelperReflectNode
from agents.collection_memory_helper_agent.prompts import (
    load_collection_memory_helper_prompts,
    render_collection_memory_helper_tool_catalog_yaml,
)
from agents.collection_memory_helper_agent.react_node import CollectionMemoryHelperReactNode
from agents.collection_memory_helper_agent.repository import CollectionMemoryRepository
from agents.collection_memory_helper_agent.tools import UpdateKeyEventMemoryTool
from src.agents.base_agent import BaseAgent
from src.nodes.response_node import ResponseNode
from src.nodes.tool_execution_node import ToolExecutionNode
from src.nodes.types import AgentState
from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry


@dataclass(slots=True)
class CollectionMemoryHelperAgent(BaseAgent):
    repository: CollectionMemoryRepository
    llm: Any | None = None
    trace_sink: Any | None = None
    trace_output_dir: Path | None = None
    agent_name: str = "collection_memory_helper_agent"

    def __post_init__(self) -> None:
        prompts = load_collection_memory_helper_prompts()
        react_prompts = prompts.get("react", {})
        reflect_prompts = prompts.get("reflect", {})
        response_prompts = prompts.get("response", {})

        BaseAgent.__init__(self, llm=self.llm, agent_name=self.agent_name, logger=None, trace_sink=self.trace_sink)

        registry = ToolRegistry()
        registry.register(UpdateKeyEventMemoryTool(repository=self.repository))
        executor = ToolExecutor(registry=registry, repository=None, memory_store=None, memory_policy=None)

        self.react_node = CollectionMemoryHelperReactNode(
            llm=self.llm,
            system_prompt=str(react_prompts.get("system_prompt", "")),
            user_prompt=str(react_prompts.get("user_prompt", "{user_input}")),
            available_tools=render_collection_memory_helper_tool_catalog_yaml(),
            max_steps=4,
        )
        self.tool_execution_node = ToolExecutionNode(executor=executor)
        self.reflect_node = CollectionMemoryHelperReflectNode(
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
            default_response="Memory helper completed.",
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("react", self.react_node.execute)
        graph.add_node("tool_execution", self.tool_execution_node.execute)
        graph.add_node("reflect", self.reflect_node.execute)
        graph.add_node("response", self.response_node.execute)

        graph.add_edge(START, "react")
        graph.add_conditional_edges(
            "react",
            self.react_node.route,
            {"act": "tool_execution", "respond": "reflect", "end": "reflect"},
        )
        graph.add_edge("tool_execution", "react")
        graph.add_conditional_edges(
            "reflect",
            self.reflect_node.route,
            {"incomplete": "react", "complete": "response"},
        )
        graph.add_edge("response", END)
        return graph.compile()

    def run_turn(self, payload: dict[str, Any]) -> AgentState:
        user_input = json.dumps(payload, ensure_ascii=True)
        return self.graph.invoke(
            {
                "session_id": str(payload.get("session_id", "memory-helper")),
                "user_input": user_input,
                "memory": None,
                "observation": None,
                "steps": 0,
            }
        )

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        state = self.run_turn(payload)
        return {
            "response": state.get("response"),
            "observation": state.get("observation"),
            "reflection_feedback": state.get("reflection_feedback"),
        }
