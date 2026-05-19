"""Created: 2026-04-01

Purpose: Implements the neutral graph-based shared agent runtime.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from src.agents.base_agent import BaseAgent
from src.nodes import AgentState, MemoryRetrieveNode, ReflectNode, ResponseNode, SessionStoreProtocol, ToolExecutionNode
from src.nodes.react_node import ReactNode
from src.memory.base import BaseMemoryStore
from src.platform_logging.tracing import ExecutionTrace, emit_trace_event, trace_node, trace_turn
from src.tools.executor import ToolExecutor


class GraphAgent(BaseAgent):
    """Runs a node-composed agent graph.

    This is the shared runnable agent abstraction for the platform. It is
    intentionally neutral about reasoning style. The current default graph uses
    `ReactNode`, `ToolExecutionNode`, `ReflectNode`, and `ResponseNode`, but
    future agents can compose different graphs from the same node vocabulary.

    The important architectural distinction is:

    - nodes contain reusable step logic
    - `GraphAgent` assembles and runs a graph of those nodes
    - concrete agents supply the tools, memory, and storage bindings

    So the platform no longer treats "ReAct agent" as the main abstraction.
    ReAct is just one node pattern inside a general graph runner.

    This runtime does not require an LLM. React routing can still run with
    deterministic fallbacks where configured.
    """

    def __init__(
        self,
        *,
        llm: Any,
        planner: Any | None = None,
        tool_registry: Any,
        memory: Any,
        storage: Any,
        logger: Any,
        session_store: SessionStoreProtocol,
        tool_executor: ToolExecutor,
        memory_store: BaseMemoryStore | None = None,
        memory_retriever: Any | None = None,
        agent_name: str = "platform",
        trace_sink: Any | None = None,
    ) -> None:
        """Initializes the graph agent and compiles its shared node graph.

        Args:
            llm: Optional LLM adapter available to the concrete agent.
            planner: Deprecated planner dependency (ignored; kept for compatibility).
            tool_registry: Registry of tools the react node may use.
            memory: Optional working-memory dependency placeholder kept for
                compatibility with the base agent shape.
            storage: Durable operational store owned by the concrete agent.
            logger: Optional structured logger.
            session_store: Loader for session-scoped working memory objects.
            tool_executor: Runtime component that validates and executes tools.
            memory_store: Optional long-term memory write path.
            memory_retriever: Optional long-term memory read path.
            agent_name: Stable agent identifier used in reflection memory.
            trace_sink: Optional real-time JSON trace sink.
        """
        self.session_store = session_store
        self.tool_executor = tool_executor
        self.memory_store = memory_store
        self.memory_retriever = memory_retriever
        self.planner = planner
        self.tool_registry = tool_registry
        self.storage = storage
        self.memory = memory
        self.agent_name = agent_name
        self._graph: Any | None = None
        self._memory_retrieve_node: MemoryRetrieveNode | None = None
        self._planner_node: ReactNode | None = None
        self._tool_execution_node: ToolExecutionNode | None = None
        self._reflect_node: ReflectNode | None = None
        self._response_node: ResponseNode | None = None
        super().__init__(
            llm=llm,
            agent_name=agent_name,
            logger=logger,
            trace_sink=trace_sink,
        )
        self._memory_retrieve_node = MemoryRetrieveNode(
            tool_registry=self.tool_registry,
            llm=self.llm,
            memory_retriever=self.memory_retriever,
        )
        self._planner_node = ReactNode(llm=self.llm)
        self._tool_execution_node = ToolExecutionNode(executor=self.tool_executor, llm=self.llm)
        self._reflect_node = ReflectNode(memory_store=self.memory_store, agent_name=self.agent_name, llm=self.llm)
        self._response_node = ResponseNode(llm=self.llm)
        self._graph = self._build_graph()
        self.log_info(
            "Graph agent initialized",
            agent_name=self.agent_name,
            llm_enabled=self.llm is not None,
            tool_count=len(self.tool_registry.list_tools()) if hasattr(self.tool_registry, "list_tools") else "unknown",
        )

    def run(self, user_input: str, session_id: str | None = None) -> str:
        """Executes one user turn through the compiled node graph."""
        session_key = session_id or "default-session"
        self.log_info(
            "Starting graph agent turn",
            agent_name=self.agent_name,
            session_id=session_key,
            llm_enabled=self.llm is not None,
        )
        memory = self.session_store.load(session_key)
        memory.add_user_message(user_input)
        try:
            trace = ExecutionTrace(agent_name=self.agent_name, session_id=session_key, user_input=user_input)
            with trace_turn(trace, sink=self.trace_sink):
                state = self._graph.invoke(
                    {
                        "session_id": session_key,
                        "turn_id": str(uuid4()),
                        "user_input": user_input,
                        "memory": memory,
                        "observation": None,
                        "observations": [],
                        "steps": 0,
                        "trace": {"trace_id": trace.trace_id},
                    }
                )
                trace.finish(status="completed")
        except Exception:
            if 'trace' in locals():
                trace.finish(status="failed", error="graph_invoke_failed")
            self.log_exception(
                "Graph agent turn failed",
                agent_name=self.agent_name,
                session_id=session_key,
            )
            raise
        response = state.get("response", "I’m not sure what to do next.")
        memory.add_agent_message(response)
        self.log_info(
            "Completed graph agent turn",
            agent_name=self.agent_name,
            session_id=session_key,
            steps=state.get("steps", 0),
            has_observation=bool(state.get("observations")) or state.get("observation") is not None,
            responded=bool(response),
        )
        return response

    def _build_graph(self) -> Any:
        """Builds the default shared node graph for an agent turn."""
        graph = StateGraph(AgentState)
        graph.add_node("retrieve_memory", self._wrap_node("retrieve_memory", self._memory_retrieve_node.execute))
        graph.add_node("plan", self._wrap_node("plan", self._planner_node.execute))
        graph.add_node("act", self._act_node)
        graph.add_node("reflect", self._wrap_node("reflect", self._reflect_node.execute))
        graph.add_node("respond", self._respond_step)
        graph.add_edge(START, "retrieve_memory")
        graph.add_edge("retrieve_memory", "plan")
        graph.add_conditional_edges(
            "plan",
            self._planner_node.route,
            {"act": "act", "respond": "respond", "end": END},
        )
        graph.add_edge("act", "reflect")
        graph.add_edge("reflect", "plan")
        graph.add_edge("respond", END)
        return graph.compile()

    def _wrap_node(self, node_name: str, fn: Any) -> Any:
        """Wraps a node function with real-time trace emission."""

        def _wrapped(state: AgentState) -> AgentState:
            with trace_node(node_name, state=state):
                result = fn(state)
            emit_trace_event(
                {
                    "event": "node_state",
                    "node_name": node_name,
                    "step": state.get("steps", 0),
                    "decision": repr(result.get("decision") if isinstance(result, dict) else None),
                    "state": {
                        "response": result.get("response") if isinstance(result, dict) else None,
                        "route": result.get("route") if isinstance(result, dict) else None,
                        "memory_context_keys": sorted((result.get("memory_context") or {}).keys())
                        if isinstance(result, dict) and isinstance(result.get("memory_context"), dict)
                        else [],
                    },
                }
            )
            return result

        return _wrapped

    def _act_node(self, state: AgentState) -> AgentState:
        """Delegates tool execution to the shared tool node."""
        with trace_node("act", state=state):
            result = self._tool_execution_node.execute(state)
        emit_trace_event(
            {
                "event": "node_state",
                "node_name": "act",
                "step": state.get("steps", 0),
                "decision": repr(state.get("decision")),
                "observation": result.get("observation"),
                "observations": result.get("observations"),
                "response": result.get("response"),
            }
        )
        return result

    def _respond_step(self, state: AgentState) -> AgentState:
        """Delegates final response generation to the shared response node."""
        with trace_node("respond", state=state):
            result = self._response_node.execute(state)
        emit_trace_event(
            {
                "event": "node_state",
                "node_name": "respond",
                "step": state.get("steps", 0),
                "decision": repr(state.get("decision")),
                "response": result.get("response"),
            }
        )
        return result


__all__ = ["GraphAgent"]
