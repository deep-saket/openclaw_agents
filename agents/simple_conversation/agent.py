"""Created: 2026-04-03

Purpose: Implements the simple conversation agent using the notebook graph structure.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

from langgraph.graph import END, START, StateGraph

from src.agents.base_agent import BaseAgent
from src.nodes import MemoryNode, MemoryRetrieveNode, ResponseNode, WhatsAppNode
from src.nodes.react_node import ReactNode
from src.nodes.types import AgentState
from src.interfaces.whatsapp import MockWhatsAppInterface, TwilioWhatsAppInterface, WhatsAppInterface
from src.llm.qwen import Qwen3_1_7BLLM
from src.memory import WorkingMemory
from src.platform_logging.tracing import ExecutionTrace, JSONLTraceSink, trace_turn
from src.tools.registry import ToolRegistry
from src.utils.config import AppSettings


class ConversationNode(ReactNode):
    """Implements the notebook conversation planning rules."""

    def _apply_pre_llm_override(self, *, state: AgentState, context: dict[str, object]):
        user_input = str(state.get("user_input", ""))
        memory = state.get("memory")
        memory_context = state.get("memory_context")
        assert memory is not None
        working_memory = (memory_context or {}).get("working", memory)

        match = re.search(r"my name is\s+([A-Za-z][A-Za-z\s'-]*)", user_input, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            return {
                "skip_llm": True,
                "reason": "name_store",
                "decision": SimpleNamespace(
                    thought="Stored the user name.",
                    tool_call=None,
                    memory_updates=[{"target": "working", "operation": "set_state", "values": {"user_name": name}}],
                    respond_directly=True,
                    response_text="I'll remember that.",
                    done=True,
                ),
            }

        if "what is my name" in user_input.lower() or "what's my name" in user_input.lower():
            name = working_memory.state.get("user_name")
            if name:
                response_text = f"Your name is {name}."
            else:
                response_text = "You have not told me your name yet."
            return {
                "skip_llm": True,
                "reason": "name_lookup",
                "decision": SimpleNamespace(
                    thought="Answered from working memory.",
                    tool_call=None,
                    respond_directly=True,
                    response_text=response_text,
                    done=True,
                ),
            }
        return None


class SimpleConversationAgent(BaseAgent):
    """Runs the same simple conversation graph shown in the notebook."""

    def __init__(self, *, llm, memory: WorkingMemory, whatsapp: WhatsAppInterface, trace_sink=None) -> None:
        """Builds the graph exactly in the notebook style."""

        super().__init__(llm=llm, agent_name="conversation_agent", logger=None, trace_sink=trace_sink)
        self.memory = memory
        self.whatsapp = whatsapp
        self.memory_retrieve_node = MemoryRetrieveNode(
            tool_registry=ToolRegistry(),
            memories=[WorkingMemory],
        )
        self.conversation_node = ConversationNode(
            llm=llm,
            system_prompt="You are a concise conversation assistant.",
            user_prompt="User: {user_input}",
        )
        self.response_node = ResponseNode(default_response="I do not know how to respond.")
        self.whatsapp_node = WhatsAppNode(interface=whatsapp)
        self.memory_node = MemoryNode(memories=[memory])
        graph = StateGraph(AgentState)
        graph.add_node("retrieve_memory", self.memory_retrieve_node.execute)
        graph.add_node("plan", self.conversation_node.execute)
        graph.add_node("respond", self.response_node.execute)
        graph.add_node("whatsapp", self.whatsapp_node.execute)
        graph.add_node("memory", self.memory_node.execute)
        graph.add_edge(START, "retrieve_memory")
        graph.add_edge("retrieve_memory", "plan")
        graph.add_conditional_edges(
            "plan",
            self.conversation_node.route,
            {"respond": "respond", "end": END, "act": "respond"},
        )
        graph.add_edge("respond", "whatsapp")
        graph.add_edge("whatsapp", "memory")
        graph.add_edge("memory", END)
        self.graph = graph.compile()

    def run(self, user_input: str, session_id: str | None = None) -> str:
        """Runs one turn through the graph."""

        session_key = session_id or self.memory.session_id
        trace = ExecutionTrace(agent_name=self.agent_name, session_id=session_key, user_input=user_input)
        with trace_turn(trace, sink=self.trace_sink):
            state = self.graph.invoke(
                {
                    "session_id": session_key,
                    "user_input": user_input,
                    "memory": self.memory,
                    "steps": 0,
                }
            )
            trace.finish(status="completed")
        return state["response"]

    @classmethod
    def build_whatsapp_interface(cls, settings: AppSettings) -> WhatsAppInterface:
        """Builds the same WhatsApp interface selection used in the notebook."""

        if settings.whatsapp_mode == "fake":
            return MockWhatsAppInterface()
        return TwilioWhatsAppInterface(
            account_sid=settings.integrations.twilio_account_sid,
            auth_token=settings.integrations.twilio_auth_token,
            whatsapp_from=settings.integrations.twilio_whatsapp_from,
        )

    @classmethod
    def from_env(cls) -> "SimpleConversationAgent":
        """Builds the notebook-style agent from env/config."""

        settings = AppSettings.from_env()
        qwen_llm = Qwen3_1_7BLLM(
            model_name=settings.llm.model_name,
            device_map=settings.llm.device_map,
            torch_dtype=settings.llm.torch_dtype,
            max_new_tokens=settings.llm.max_new_tokens,
            enable_thinking=False,
        )
        memory = WorkingMemory(session_id=settings.notification_destination or "whatsapp:+919999999999")
        trace_sink = JSONLTraceSink(settings.log_path.with_name("conversation_agent_trace.jsonl"))
        whatsapp = cls.build_whatsapp_interface(settings)
        return cls(llm=qwen_llm, memory=memory, whatsapp=whatsapp, trace_sink=trace_sink)
