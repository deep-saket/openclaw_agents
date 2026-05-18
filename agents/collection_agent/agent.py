"""Collection Agent demo with plan loop and mode switching."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import threading
from pathlib import Path
import time
from typing import Any, ClassVar
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from agents.collection_memory_helper_agent.repository import CollectionMemoryRepository
from agents.collection_agent.nodes import (
    CollectionEntityExtractNode,
    CollectionReflectNode,
    CollectionReactNode,
    CollectionResponseNode,
    ExecutionPathIntentNode,
    PostMemoryPlanIntentNode,
    PrePlanIntentNode,
    PlanProposalNode,
    RelevanceIntentNode,
)
from agents.collection_agent.prompts import load_collection_agent_prompts, render_collection_tool_catalog_yaml
from agents.collection_agent.repository import CollectionRepository
from agents.collection_agent.state import CollectionGraphState
from agents.collection_agent.tools import (
    EntityExtractTool,
    HumanEscalationTool,
    LoanPolicyLookupTool,
    OfferEligibilityTool,
    PaymentLinkCreateTool,
    PlanProposeTool,
    PromiseCaptureTool,
    VerifyDOBTool,
    VerifyMobileTool,
    VerificationEntityExtractTool,
    VerificationMemoryVerifyTool,
)
from agents.collection_agent.tools.data_store import CollectionDataStore
from agents.collection_agent.tools.schemas import (
    EntityExtractInput,
    VerifyDOBInput,
    VerifyMobileInput,
    VerificationEntityExtractInput,
    VerificationMemoryVerifyInput,
)
from src.agents.base_agent import BaseAgent
from src.memory.session_store import SessionStore
from src.memory.types import WorkingMemory
from src.nodes.memory_retrieve_node import MemoryRetrieveNode
from src.nodes.tool_execution_node import ToolExecutionNode
from src.nodes.types import AgentState
from src.platform_logging.tracing import ExecutionTrace, emit_trace_event, trace_node, trace_turn
from src.schemas.messages import ConversationMessage
from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry


@dataclass(slots=True)
class CollectionAgent(BaseAgent):
    """Runs the collections demo graph with plan proposal loop."""

    repository: CollectionRepository
    data_store: CollectionDataStore
    llm: Any | None = None
    session_store: SessionStore | None = None
    tool_registry: ToolRegistry | None = None
    tool_executor: ToolExecutor | None = None
    logger: Any | None = None
    trace_sink: Any | None = None
    trace_output_dir: Path | None = None
    memory_repository: CollectionMemoryRepository | None = None
    verification_policy: dict[str, Any] | None = None
    entity_extract_tool: Any | None = None
    verification_entity_extract_tool: Any | None = None
    verification_memory_verify_tool: Any | None = None
    allow_deterministic_fallback: bool = False
    strict_llm_mode: bool = True
    agent_name: str = "collection_agent"
    last_trace: ExecutionTrace | None = None
    _session_locks: dict[str, threading.Lock] = field(default_factory=dict, init=False, repr=False)
    _session_locks_guard: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    _STATIC_NEXT_NODE_MAP: ClassVar[dict[str, str]] = {
        "entity_extract": "pre_plan_intent",
        "memory_retrieve": "post_memory_plan_intent",
        "tool_execution": "react",
        "relevant_response": "END",
        "irrelevant_response": "END",
    }

    _ROUTE_NEXT_NODE_MAP: ClassVar[dict[str, dict[str, str]]] = {
        "relevance_intent": {
            "relevant": "entity_extract",
            "irrelevant": "irrelevant_response",
            "empty": "irrelevant_response",
        },
        "pre_plan_intent": {
            "plan": "plan_proposal",
            "decide": "execution_path_intent",
        },
        "execution_path_intent": {
            "need_memory": "memory_retrieve",
            "need_tool": "react",
        },
        "post_memory_plan_intent": {
            "plan": "plan_proposal",
            "react": "react",
        },
        "react": {
            "act": "tool_execution",
            "respond": "plan_proposal",
            "end": "plan_proposal",
        },
        "plan_proposal": {
            "continue": "reflect",
        },
        "reflect": {
            "retry_plan_proposal": "plan_proposal",
            "complete": "relevant_response",
        },
    }

    def __post_init__(self) -> None:
        prompts = load_collection_agent_prompts()
        intent_prompts = prompts.get("intent", {})
        react_prompts = prompts.get("react", {})
        plan_proposal_prompts = prompts.get("plan_proposal", {})
        reflect_prompts = prompts.get("reflect", {})
        response_prompts = prompts.get("response", {})
        entity_extract_prompts = prompts.get("entity_extract", {})

        BaseAgent.__init__(
            self,
            llm=self.llm,
            agent_name=self.agent_name,
            logger=self.logger,
            trace_sink=self.trace_sink,
        )
        self.session_store = self.session_store or SessionStore(self.repository)
        self.memory_repository = self.memory_repository or CollectionMemoryRepository(
            collection_runtime_dir=self.repository.runtime_dir
        )
        self.tool_registry = self.tool_registry or self._build_tool_registry()
        self.tool_executor = self.tool_executor or ToolExecutor(
            registry=self.tool_registry,
            repository=self.repository,
            memory_store=None,
            memory_policy=None,
        )
        self.entity_extract_tool = self.entity_extract_tool or EntityExtractTool()
        self.verification_entity_extract_tool = self.verification_entity_extract_tool or VerificationEntityExtractTool()
        self.verification_memory_verify_tool = self.verification_memory_verify_tool or VerificationMemoryVerifyTool()
        deterministic_fallback_enabled = bool(self.allow_deterministic_fallback and not self.strict_llm_mode)
        self.memory_retrieve_node = MemoryRetrieveNode(tool_registry=self.tool_registry, memories=[WorkingMemory])
        self.relevance_intent_node = RelevanceIntentNode(
            llm=self.llm,
            allow_deterministic_fallback=deterministic_fallback_enabled,
            strict_llm_mode=self.strict_llm_mode,
            system_prompt=str(intent_prompts.get("relevance_system_prompt", "")),
            user_prompt=str(intent_prompts.get("relevance_user_prompt", "")),
        )
        self.pre_plan_intent_node = PrePlanIntentNode(
            llm=self.llm,
            allow_deterministic_fallback=deterministic_fallback_enabled,
            system_prompt=str(intent_prompts.get("pre_plan_system_prompt", "")),
            user_prompt=str(intent_prompts.get("pre_plan_user_prompt", "")),
        )
        self.execution_path_intent_node = ExecutionPathIntentNode(
            llm=self.llm,
            allow_deterministic_fallback=deterministic_fallback_enabled,
            system_prompt=str(intent_prompts.get("execution_path_system_prompt", "")),
            user_prompt=str(intent_prompts.get("execution_path_user_prompt", "")),
        )
        self.post_memory_plan_intent_node = PostMemoryPlanIntentNode(
            llm=self.llm,
            allow_deterministic_fallback=deterministic_fallback_enabled,
            system_prompt=str(intent_prompts.get("post_memory_plan_system_prompt", "")),
            user_prompt=str(intent_prompts.get("post_memory_plan_user_prompt", "")),
        )
        self.react_node = CollectionReactNode(
            llm=self.llm,
            system_prompt=str(react_prompts.get("system_prompt", "")),
            user_prompt=str(react_prompts.get("user_prompt", "{user_input}")),
            available_tools=render_collection_tool_catalog_yaml(),
            max_steps=8,
        )
        self.plan_node = PlanProposalNode(
            llm=self.llm,
            system_prompt=str(plan_proposal_prompts.get("system_prompt", "")),
            user_prompt=str(plan_proposal_prompts.get("user_prompt", "")),
            classifier_system_prompt=str(plan_proposal_prompts.get("classifier_system_prompt", "")),
            classifier_user_prompt=str(plan_proposal_prompts.get("classifier_user_prompt", "")),
            strict_llm_mode=self.strict_llm_mode,
        )
        self.tool_execution_node = ToolExecutionNode(executor=self.tool_executor)
        self.reflect_node = CollectionReflectNode(
            llm=self.llm,
            system_prompt=str(reflect_prompts.get("system_prompt", "")),
            user_prompt=str(reflect_prompts.get("user_prompt", "")),
            complete_route="complete",
            incomplete_route="incomplete",
            merge_feedback_into_observation=True,
            emit_memory_update=False,
        )
        self.relevant_response_node = CollectionResponseNode(
            llm=self.llm,
            system_prompt=str(response_prompts.get("system_prompt", "")),
            user_prompt=str(response_prompts.get("user_prompt", "{observation}")),
            render_system_prompt=str(response_prompts.get("render_system_prompt", "")),
            render_user_prompt=str(response_prompts.get("render_user_prompt", "")),
            verification_opening_template=str(response_prompts.get("verification_opening_template", "")),
            verification_followup_template=str(response_prompts.get("verification_followup_template", "")),
            verification_default_missing_text=str(
                response_prompts.get("verification_default_missing_text", "your date of birth (YYYY-MM-DD) and your registered phone number")
            ),
            verification_hardship_prefix=str(
                response_prompts.get("verification_hardship_prefix", "I am sorry to hear this, and I appreciate you sharing it. ")
            ),
            verification_ack_template=str(response_prompts.get("verification_ack_template", "Thank you{customer_suffix}. ")),
            strict_llm_mode=self.strict_llm_mode,
            default_response="No action selected.",
            default_target="customer",
        )
        self.irrelevant_response_node = CollectionResponseNode(
            llm=None,
            system_prompt="",
            user_prompt="{response}",
            default_response="This request is outside collections scope. I can only help with loan dues, EMI, payments, verification, hardship plans, and follow-ups.",
            default_target="customer",
        )
        self.entity_extract_node = CollectionEntityExtractNode(
            llm=self.llm,
            extract_callback=self._capture_verification_evidence,
            reconcile_callback=self._reconcile_verification_from_collected,
            # Keep callback fallback aligned with deterministic fallback mode.
            # In strict LLM mode this remains disabled to avoid hidden hardcoded extraction.
            allow_callback_fallback=deterministic_fallback_enabled,
            system_prompt=str(entity_extract_prompts.get("system_prompt", "")),
            user_prompt=str(entity_extract_prompts.get("user_prompt", "")),
        )
        self.graph = self._build_graph()

    def _build_tool_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(VerifyDOBTool(store=self.data_store))
        registry.register(VerifyMobileTool(store=self.data_store))
        registry.register(LoanPolicyLookupTool(store=self.data_store))
        registry.register(OfferEligibilityTool(store=self.data_store))
        registry.register(PaymentLinkCreateTool(store=self.data_store))
        registry.register(PromiseCaptureTool(store=self.data_store))
        registry.register(HumanEscalationTool(store=self.data_store))
        registry.register(PlanProposeTool(store=self.data_store))
        return registry

    def _build_graph(self) -> Any:
        graph = StateGraph(CollectionGraphState)
        graph.add_node("relevance_intent", self._wrap_node("relevance_intent", self.relevance_intent_node.execute))
        graph.add_node("entity_extract", self._wrap_node("entity_extract", self.entity_extract_node.execute))
        graph.add_node("pre_plan_intent", self._wrap_node("pre_plan_intent", self.pre_plan_intent_node.execute))
        graph.add_node(
            "execution_path_intent", self._wrap_node("execution_path_intent", self.execution_path_intent_node.execute)
        )
        graph.add_node("memory_retrieve", self._wrap_node("memory_retrieve", self.memory_retrieve_node.execute))
        graph.add_node(
            "post_memory_plan_intent",
            self._wrap_node("post_memory_plan_intent", self.post_memory_plan_intent_node.execute),
        )
        graph.add_node("react", self._wrap_node("react", self.react_node.execute))
        graph.add_node("plan_proposal", self._wrap_node("plan_proposal", self.plan_node.execute))
        graph.add_node("tool_execution", self._wrap_node("tool_execution", self.tool_execution_node.execute))
        graph.add_node("reflect", self._wrap_node("reflect", self.reflect_node.execute))
        graph.add_node("relevant_response", self._wrap_node("relevant_response", self.relevant_response_node.execute))
        graph.add_node(
            "irrelevant_response", self._wrap_node("irrelevant_response", self.irrelevant_response_node.execute)
        )

        graph.add_edge(START, "relevance_intent")
        graph.add_conditional_edges(
            "relevance_intent",
            self.relevance_intent_node.route,
            {
                "relevant": "entity_extract",
                "irrelevant": "irrelevant_response",
                "empty": "irrelevant_response",
            },
        )
        graph.add_edge("entity_extract", "pre_plan_intent")
        graph.add_conditional_edges(
            "pre_plan_intent",
            self.pre_plan_intent_node.route,
            {
                "plan": "plan_proposal",
                "decide": "execution_path_intent",
            },
        )
        graph.add_conditional_edges(
            "execution_path_intent",
            self.execution_path_intent_node.route,
            {
                "need_memory": "memory_retrieve",
                "need_tool": "react",
            },
        )
        graph.add_edge("memory_retrieve", "post_memory_plan_intent")
        graph.add_conditional_edges(
            "post_memory_plan_intent",
            self.post_memory_plan_intent_node.route,
            {
                "plan": "plan_proposal",
                "react": "react",
            },
        )
        graph.add_conditional_edges(
            "react",
            self.react_node.route,
            {
                "act": "tool_execution",
                "respond": "plan_proposal",
                "end": "plan_proposal",
            },
        )
        graph.add_conditional_edges(
            "plan_proposal",
            self.plan_node.route,
            {
                "continue": "reflect",
            },
        )
        graph.add_edge("tool_execution", "react")
        graph.add_conditional_edges(
            "reflect",
            self.reflect_node.route,
            {
                "retry_plan_proposal": "plan_proposal",
                "complete": "relevant_response",
            },
        )
        graph.add_edge("relevant_response", END)
        graph.add_edge("irrelevant_response", END)
        return graph.compile()

    def _wrap_node(self, node_name: str, fn: Any) -> Any:
        def _wrapped(state: CollectionGraphState) -> CollectionGraphState:
            trace_state = {k: v for k, v in state.items() if k != "memory"}
            with trace_node(node_name, state=trace_state):
                result = fn(state)
            state_update: dict[str, Any] = {}
            if isinstance(result, dict):
                if node_name == "tool_execution":
                    self._persist_tool_observation_history(state=state, update=result)
                route_value = self._infer_route_value(node_name=node_name, prior_state=state, update=result)
                self._apply_plan_origin_context(node_name=node_name, route_value=route_value, update=result)
                history = list(state.get("node_history", []))
                previous_node = history[-1] if history else "START"
                result["node_history"] = [*history, node_name]
                result["previous_node"] = previous_node
                result["next_node"] = self._resolve_next_node(
                    node_name=node_name,
                    update=result,
                    route_value=route_value,
                )
                result.setdefault("conversation_phase", self._phase_for_node(node_name))
                state_update = {k: v for k, v in result.items() if k != "memory"}
                if route_value:
                    state_update.setdefault("route", route_value)
                llm_response = state_update.get("llm_response")
                llm_error = state_update.get("llm_error")
                prompt_text = state_update.get("prompt")
                llm_status = None
                if llm_error:
                    llm_status = "llm_error"
                elif llm_response is not None:
                    llm_status = "used_llm"
                elif prompt_text:
                    llm_status = "prompt_rendered_no_output"
                if llm_status:
                    state_update["llm_status"] = llm_status
                    result.setdefault("llm_status", llm_status)
            debug_message = self._build_node_debug_message(node_name=node_name, state=state, update=state_update)
            emit_trace_event(
                {
                    "event": "node_state",
                    "node_name": node_name,
                    "step": state.get("steps", 0),
                    "decision": repr(state.get("decision")),
                    "observation": state_update.get("observation") if isinstance(state_update, dict) else None,
                    "response": state_update.get("response") if isinstance(state_update, dict) else None,
                    "route": state_update.get("route") if isinstance(state_update, dict) else None,
                    "prompt": state_update.get("prompt") if isinstance(state_update, dict) else None,
                    "system_prompt": state_update.get("system_prompt") if isinstance(state_update, dict) else None,
                    "llm_response": state_update.get("llm_response") if isinstance(state_update, dict) else None,
                    "llm_error": state_update.get("llm_error") if isinstance(state_update, dict) else None,
                    "llm_status": state_update.get("llm_status") if isinstance(state_update, dict) else None,
                    "messages": state_update.get("messages") if isinstance(state_update, dict) else None,
                    "tool_calls": state_update.get("tool_calls") if isinstance(state_update, dict) else None,
                    "state_update_keys": sorted(state_update.keys()),
                    "state_update": state_update,
                    "human_message": debug_message,
                }
            )
            return result

        return _wrapped

    def _persist_tool_observation_history(self, *, state: CollectionGraphState, update: dict[str, Any]) -> None:
        memory = state.get("memory")
        if memory is None:
            return
        observation = update.get("observation")
        if not isinstance(observation, dict):
            return
        tool_phase = observation.get("tool_phase")
        payload = tool_phase if isinstance(tool_phase, dict) else observation
        if not isinstance(payload, dict):
            return
        tool_name = str(payload.get("tool_name", "")).strip()
        if not tool_name:
            return
        status = str(payload.get("status", "")).strip()
        output = payload.get("output") if isinstance(payload.get("output"), dict) else {}
        entry = {
            "tool_name": tool_name,
            "status": status,
            "output": output,
            "timestamp_ms": int(time.time() * 1000),
        }
        memory_state = dict(getattr(memory, "state", {}))
        history = (
            list(memory_state.get("tool_observations_history", []))
            if isinstance(memory_state.get("tool_observations_history"), list)
            else []
        )
        if history:
            last = history[-1] if isinstance(history[-1], dict) else {}
            if (
                str(last.get("tool_name", "")).strip() == entry["tool_name"]
                and str(last.get("status", "")).strip() == entry["status"]
                and dict(last.get("output", {})) == entry["output"]
            ):
                memory.set_state(last_tool_observation=entry, tool_observations_history=history[-40:])
                update["tool_observations_history"] = history[-40:]
                return
        history.append(entry)
        history = history[-40:]
        memory.set_state(last_tool_observation=entry, tool_observations_history=history)
        update["tool_observations_history"] = history

    def _resolve_next_node(
        self,
        *,
        node_name: str,
        update: dict[str, Any],
        route_value: str | None = None,
    ) -> str | list[str]:
        if node_name in self._STATIC_NEXT_NODE_MAP:
            return self._STATIC_NEXT_NODE_MAP[node_name]

        route_map = self._ROUTE_NEXT_NODE_MAP.get(node_name, {})
        if not route_value:
            route_value = str(update.get("route", "")).strip().lower()
        if route_value and route_value in route_map:
            return route_map[route_value]

        next_candidates = sorted(set(route_map.values()))
        if not next_candidates:
            return "unknown"
        if len(next_candidates) == 1:
            return next_candidates[0]
        return next_candidates

    def _infer_route_value(self, *, node_name: str, prior_state: CollectionGraphState, update: dict[str, Any]) -> str | None:
        merged_state: dict[str, Any] = dict(prior_state)
        merged_state.update(update)
        if node_name == "relevance_intent":
            return str(self.relevance_intent_node.route(merged_state)).strip().lower()
        if node_name == "pre_plan_intent":
            return str(self.pre_plan_intent_node.route(merged_state)).strip().lower()
        if node_name == "execution_path_intent":
            return str(self.execution_path_intent_node.route(merged_state)).strip().lower()
        if node_name == "post_memory_plan_intent":
            return str(self.post_memory_plan_intent_node.route(merged_state)).strip().lower()
        if node_name == "react":
            return str(self.react_node.route(merged_state)).strip().lower()
        if node_name == "plan_proposal":
            return str(self.plan_node.route(merged_state)).strip().lower()
        if node_name == "reflect":
            return str(self.reflect_node.route(merged_state)).strip().lower()
        return None

    @staticmethod
    def _set_plan_origin(update: dict[str, Any], *, plan_origin: str) -> None:
        context = update.get("routing_context")
        context_map = dict(context) if isinstance(context, dict) else {}
        context_map["plan_origin"] = plan_origin
        update["routing_context"] = context_map

    def _apply_plan_origin_context(self, *, node_name: str, route_value: str | None, update: dict[str, Any]) -> None:
        if node_name == "pre_plan_intent" and route_value == "plan":
            self._set_plan_origin(update, plan_origin="pre_plan_intent")
            return
        if node_name == "post_memory_plan_intent" and route_value == "plan":
            self._set_plan_origin(update, plan_origin="post_memory_plan_intent")
            return
        if node_name == "react" and route_value in {"respond", "end"}:
            self._set_plan_origin(update, plan_origin="react")

    @staticmethod
    def _build_node_debug_message(*, node_name: str, state: CollectionGraphState, update: dict[str, Any]) -> str:
        if node_name in {"relevance_intent", "pre_plan_intent", "execution_path_intent", "post_memory_plan_intent"}:
            payload = update.get(node_name) if isinstance(update.get(node_name), dict) else {}
            intent = str(payload.get("intent", "unknown"))
            reason = str(payload.get("reason", "")).strip()
            route = str(update.get("route", "")).strip()
            reason_part = f" reason={reason}" if reason else ""
            route_part = f", route={route}" if route else ""
            return f"{node_name}: classified intent={intent}{route_part}.{reason_part}".strip()

        if node_name == "entity_extract":
            extracted_turn = (
                update.get("extracted_entities_turn")
                if isinstance(update.get("extracted_entities_turn"), dict)
                else {}
            )
            extracted_all = (
                update.get("extracted_entities")
                if isinstance(update.get("extracted_entities"), dict)
                else {}
            )
            updated_fields = (
                update.get("extracted_entities_updated_fields")
                if isinstance(update.get("extracted_entities_updated_fields"), list)
                else []
            )
            verified = bool(update.get("identity_verified", False))
            llm_status = str(update.get("llm_status", "")).strip()
            updated_suffix = f", updated={len([x for x in updated_fields if str(x).strip()])}" if updated_fields else ""
            llm_suffix = f"; llm_status={llm_status}" if llm_status else ""
            return (
                f"entity_extract: turn_entities={len(extracted_turn)}; "
                f"session_entities={len(extracted_all)}{updated_suffix}; "
                f"identity_verified={verified}{llm_suffix}."
            )

        if node_name == "react":
            decision = update.get("decision")
            tool_call = None
            if isinstance(decision, dict):
                tool_call = decision.get("tool_call")
            elif decision is not None:
                tool_call = getattr(decision, "tool_call", None)
            if isinstance(tool_call, dict):
                tool_name = str(tool_call.get("tool_name", "unknown_tool"))
                return f"react: selected tool `{tool_name}` for execution."
            if tool_call is not None:
                tool_name = str(getattr(tool_call, "tool_name", "unknown_tool"))
                return f"react: selected tool `{tool_name}` for execution."
            if update.get("response"):
                return "react: prepared direct response path."
            return "react: planned next action."

        if node_name == "tool_execution":
            obs = update.get("observation") if isinstance(update.get("observation"), dict) else {}
            phase = obs.get("tool_phase") if isinstance(obs.get("tool_phase"), dict) else obs
            tool_name = str(phase.get("tool_name", "unknown_tool")) if isinstance(phase, dict) else "unknown_tool"
            return f"tool_execution: executed `{tool_name}` and captured observation."

        if node_name == "plan_proposal":
            proposal = update.get("plan_proposal") if isinstance(update.get("plan_proposal"), dict) else {}
            outline = str(proposal.get("plan_outline", "")).strip()
            if outline:
                return f"plan_proposal: built plan outline - {outline}"
            intent = str(proposal.get("intent", "generic_plan"))
            return f"plan_proposal: built proposal intent={intent}."

        if node_name == "reflect":
            feedback = update.get("reflection_feedback") if isinstance(update.get("reflection_feedback"), dict) else {}
            complete = feedback.get("is_complete")
            reason = str(feedback.get("reason", "")).strip()
            llm_status = str(update.get("llm_status", "")).strip()
            completion = "complete" if complete else "needs retry"
            reason_part = f" reason={reason}" if reason else ""
            llm_part = f" llm_status={llm_status}" if llm_status else ""
            return f"reflect: validation is {completion}.{reason_part}{llm_part}".strip()

        if node_name in {"relevant_response", "irrelevant_response"}:
            target = str(update.get("response_target", state.get("response_target", "customer")))
            return f"{node_name}: packaged final response for target={target}."

        return f"{node_name}: node executed."

    def run_turn(self, user_input: str, session_id: str | None = None, sender: str = "customer") -> AgentState:
        if self.llm is None and not self.allow_deterministic_fallback:
            raise RuntimeError(
                "CollectionAgent requires an active LLM. "
                "Deterministic classification fallback is disabled."
            )
        session_key = session_id or "collection-demo-session"
        sender_norm = str(sender).strip().lower() or "customer"
        session_lock = self._get_session_lock(session_key)
        session_lock.acquire()
        try:
            memory = self.session_store.load(session_key)
            if "mode" not in memory.state:
                memory.set_state(mode="strict_collections", active_channel="sms", active_case_id="COLL-1001")
            admin_state = self._maybe_handle_admin_message(user_input=user_input, sender=sender, memory=memory, session_key=session_key)
            if admin_state is not None:
                return admin_state
            turn_index = int(memory.state.get("turn_index", 0))
            if self._should_log_customer_input(user_input=user_input, sender=sender_norm):
                self.repository.add_conversation_message(
                    ConversationMessage(
                        session_id=session_key,
                        role=sender_norm,
                        content=user_input.strip(),
                    )
                )
            self._sync_user_and_memory_context(memory=memory, user_input=user_input)
            user_id = str(memory.state.get("active_user_id", "")).strip()
            case_id = str(memory.state.get("active_case_id", "COLL-1001")).strip()
            channel = str(memory.state.get("active_channel", "sms")).strip()
            existing_conversation_plan = (
                dict(memory.state.get("active_conversation_plan", {}))
                if isinstance(memory.state.get("active_conversation_plan"), dict)
                else {}
            )
            trace = ExecutionTrace(agent_name=self.agent_name, session_id=session_key, user_input=user_input)
            try:
                with trace_turn(trace, sink=self.trace_sink):
                    state = self.graph.invoke(
                        {
                            "session_id": session_key,
                            "turn_id": str(uuid4()),
                            "user_input": user_input,
                            "memory": memory,
                            "user_id": user_id,
                            "case_id": case_id,
                            "channel": channel,
                            "message_source": sender,
                            "conversation_history": (
                                list(memory.state.get("conversation_history", []))
                                if isinstance(memory.state.get("conversation_history"), list)
                                else []
                            ),
                            "memory_targets": [{"type": "working", "enabled": True, "limit": 8}],
                            "observation": None,
                            "node_history": [],
                            "conversation_phase": "turn_started",
                            "tool_errors": [],
                            "conversation_plan": existing_conversation_plan,
                            "steps": 0,
                            "turn_index": turn_index,
                        }
                    )
                    state = self._finalize_output_state(state)
                    self._apply_post_turn_verification_state(memory=memory, state=state)
                    memory_updates: dict[str, Any] = {
                        "turn_index": turn_index + 1,
                        "last_user_input": user_input,
                        "last_agent_response": str(state.get("response", "")).strip(),
                        "last_response_target": str(state.get("response_target", "customer")).strip().lower() or "customer",
                    }
                    if isinstance(state.get("conversation_plan"), dict):
                        memory_updates["active_conversation_plan"] = dict(state["conversation_plan"])
                    memory.set_state(**memory_updates)
                    if str(state.get("response_target", "customer")).strip().lower() == "customer":
                        response_text = str(state.get("response", "")).strip()
                        if response_text:
                            self.repository.add_conversation_message(
                                ConversationMessage(
                                    session_id=session_key,
                                    role="agent",
                                    content=response_text,
                                )
                            )
                    trace.finish(status="completed")
            except Exception as exc:
                trace.finish(status="failed", error=str(exc))
                self.last_trace = trace
                self._persist_trace(trace)
                raise
            self.last_trace = trace
            self._persist_trace(trace)
            return state
        finally:
            session_lock.release()

    def run(self, user_input: str, session_id: str | None = None, sender: str = "customer") -> str:
        state = self.run_turn(user_input=user_input, session_id=session_id, sender=sender)
        return str(state.get("response", "No response generated."))

    @staticmethod
    def _should_log_customer_input(*, user_input: str, sender: str) -> bool:
        if sender in {"self", "system"}:
            return False
        text = str(user_input).strip()
        if not text:
            return False
        lowered = text.lower()
        if lowered.startswith("system loop guard") or lowered.startswith("system hard loop guard"):
            return False
        if lowered.startswith("start outbound collections call now."):
            return False
        return True

    def _finalize_output_state(self, state: AgentState) -> AgentState:
        """Guarantees output contract keys for downstream orchestration."""
        output = dict(state)
        response_target = str(output.get("response_target", "customer")).strip().lower()
        if response_target not in {"customer", "self", "discount_planning_agent"}:
            response_target = "customer"
        output["response_target"] = response_target

        response = output.get("response")
        if not isinstance(response, str) or not response.strip():
            plan = output.get("plan_proposal") if isinstance(output.get("plan_proposal"), dict) else {}
            planned = str(plan.get("draft_response", "")).strip() if plan else ""
            if planned:
                output["response"] = planned
            else:
                decision = output.get("decision")
                decision_text = str(getattr(decision, "response_text", "")).strip() if decision is not None else ""
                fallback_outline = str(plan.get("plan_outline", "")).strip() if plan else ""
                output["response"] = decision_text or fallback_outline or "No response generated."
        return output

    def _maybe_handle_admin_message(
        self,
        *,
        user_input: str,
        sender: str,
        memory: Any,
        session_key: str,
    ) -> AgentState | None:
        lowered = user_input.lower().strip()
        sender_norm = str(sender).strip().lower()
        is_admin = sender_norm == "admin" or lowered.startswith("admin:") or "initialize" in lowered
        if not is_admin:
            return None

        extracted_customer = re.search(r"(CUST-\d+)", user_input, re.IGNORECASE)
        extracted_case = re.search(r"(COLL-\d+)", user_input, re.IGNORECASE)
        extracted_channel = re.search(r"\b(voice|sms|email|whatsapp)\b", user_input, re.IGNORECASE)

        customer_id = extracted_customer.group(1).upper() if extracted_customer else str(memory.state.get("active_user_id", "")).upper()
        case_id = extracted_case.group(1).upper() if extracted_case else str(memory.state.get("active_case_id", "COLL-1001")).upper()
        channel = extracted_channel.group(1).lower() if extracted_channel else str(memory.state.get("active_channel", "voice")).lower()

        if customer_id and not extracted_case:
            case_row = self.data_store.get_case(customer_id=customer_id)
            if isinstance(case_row, dict) and case_row.get("case_id"):
                case_id = str(case_row["case_id"]).upper()

        memory.set_state(
            mode="strict_collections",
            active_user_id=customer_id or memory.state.get("active_user_id"),
            active_case_id=case_id or memory.state.get("active_case_id", "COLL-1001"),
            active_channel=channel or memory.state.get("active_channel", "voice"),
            agent_loop_blocked=False,
            agent_loop_count=0,
            active_conversation_plan={},
            identity_verified=False,
            verification_collected={},
            last_admin_message=user_input,
        )
        self._hydrate_case_context(memory=memory)

        self.last_trace = ExecutionTrace(agent_name=self.agent_name, session_id=session_key, user_input=user_input)
        self.last_trace.finish(status="completed")
        response = (
            "Admin initialization applied. "
            f"Session is set for customer {str(memory.state.get('active_user_id', 'unknown'))}, "
            f"case {str(memory.state.get('active_case_id', 'unknown'))}, channel {str(memory.state.get('active_channel', 'voice'))}. "
            "You can now continue as customer."
        )
        return {
            "session_id": session_key,
            "turn_id": str(uuid4()),
            "user_input": user_input,
            "message_source": "admin",
            "response": response,
            "response_target": "customer",
            "route": "continue",
            "conversation_phase": "admin_initialization",
            "node_history": ["admin_initialization"],
            "previous_node": "START",
            "next_node": "END",
            "steps": 0,
            "user_id": str(memory.state.get("active_user_id", "")),
            "case_id": str(memory.state.get("active_case_id", "")),
            "channel": str(memory.state.get("active_channel", "voice")),
            "timestamp_ms": int(time.time() * 1000),
        }

    def _persist_trace(self, trace: ExecutionTrace) -> None:
        target_dir = self.trace_output_dir
        if target_dir is None:
            return
        target_dir.mkdir(parents=True, exist_ok=True)
        payload = trace.to_dict()
        timestamp = trace.started_at.strftime("%Y%m%dT%H%M%S")
        trace_path = target_dir / f"{timestamp}_{trace.trace_id}.json"
        trace_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        latest_path = target_dir / "latest_trace.json"
        latest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    @classmethod
    def from_local_files(cls, base_dir: Path | None = None) -> "CollectionAgent":
        root = base_dir or Path(__file__).resolve().parent
        repository = CollectionRepository(runtime_dir=root / "runtime")
        data_store = CollectionDataStore(base_dir=root)
        return cls(repository=repository, data_store=data_store, trace_output_dir=root / "runtime" / "traces")

    def _sync_user_and_memory_context(self, *, memory: Any, user_input: str) -> None:
        case_match = re.search(r"(COLL-\d+)", user_input, re.IGNORECASE)
        if case_match:
            memory.set_state(active_case_id=case_match.group(1).upper())
        user_id = self._resolve_user_id(memory_state=dict(memory.state), user_input=user_input)
        if user_id:
            memory.set_state(active_user_id=user_id)
        self._hydrate_case_context(memory=memory)
        global_context = self.memory_repository.get_global_memory_context(limit=6) if self.memory_repository else {}
        user_context = (
            self.memory_repository.get_user_memory_context(user_id) if self.memory_repository and user_id else None
        )
        memory.set_state(
            global_key_event_memory=global_context,
            user_key_event_memory=(user_context or {}),
        )

    def _verification_cfg(self) -> dict[str, Any]:
        raw = self.verification_policy if isinstance(self.verification_policy, dict) else {}
        return {
            "required_fields": raw.get("required_fields", ["dob", "phone"]),
            "require_field_in_challenge": bool(raw.get("require_field_in_challenge", True)),
            "auto_verify_from_memory_evidence": bool(raw.get("auto_verify_from_memory_evidence", False)),
            "require_name_match_for_auto_verify": bool(raw.get("require_name_match_for_auto_verify", False)),
            "tool_only_verification": bool(raw.get("tool_only_verification", True)),
            "prefer_memory_verify": bool(raw.get("prefer_memory_verify", False)),
            "fallback_to_database_verify": bool(raw.get("fallback_to_database_verify", False)),
            "fallback_on_insufficient_entities": bool(raw.get("fallback_on_insufficient_entities", False)),
        }

    def _required_verification_fields(self, *, challenge: dict[str, Any]) -> list[str]:
        cfg = self._verification_cfg()
        configured = cfg.get("required_fields")
        fields = [str(x).strip() for x in configured] if isinstance(configured, list) else ["dob", "phone"]
        fields = [field for field in fields if field]
        if not fields:
            fields = ["dob", "phone"]
        if not bool(cfg.get("require_field_in_challenge", True)):
            return fields
        return [field for field in fields if str(challenge.get(field, "")).strip()]

    def _hydrate_case_context(self, *, memory: Any) -> None:
        memory_state = dict(memory.state)
        active_case_id = str(memory_state.get("active_case_id", "")).strip()
        active_user_id = str(memory_state.get("active_user_id", "")).strip()

        case_row = self.data_store.get_case(case_id=active_case_id) if active_case_id else None
        if not case_row and active_user_id:
            case_row = self.data_store.get_case(customer_id=active_user_id)

        customer_id = active_user_id
        if isinstance(case_row, dict):
            customer_id = str(case_row.get("customer_id", customer_id)).strip()
            memory.set_state(
                active_case_id=str(case_row.get("case_id", active_case_id or "COLL-1001")).strip().upper(),
                active_overdue_amount=float(case_row.get("overdue_amount", 0.0) or 0.0),
                active_emi_amount=float(case_row.get("emi_amount", 0.0) or 0.0),
                active_late_fee=float(case_row.get("late_fee", 0.0) or 0.0),
                active_dpd=int(case_row.get("dpd", 0) or 0),
                active_loan_id=str(case_row.get("loan_id", "")).strip(),
            )
            if customer_id:
                memory.set_state(active_user_id=customer_id)

        if customer_id:
            customer_row = self.data_store.get_customer(customer_id)
            if isinstance(customer_row, dict):
                challenge = customer_row.get("challenge") if isinstance(customer_row.get("challenge"), dict) else {}
                required_fields = self._required_verification_fields(challenge=challenge)
                challenge_cache = {
                    field: str(challenge.get(field, "")).strip()
                    for field in required_fields
                    if str(challenge.get(field, "")).strip()
                }
                memory.set_state(
                    active_customer_name=str(customer_row.get("name", memory_state.get("active_customer_name", "Customer")))
                    .strip()
                    or "Customer",
                    active_customer_phone=str(customer_row.get("phone", "")).strip(),
                    active_customer_email=str(customer_row.get("email", "")).strip(),
                    active_verification_required_fields=required_fields,
                    active_verification_challenge=challenge_cache,
                )

    def _resolve_user_id(self, *, memory_state: dict[str, Any], user_input: str) -> str | None:
        from_state = memory_state.get("active_user_id") or memory_state.get("user_id")
        if from_state is not None and str(from_state).strip():
            return str(from_state).strip()

        customer_match = re.search(r"(CUST-\d+)", user_input, re.IGNORECASE)
        if customer_match:
            return customer_match.group(1).upper()

        case_match = re.search(r"(COLL-\d+)", user_input, re.IGNORECASE)
        case_id = case_match.group(1).upper() if case_match else str(memory_state.get("active_case_id", "")).upper()
        if case_id:
            case_row = self.data_store.get_case(case_id=case_id)
            if isinstance(case_row, dict) and case_row.get("customer_id"):
                return str(case_row["customer_id"])
        return None

    def _get_session_lock(self, session_key: str) -> threading.Lock:
        with self._session_locks_guard:
            lock = self._session_locks.get(session_key)
            if lock is None:
                lock = threading.Lock()
                self._session_locks[session_key] = lock
        return lock

    def _capture_verification_evidence(self, *, memory: Any, user_input: str) -> None:
        text = str(user_input or "").strip()
        if not text:
            return
        memory_state = dict(getattr(memory, "state", {}))
        collected = (
            dict(memory_state.get("verification_entities", {}))
            if isinstance(memory_state.get("verification_entities"), dict)
            else {}
        )

        required_fields = memory_state.get("active_verification_required_fields")
        required = [str(x).strip() for x in required_fields if str(x).strip()] if isinstance(required_fields, list) else []
        if self.entity_extract_tool is not None:
            raw_entities = self.entity_extract_tool.execute(input=EntityExtractInput(text=text))
            if isinstance(raw_entities.entities, dict):
                memory.set_state(extracted_entities=dict(raw_entities.entities))
        if self.verification_entity_extract_tool is None:
            return
        extracted = self.verification_entity_extract_tool.execute(
            input=VerificationEntityExtractInput(
                text=text,
                required_fields=required,
                include_name=True,
            )
        )
        entities = extracted.entities if isinstance(extracted.entities, dict) else {}
        for key, value in entities.items():
            val = str(value).strip()
            if val:
                collected[key] = val

        active_name = str(memory_state.get("active_customer_name", "")).strip().lower()
        provided_name = str(collected.get("name", "")).strip().lower()
        if active_name and provided_name and active_name == provided_name:
            collected["name_confirmed"] = True
        elif provided_name:
            collected["name_confirmed"] = False

        if collected != dict(memory_state.get("verification_entities", {})):
            memory.set_state(
                verification_entities=collected,
                verification_collected=collected,  # compatibility key used by existing nodes/UI
            )

    def _reconcile_verification_from_collected(self, *, memory: Any) -> None:
        memory_state = dict(getattr(memory, "state", {}))
        customer_id = str(memory_state.get("active_user_id", "")).strip()
        if not customer_id:
            return
        cfg = self._verification_cfg()

        collected = (
            dict(memory_state.get("verification_entities", {}))
            if isinstance(memory_state.get("verification_entities"), dict)
            else {}
        )
        if not collected and isinstance(memory_state.get("verification_collected"), dict):
            collected = dict(memory_state.get("verification_collected", {}))

        challenge_from_memory = (
            dict(memory_state.get("active_verification_challenge", {}))
            if isinstance(memory_state.get("active_verification_challenge"), dict)
            else {}
        )
        has_memory_challenge = bool(challenge_from_memory)
        challenge = challenge_from_memory
        if not challenge:
            customer_row = self.data_store.get_customer(customer_id)
            if isinstance(customer_row, dict) and isinstance(customer_row.get("challenge"), dict):
                raw = customer_row.get("challenge", {})
                required_hint = memory_state.get("active_verification_required_fields")
                required_from_state = (
                    [str(x).strip() for x in required_hint if str(x).strip()]
                    if isinstance(required_hint, list)
                    else []
                )
                required = required_from_state or self._required_verification_fields(challenge=raw)
                challenge = {
                    field: str(raw.get(field, "")).strip()
                    for field in required
                    if str(raw.get(field, "")).strip()
                }
                memory.set_state(active_verification_challenge=challenge)
        if not challenge:
            return

        # Always bound verification fields to configured policy.
        # This prevents stale memory challenge payloads from expanding checks
        # (for example asking for email when policy is dob/phone only).
        required_fields = self._required_verification_fields(challenge=challenge)
        if not required_fields:
            required_fields = [field for field in challenge.keys() if str(field).strip()]
        challenge = {
            field: str(challenge.get(field, "")).strip()
            for field in required_fields
            if str(challenge.get(field, "")).strip()
        }
        updates: dict[str, Any] = {
            "active_verification_required_fields": required_fields,
            "active_verification_challenge": challenge,
        }
        expected_name = str(memory_state.get("active_customer_name", "")).strip()

        if bool(cfg.get("tool_only_verification", False)):
            verified_fields = (
                [str(x).strip() for x in memory_state.get("verification_verified_fields", []) if str(x).strip()]
                if isinstance(memory_state.get("verification_verified_fields"), list)
                else []
            )
            missing_tool_fields = [field for field in required_fields if field not in set(verified_fields)]
            updates["verification_verified_fields"] = sorted(set(verified_fields))
            updates["verification_missing_fields"] = sorted(set(missing_tool_fields))
            updates["identity_verified"] = not bool(missing_tool_fields)
            updates["verification_entities"] = collected
            updates["verification_collected"] = collected
            memory.set_state(**updates)
            return

        matched = False
        status = "insufficient"
        missing_required = [field for field in required_fields if not str(collected.get(field, "")).strip()]
        missing_fields_state = list(missing_required)
        mismatched_fields_state: list[str] = []
        compared_fields_state: list[str] = []

        if (not has_memory_challenge) and bool(cfg.get("fallback_to_database_verify", True)):
            if not missing_required:
                db_result = self._verify_from_database_with_entities(
                    customer_id=customer_id,
                    entities=collected,
                )
                if db_result is not None:
                    status = str(db_result.get("status", status)).strip().lower()
                    matched = status == "verified"
                    compared_fields_state = [str(x).strip() for x in db_result.get("required_fields", []) if str(x).strip()]
                    if status == "failed" and not missing_required:
                        mismatched_fields_state = list(compared_fields_state or required_fields)
        elif bool(cfg.get("prefer_memory_verify", True)) and self.verification_memory_verify_tool is not None:
            entities_for_match = {
                str(key): str(value)
                for key, value in collected.items()
                if str(key).strip() and (
                    str(key) in set(required_fields) or str(key) == "name"
                ) and str(value).strip()
            }
            memory_verify = self.verification_memory_verify_tool.execute(
                input=VerificationMemoryVerifyInput(
                    entities=entities_for_match,
                    expected_challenge=challenge,
                    required_fields=required_fields,
                    require_name_match=bool(cfg.get("require_name_match_for_auto_verify", False)),
                    expected_name=expected_name or None,
                )
            )
            status = str(memory_verify.status).strip().lower()
            matched = bool(memory_verify.matched)
            missing_fields_state = [str(x).strip() for x in memory_verify.missing_fields if str(x).strip()]
            mismatched_fields_state = [str(x).strip() for x in memory_verify.mismatched_fields if str(x).strip()]
            compared_fields_state = [str(x).strip() for x in memory_verify.compared_fields if str(x).strip()]
        should_fallback = has_memory_challenge and (status == "failed" or (
            status == "insufficient" and bool(cfg.get("fallback_on_insufficient_entities", False))
        ))
        if should_fallback and bool(cfg.get("fallback_to_database_verify", True)):
            db_result = self._verify_from_database_with_entities(
                customer_id=customer_id,
                entities=collected,
            )
            if db_result is not None:
                status = str(db_result.get("status", status)).strip().lower()
                matched = status == "verified"
                compared_fields_state = [str(x).strip() for x in db_result.get("required_fields", []) if str(x).strip()]
                if status == "failed" and not missing_fields_state:
                    mismatched_fields_state = list(compared_fields_state or required_fields)

        if not bool(cfg.get("auto_verify_from_memory_evidence", True)) and status != "verified":
            matched = False

        updates["identity_verified"] = bool(matched)
        updates["verification_entities"] = collected
        updates["verification_collected"] = collected
        updates["verification_last_status"] = ("verified" if matched else (status or "failed"))
        updates["verification_missing_fields"] = sorted(set(missing_fields_state))
        updates["verification_mismatched_fields"] = sorted(set(mismatched_fields_state))
        updates["verification_compared_fields"] = sorted(set(compared_fields_state))
        memory.set_state(**updates)

    def _verify_from_database_with_entities(self, *, customer_id: str, entities: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(entities, dict) or not entities:
            return None
        case_row = self.data_store.get_case(customer_id=customer_id)
        case_id = str(case_row.get("case_id", "")).strip() if isinstance(case_row, dict) else ""
        if not case_id:
            return None
        status_by_field: dict[str, str] = {}
        if str(entities.get("dob", "")).strip():
            dob_result = VerifyDOBTool(store=self.data_store).execute(
                input=VerifyDOBInput(
                    case_id=case_id,
                    customer_id=customer_id,
                    dob=str(entities.get("dob", "")).strip(),
                )
            )
            status_by_field["dob"] = str(dob_result.status)
        if str(entities.get("phone", "")).strip():
            phone_result = VerifyMobileTool(store=self.data_store).execute(
                input=VerifyMobileInput(
                    case_id=case_id,
                    customer_id=customer_id,
                    phone=str(entities.get("phone", "")).strip(),
                )
            )
            status_by_field["phone"] = str(phone_result.status)
        required_fields = ["dob", "phone"]
        if not status_by_field:
            return None
        if any(status_by_field.get(field) == "locked" for field in required_fields):
            status = "locked"
        elif all(status_by_field.get(field) == "verified" for field in required_fields if field in status_by_field):
            status = "verified"
        else:
            status = "failed"
        return {
            "status": status,
            "required_fields": required_fields,
            "status_by_field": status_by_field,
        }

    def _apply_post_turn_verification_state(self, *, memory: Any, state: AgentState) -> None:
        observation = state.get("observation")
        if not isinstance(observation, dict):
            return
        tool_phase = observation.get("tool_phase") if isinstance(observation.get("tool_phase"), dict) else observation
        if not isinstance(tool_phase, dict):
            return
        tool_name = str(tool_phase.get("tool_name", "")).strip()
        if tool_name not in {"verify_dob", "verify_mobile"}:
            return

        output = tool_phase.get("output") if isinstance(tool_phase.get("output"), dict) else {}
        status = str(output.get("status", "")).strip().lower()
        memory_state = dict(getattr(memory, "state", {}))
        required_fields = (
            [str(x).strip() for x in memory_state.get("active_verification_required_fields", []) if str(x).strip()]
            if isinstance(memory_state.get("active_verification_required_fields"), list)
            else ["dob", "phone"]
        )
        verified_fields = (
            [str(x).strip() for x in memory_state.get("verification_verified_fields", []) if str(x).strip()]
            if isinstance(memory_state.get("verification_verified_fields"), list)
            else []
        )
        field_name = str(output.get("field", "")).strip().lower()
        if not field_name:
            field_name = "dob" if tool_name == "verify_dob" else "phone"
        updates: dict[str, Any] = {}
        if status == "verified" and field_name:
            verified_fields = sorted(set([*verified_fields, field_name]))
        elif status in {"failed", "locked"} and field_name:
            verified_fields = [field for field in verified_fields if field != field_name]
            if isinstance(output.get("failed_attempts"), int):
                updates[f"verification_failed_attempts_{field_name}"] = int(output["failed_attempts"])
        missing_fields = [field for field in required_fields if field not in set(verified_fields)]
        updates["active_verification_required_fields"] = sorted(set(required_fields))
        updates["verification_verified_fields"] = sorted(set(verified_fields))
        updates["verification_missing_fields"] = sorted(set(missing_fields))
        updates["verification_last_status"] = status or "unknown"
        updates["identity_verified"] = not bool(missing_fields)
        if updates:
            memory.set_state(**updates)

    @staticmethod
    def _phase_for_node(node_name: str) -> str:
        phase_map = {
            "relevance_intent": "relevance_classification",
            "entity_extract": "entity_extraction",
            "pre_plan_intent": "pre_plan_routing",
            "execution_path_intent": "execution_routing",
            "memory_retrieve": "memory_hydration",
            "post_memory_plan_intent": "post_memory_routing",
            "react": "action_planning",
            "tool_execution": "tool_execution",
            "plan_proposal": "plan_proposal",
            "reflect": "quality_reflection",
            "relevant_response": "response_packaging",
            "irrelevant_response": "response_packaging",
        }
        return phase_map.get(node_name, "graph_processing")


__all__ = ["CollectionAgent"]
