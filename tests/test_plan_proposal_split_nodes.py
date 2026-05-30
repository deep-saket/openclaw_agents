from __future__ import annotations

from agents.collection_agent.agent import CollectionAgent
from agents.collection_agent.nodes.plan_proposal_directive_node import PlanProposalDirectiveNode
from agents.collection_agent.nodes.plan_proposal_graph_node import PlanProposalGraphNode
from agents.collection_agent.nodes.plan_proposal_state_node import PlanProposalStateNode
from src.nodes.base import BaseGraphNode
from src.memory.types import WorkingMemory


def _memory(state: dict) -> WorkingMemory:
    return WorkingMemory(session_id="plan-split", state=state)


def _base_state(memory: WorkingMemory, user_input: str = "hello") -> dict:
    return {
        "user_input": user_input,
        "memory": memory,
        "steps": 0,
        "observation": None,
        "observations": [],
    }


def _run_split_chain(base_memory_state: dict, user_input: str) -> tuple[dict, dict, dict]:
    memory = _memory(base_memory_state)
    state = _base_state(memory, user_input=user_input)
    state_node = PlanProposalStateNode(llm=None, strict_llm_mode=False)
    graph_node = PlanProposalGraphNode(llm=None, strict_llm_mode=False)
    directive_node = PlanProposalDirectiveNode(llm=None, strict_llm_mode=False)
    state_update = state_node.execute(state)
    graph_input = {**state, **state_update}
    graph_update = graph_node.execute(graph_input)
    directive_input = {**state, **state_update, **graph_update}
    directive_update = directive_node.execute(directive_input)
    return state_update, graph_update, directive_update


def test_split_plan_nodes_use_base_graph_inheritance_only() -> None:
    state_node = PlanProposalStateNode(llm=None, strict_llm_mode=False)
    graph_node = PlanProposalGraphNode(llm=None, strict_llm_mode=False)
    directive_node = PlanProposalDirectiveNode(llm=None, strict_llm_mode=False)

    for node in (state_node, graph_node, directive_node):
        assert isinstance(node, BaseGraphNode)
        assert type(node).__bases__ == (BaseGraphNode,)


def test_plan_proposal_state_node_outputs_signals_and_mode() -> None:
    memory = _memory(
        {
            "mode": "strict_collections",
            "active_case_id": "COLL-1001",
            "active_user_id": "USER-1",
            "active_customer_name": "Aditi",
            "active_overdue_amount": 1200.0,
            "conversation_mode": "hardship_negotiation",
            "negotiation_stage": "assessing_capacity",
            "customer_payment_posture": "partial_now",
            "customer_payment_capacity": 3000.0,
            "customer_payment_capacity_pct": 25.0,
            "discount_stage": "requested",
            "discount_requested": True,
            "counter_offer_present": True,
            "hardship_context": {
                "hardship_detected": True,
                "hardship_reason": "job_loss",
                "confidence": 0.96,
            },
            "response_mode": "empathetic",
            "active_dialogue_owner": "plan_proposal",
            "identity_verified": True,
        }
    )
    node = PlanProposalStateNode(llm=None, strict_llm_mode=False)

    update = node.execute(_base_state(memory, user_input="I lost my job"))

    assert update["plan_mode"] == "hardship_negotiation"
    assert update["plan_signals"]["hardship_signal"] is True
    assert update["plan_signals"]["suggested_plan_mode"] == "hardship_negotiation"
    assert update["plan_signals"]["customer_payment_posture"] == "partial_now"
    assert update["plan_signals"]["customer_payment_capacity"] == 3000.0
    assert update["plan_signals"]["customer_payment_capacity_pct"] == 25.0
    assert update["plan_signals"]["discount_stage"] == "requested"
    assert update["plan_signals"]["discount_requested"] is True
    assert update["plan_signals"]["counter_offer_present"] is True
    assert update["effective_identity_verified"] is True


def test_plan_proposal_graph_node_keeps_identity_gate_in_plan_tree() -> None:
    state_update, graph_update, _ = _run_split_chain(
        {
            "mode": "strict_collections",
            "active_case_id": "COLL-1001",
            "active_user_id": "USER-1",
            "active_customer_name": "Aditi",
            "active_overdue_amount": 1200.0,
            "active_verification_required_fields": ["dob", "phone"],
            "verification_entities": {"dob": "1991-08-19"},
            "verification_missing_fields": ["phone"],
            "verification_verified_fields": ["dob"],
            "identity_verified": False,
        },
        user_input="hello",
    )

    plan = graph_update["conversation_plan"]
    assert state_update["effective_identity_verified"] is False
    assert plan["current_node_id"] == "verify_identity"
    assert graph_update["plan_tree_context"]["current_node_id"] == "verify_identity"


def test_plan_proposal_directive_node_returns_response_directive() -> None:
    _, _, directive_update = _run_split_chain(
        {
            "mode": "strict_collections",
            "active_case_id": "COLL-1001",
            "active_user_id": "USER-1",
            "active_customer_name": "Aditi",
            "active_overdue_amount": 1200.0,
            "active_verification_required_fields": ["dob", "phone"],
            "verification_entities": {"dob": "1991-08-19"},
            "verification_missing_fields": ["phone"],
            "verification_verified_fields": ["dob"],
            "identity_verified": False,
        },
        user_input="hello",
    )

    proposal = directive_update["plan_proposal"]
    directive = proposal["response_directive"]
    assert proposal["plan_tree_update"]["selected_next_node_id"] == "verify_identity"
    assert directive["conversation_objective"] == "collect_verification"
    assert directive["dialogue_action"] == "ask_verification"


def test_plan_proposal_directive_uses_hardship_arrangement_directive() -> None:
    _, _, directive_update = _run_split_chain(
        {
            "mode": "strict_collections",
            "active_case_id": "COLL-1001",
            "active_user_id": "USER-1",
            "active_customer_name": "Aditi",
            "active_overdue_amount": 1200.0,
            "active_verification_required_fields": ["dob", "phone"],
            "identity_verified": True,
            "conversation_mode": "hardship_negotiation",
            "negotiation_stage": "assessing_capacity",
            "customer_payment_posture": "negotiating",
            "hardship_context": {
                "hardship_detected": True,
                "hardship_reason": "job_loss",
                "confidence": 0.96,
            },
            "response_mode": "empathetic",
            "active_dialogue_owner": "plan_proposal",
        },
        user_input="I lost my job",
    )

    directive = directive_update["plan_proposal"]["response_directive"]
    assert directive["conversation_objective"] == "assess_affordability"
    assert directive["dialogue_action"] == "ask_affordable_amount"
    assert "ask_pay_now_or_arrangement" in directive["forbidden_dialogue_actions"]
    assert "monthly amount" in directive["customer_facing_goal"].lower()


def test_plan_proposal_directive_discount_handoff_remains_intact() -> None:
    _, _, directive_update = _run_split_chain(
        {
            "mode": "strict_collections",
            "active_case_id": "COLL-1001",
            "active_user_id": "USER-1",
            "active_customer_name": "Aditi",
            "active_overdue_amount": 1200.0,
            "identity_verified": True,
            "conversation_mode": "collections",
            "negotiation_stage": "none",
            "customer_payment_posture": "unknown",
            "hardship_context": {
                "hardship_detected": False,
                "hardship_reason": None,
                "confidence": 0.0,
            },
            "response_mode": "informational",
            "active_dialogue_owner": "collections",
        },
        user_input="Can I get a discount or waiver?",
    )

    proposal = directive_update["plan_proposal"]
    assert proposal["target"] == "discount_planning_agent"
    assert directive_update["response_target"] == "discount_planning_agent"
    assert directive_update["handoff_payload"]["case_id"] == "COLL-1001"


def test_plan_proposal_directive_routes_partial_payment_to_discount_planning() -> None:
    _, _, directive_update = _run_split_chain(
        {
            "mode": "strict_collections",
            "active_case_id": "COLL-1001",
            "active_user_id": "USER-1",
            "active_customer_name": "Aditi",
            "active_overdue_amount": 1200.0,
            "identity_verified": True,
            "conversation_mode": "collections",
            "negotiation_stage": "evaluating_options",
            "customer_payment_posture": "partial_now",
            "customer_payment_capacity": 2000.0,
            "discount_stage": "requested",
            "discount_requested": True,
            "response_mode": "negotiation",
            "active_dialogue_owner": "plan_proposal",
        },
        user_input="I can pay 2000 today if you can settle this.",
    )

    assert directive_update["response_target"] == "discount_planning_agent"
    assert directive_update["handoff_payload"]["customer_payment_capacity"] == 2000.0
    assert directive_update["handoff_payload"]["discount_stage"] == "requested"
    assert directive_update["handoff_payload"]["customer_payment_posture"] == "partial_now"


def test_plan_proposal_directive_termination_remains_intact() -> None:
    _, _, directive_update = _run_split_chain(
        {
            "mode": "strict_collections",
            "active_case_id": "COLL-1001",
            "active_user_id": "USER-1",
            "active_customer_name": "Aditi",
            "active_overdue_amount": 1200.0,
            "identity_verified": True,
            "conversation_mode": "collections",
            "negotiation_stage": "none",
            "customer_payment_posture": "unknown",
            "hardship_context": {
                "hardship_detected": False,
                "hardship_reason": None,
                "confidence": 0.0,
            },
            "response_mode": "informational",
            "active_dialogue_owner": "collections",
        },
        user_input="bye",
    )

    proposal = directive_update["plan_proposal"]
    assert proposal["intent"] == "conversation_termination"
    assert proposal["plan_tree_update"]["operation"] == "complete"


def test_collection_agent_graph_wires_split_plan_nodes() -> None:
    agent = CollectionAgent.from_local_files()
    graph = agent.graph.get_graph()
    node_ids = set(graph.nodes.keys())
    edge_pairs = {(edge.source, edge.target) for edge in graph.edges}

    assert "plan_proposal" not in node_ids
    assert {"plan_proposal_state", "plan_proposal_graph", "plan_proposal_directive"}.issubset(node_ids)
    assert ("pre_plan_intent", "plan_proposal_state") in edge_pairs
    assert ("post_memory_plan_intent", "plan_proposal_state") in edge_pairs
    assert ("post_verification_intent", "plan_proposal_state") in edge_pairs
    assert ("react", "plan_proposal_state") in edge_pairs
    assert ("reflect", "plan_proposal_state") in edge_pairs
    assert ("plan_proposal_state", "plan_proposal_graph") in edge_pairs
    assert ("plan_proposal_graph", "plan_proposal_directive") in edge_pairs
    assert ("plan_proposal_directive", "reflect") in edge_pairs
