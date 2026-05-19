from __future__ import annotations

from types import SimpleNamespace

from agents.collection_agent.nodes.plan_proposal_node import PlanProposalNode
from src.memory.types import WorkingMemory


def _build_node() -> PlanProposalNode:
    return PlanProposalNode(
        llm=None,
        strict_llm_mode=False,
    )


def _base_state(memory: WorkingMemory) -> dict:
    return {
        "user_input": "hello",
        "memory": memory,
        "steps": 0,
        "decision": SimpleNamespace(
            response_text="",
            respond_directly=True,
            response_target="customer",
            tool_call=None,
        ),
        "observation": None,
        "observations": [],
    }


def test_plan_proposal_keeps_verify_identity_when_identity_verified_false() -> None:
    memory = WorkingMemory(
        session_id="collection-plan-false",
        state={
            "mode": "strict_collections",
            "active_case_id": "COLL-1001",
            "active_user_id": "USER-1",
            "active_customer_name": "Aditi",
            "active_overdue_amount": 1200.0,
            "active_verification_required_fields": ["dob", "phone"],
            "verification_entities": {"dob": "1991-08-19", "phone": "919900001001"},
            "verification_missing_fields": [],
            "verification_verified_fields": [],
            "identity_verified": False,
        },
    )
    node = _build_node()

    update = node.execute(_base_state(memory))

    proposal = update["plan_proposal"]
    tree = proposal["plan_tree_update"]
    assert tree["selected_next_node_id"] == "verify_identity"
    assert tree["current_node_id"] == "verify_identity"
    assert proposal["next_actions"][0] == "verify_identity"


def test_plan_proposal_advances_when_identity_verified_true_even_if_missing_fields_stale() -> None:
    memory = WorkingMemory(
        session_id="collection-plan-true",
        state={
            "mode": "strict_collections",
            "active_case_id": "COLL-1001",
            "active_user_id": "USER-1",
            "active_customer_name": "Aditi",
            "active_overdue_amount": 1200.0,
            "active_verification_required_fields": ["dob", "phone"],
            "verification_entities": {"dob": "1991-08-19", "phone": "919900001001"},
            "verification_missing_fields": ["phone"],
            "verification_verified_fields": ["dob", "phone"],
            "identity_verified": True,
        },
    )
    node = _build_node()
    state = _base_state(memory)
    state["verification_missing_fields"] = ["phone"]

    update = node.execute(state)

    proposal = update["plan_proposal"]
    tree = proposal["plan_tree_update"]
    assert tree["selected_next_node_id"] == "explain_dues"
    assert tree["current_node_id"] == "explain_dues"
    assert "verify_identity" not in proposal["next_actions"]
