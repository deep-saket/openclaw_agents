from __future__ import annotations

from types import SimpleNamespace

from agents.collection_agent.nodes.plan_proposal_directive_node import PlanProposalDirectiveNode
from agents.collection_agent.nodes.plan_proposal_graph_node import PlanProposalGraphNode
from agents.collection_agent.nodes.plan_proposal_state_node import PlanProposalStateNode
from src.memory.types import WorkingMemory


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


def _run_chain(memory: WorkingMemory, *, user_input: str = "hello") -> dict:
    state = _base_state(memory)
    state["user_input"] = user_input
    state_node = PlanProposalStateNode(llm=None, strict_llm_mode=False)
    graph_node = PlanProposalGraphNode(llm=None, strict_llm_mode=False)
    directive_node = PlanProposalDirectiveNode(llm=None, strict_llm_mode=False)
    state_update = state_node.execute(state)
    graph_update = graph_node.execute({**state, **state_update})
    return directive_node.execute({**state, **state_update, **graph_update})


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

    update = _run_chain(memory)

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
    state = _base_state(memory)
    state["verification_missing_fields"] = ["phone"]

    state_node = PlanProposalStateNode(llm=None, strict_llm_mode=False)
    graph_node = PlanProposalGraphNode(llm=None, strict_llm_mode=False)
    directive_node = PlanProposalDirectiveNode(llm=None, strict_llm_mode=False)
    state_update = state_node.execute(state)
    graph_update = graph_node.execute({**state, **state_update})
    update = directive_node.execute({**state, **state_update, **graph_update})

    proposal = update["plan_proposal"]
    tree = proposal["plan_tree_update"]
    assert tree["selected_next_node_id"] == "explain_dues"
    assert tree["current_node_id"] == "explain_dues"
    assert "verify_identity" not in proposal["next_actions"]


def test_plan_proposal_attaches_hardship_response_directive() -> None:
    memory = WorkingMemory(
        session_id="collection-plan-directive",
        state={
            "mode": "strict_collections",
            "active_case_id": "COLL-1001",
            "active_user_id": "USER-1",
            "active_customer_name": "Aditi",
            "active_overdue_amount": 1200.0,
            "active_verification_required_fields": ["dob", "phone"],
            "identity_verified": True,
            "conversation_mode": "hardship_negotiation",
            "negotiation_stage": "assessing_capacity",
            "customer_payment_posture": "needs_arrangement",
            "hardship_context": {
                "hardship_detected": True,
                "hardship_reason": "job_loss",
                "confidence": 0.96,
            },
            "response_mode": "empathetic",
            "active_dialogue_owner": "plan_proposal",
        },
    )
    state = _base_state(memory)
    state["conversation_mode"] = "hardship_negotiation"
    state["negotiation_stage"] = "assessing_capacity"
    state["customer_payment_posture"] = "needs_arrangement"
    state["hardship_context"] = dict(memory.state["hardship_context"])
    state["response_mode"] = "empathetic"
    state["active_dialogue_owner"] = "plan_proposal"

    state_node = PlanProposalStateNode(llm=None, strict_llm_mode=False)
    graph_node = PlanProposalGraphNode(llm=None, strict_llm_mode=False)
    directive_node = PlanProposalDirectiveNode(llm=None, strict_llm_mode=False)
    state_update = state_node.execute(state)
    graph_update = graph_node.execute({**state, **state_update})
    update = directive_node.execute({**state, **state_update, **graph_update})

    proposal = update["plan_proposal"]
    directive = proposal["response_directive"]
    assert directive["conversation_objective"] == "assess_affordability"
    assert directive["dialogue_action"] == "ask_affordable_amount"
    assert directive["response_mode"] == "empathetic"
    assert "ask_affordable_amount" in directive["required_response_elements"]
    assert "ask_pay_now_or_arrangement" in directive["forbidden_dialogue_actions"]


def test_plan_proposal_falls_back_on_rate_limit_even_in_strict_mode(monkeypatch) -> None:
    memory = WorkingMemory(
        session_id="collection-plan-rate-limit",
        state={
            "mode": "strict_collections",
            "active_case_id": "COLL-1001",
            "active_user_id": "USER-1",
            "active_customer_name": "Aditi",
            "active_overdue_amount": 1200.0,
            "identity_verified": True,
            "conversation_mode": "hardship_negotiation",
            "negotiation_stage": "assessing_capacity",
            "customer_payment_posture": "needs_arrangement",
            "hardship_context": {
                "hardship_detected": True,
                "hardship_reason": "job_loss",
                "confidence": 0.96,
            },
            "response_mode": "empathetic",
            "active_dialogue_owner": "plan_proposal",
        },
    )
    node = PlanProposalDirectiveNode(
        llm=object(),
        strict_llm_mode=True,
    )

    def _fake_llm_build(**kwargs):
        del kwargs
        node.last_debug["llm_error"] = (
            "primary=Structured output failed after retries: Error code: 429 - "
            "{'error': {'message': 'Rate limit reached', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
        )
        return None

    monkeypatch.setattr(node, "_build_plan_proposal_with_llm", _fake_llm_build)

    proposal = node._build_plan_proposal(
        state=_base_state(memory),
        user_input="I lost my job",
        memory_state=dict(memory.state),
        observation=None,
        decision=SimpleNamespace(
            response_text="",
            respond_directly=False,
            response_target="customer",
            tool_call=None,
        ),
        default_plan="Plan for COLL-1001: validate hardship constraints and continue arrangement negotiation.",
        plan_origin="react",
        mode="hardship_negotiation",
        existing_plan={},
    )

    assert proposal["intent"] == "generic_plan"
    assert proposal["plan_outline"]
    assert proposal["context"]["conversation_mode"] == "hardship_negotiation"
