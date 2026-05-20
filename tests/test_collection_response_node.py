from __future__ import annotations

from agents.collection_agent.nodes.collection_response_node import CollectionResponseNode
from src.memory.types import WorkingMemory


def _build_node() -> CollectionResponseNode:
    return CollectionResponseNode(
        llm=None,
        strict_llm_mode=False,
    )


def test_response_node_uses_hardship_objective_fallback() -> None:
    node = _build_node()
    memory = WorkingMemory(
        session_id="response-hardship",
        state={
            "active_customer_name": "Aditi",
            "active_case_id": "COLL-1001",
            "active_overdue_amount": 1200.0,
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
            "identity_verified": True,
        },
    )
    state = {
        "user_input": "I lost my job",
        "memory": memory,
        "plan_proposal": {
            "target": "customer",
            "intent": "generic_plan",
        },
    }

    response = node.execute(state)["response"]

    lowered = response.lower()
    assert "pay now" not in lowered
    assert "schedule a follow-up" not in lowered
    assert "monthly amount" in lowered
    assert "realistically work for you right now" in lowered


def test_response_node_collects_missing_verification_fields_only() -> None:
    node = _build_node()
    memory = WorkingMemory(
        session_id="response-verification",
        state={
            "active_customer_name": "Aditi",
            "active_case_id": "COLL-1001",
            "active_overdue_amount": 1200.0,
            "conversation_mode": "verification",
            "negotiation_stage": "none",
            "customer_payment_posture": "unknown",
            "hardship_context": {
                "hardship_detected": False,
                "hardship_reason": None,
                "confidence": 0.0,
            },
            "response_mode": "compliance",
            "active_dialogue_owner": "verification",
            "identity_verified": False,
            "active_verification_required_fields": ["dob", "phone"],
            "verification_missing_fields": ["dob", "phone"],
            "verification_entities": {},
        },
    )
    state = {
        "user_input": "hello",
        "memory": memory,
        "plan_proposal": {
            "target": "customer",
            "intent": "generic_plan",
        },
    }

    response = node.execute(state)["response"].lower()

    assert "overdue amount" not in response
    assert "date of birth" in response
    assert "registered phone number" in response


def test_response_node_minimal_safety_cleanup_strips_internal_leakage() -> None:
    node = _build_node()

    cleaned = node._apply_minimal_safety_cleanup(
        text="Please wait while I evaluate. ```json {\"foo\": \"bar\"}```",
        context={},
        directive={},
    )

    lowered = cleaned.lower()
    assert "please wait while i evaluate" not in lowered
    assert "```" not in cleaned
