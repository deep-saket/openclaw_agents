from __future__ import annotations

from agents.collection_agent.nodes.collection_response_node import CollectionResponseNode
from src.memory.types import WorkingMemory


def _build_node() -> CollectionResponseNode:
    return CollectionResponseNode(
        llm=None,
        strict_llm_mode=False,
    )


def test_response_node_renders_from_hardship_response_directive() -> None:
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
            "response_directive": {
                "conversation_objective": "assess_affordability",
                "dialogue_action": "ask_affordable_amount",
                "response_mode": "empathetic",
                "required_response_elements": ["acknowledge_hardship", "ask_affordable_amount"],
                "forbidden_dialogue_actions": ["restart_collections_menu", "ask_pay_now_or_arrangement"],
                "allowed_dialogue_actions": ["acknowledge_hardship", "ask_affordable_amount"],
                "customer_facing_goal": "Ask the customer what monthly amount is manageable.",
                "handoff_target": None,
            },
        },
    }

    update = node.execute(state)
    response = update["response"]

    lowered = response.lower()
    assert "pay now" not in lowered
    assert "schedule a follow-up" not in lowered
    assert "monthly amount" in lowered
    assert "realistically work for you right now" in lowered
    assert update["response_render_debug"]["conversation_objective"] == "assess_affordability"
    assert update["response_render_debug"]["template_selected"] == "affordability_question"
    assert update["response_render_debug"]["renderer_fallback_used"] is True


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


def test_response_node_missing_directive_does_not_infer_hardship_objective() -> None:
    node = _build_node()
    memory = WorkingMemory(
        session_id="response-no-directive",
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

    update = node.execute(state)
    response = update["response"].lower()

    assert "monthly amount" not in response
    assert "pay now" not in response
    assert "please let me know how you would like to proceed" in response
    assert update["response_render_debug"]["conversation_objective"] == "close_conversation"
    assert update["response_render_debug"]["renderer_fallback_used"] is True


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


def test_response_node_validator_blocks_dialogue_action_mismatch() -> None:
    node = _build_node()

    validation = node._validate_response_against_directive(
        text="Thank you. What amount and payment date can you confidently commit to for the next step?",
        directive={
            "conversation_objective": "assess_affordability",
            "dialogue_action": "ask_affordable_amount",
            "response_mode": "empathetic",
            "required_response_elements": ["ask_affordable_amount"],
            "forbidden_dialogue_actions": [],
            "allowed_dialogue_actions": ["ask_affordable_amount"],
        },
        context={
            "response_target": "customer",
            "negotiation_stage": "assessing_capacity",
            "verification_context": {"identity_verified": True},
        },
    )

    assert validation["text"] is None
    assert "dialogue_action_mismatch" in validation["forbidden_actions_blocked"]


def test_response_node_validator_blocks_stage_contradiction() -> None:
    node = _build_node()

    validation = node._validate_response_against_directive(
        text="Thank you. Your overdue amount is INR 1200.00. What amount and payment date can you confidently commit to?",
        directive={
            "conversation_objective": "assess_affordability",
            "dialogue_action": "ask_affordable_amount",
            "response_mode": "empathetic",
            "required_response_elements": ["ask_affordable_amount"],
            "forbidden_dialogue_actions": [],
            "allowed_dialogue_actions": ["ask_affordable_amount"],
        },
        context={
            "response_target": "customer",
            "negotiation_stage": "assessing_capacity",
            "verification_context": {"identity_verified": True},
        },
    )

    assert validation["text"] is None
    assert "stage_contradiction" in validation["forbidden_actions_blocked"]
