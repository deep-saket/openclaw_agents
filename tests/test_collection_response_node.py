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
            "customer_payment_posture": "cannot_pay",
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
    assert update["response_render_debug"]["template_selected"] == "capacity_question"
    assert update["response_render_debug"]["response_mode"] == "empathetic"
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
            "customer_payment_posture": "cannot_pay",
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
    assert update["response_render_debug"]["template_selected"] == "safe_follow_up"
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


def test_response_node_validator_blocks_unresolved_placeholders() -> None:
    node = _build_node()

    validation = node._validate_response_against_directive(
        text="Hello [name], please confirm {missing_fields}.",
        directive={
            "template_id": "verification_request",
            "response_target": "customer",
            "tone": "compliance",
            "render_variables": {},
            "response_constraints": {"avoid_placeholders": True},
            "fallback_template_id": "verification_request",
        },
        context={
            "response_target": "customer",
            "verification_context": {"identity_verified": False},
        },
    )

    assert validation["text"] is None
    assert "unresolved_placeholders" in validation["forbidden_actions_blocked"]


def test_response_node_validator_blocks_dues_before_verification() -> None:
    node = _build_node()

    validation = node._validate_response_against_directive(
        text="Your overdue amount is INR 1200.00.",
        directive={
            "template_id": "dues_explanation",
            "response_target": "customer",
            "tone": "informational",
            "render_variables": {},
            "response_constraints": {"no_dues_before_verification": True},
            "fallback_template_id": "dues_explanation",
        },
        context={
            "response_target": "customer",
            "verification_context": {"identity_verified": False},
        },
    )

    assert validation["text"] is None
    assert "disclose_dues_before_verification" in validation["forbidden_actions_blocked"]


def test_response_node_negotiation_mode_does_not_add_empathy_language() -> None:
    node = _build_node()
    memory = WorkingMemory(
        session_id="response-negotiation-tone",
        state={
            "active_customer_name": "Aditi",
            "active_case_id": "COLL-1001",
            "active_overdue_amount": 1200.0,
            "identity_verified": True,
        },
    )
    state = {
        "user_input": "What arrangements can I request?",
        "memory": memory,
        "plan_proposal": {
            "target": "customer",
            "intent": "generic_plan",
            "response_directive": {
                "conversation_objective": "present_arrangement_options",
                "dialogue_action": "discuss_arrangement",
                "response_mode": "negotiation",
                "required_response_elements": ["discuss_arrangement"],
                "forbidden_dialogue_actions": ["restart_collections_menu"],
                "allowed_dialogue_actions": ["discuss_arrangement"],
                "customer_facing_goal": "Continue arrangement discussion directly.",
                "handoff_target": None,
            },
        },
    }

    response = node.execute(state)["response"].lower()

    assert "sorry" not in response
    assert "appreciate you sharing" not in response
    assert "installment" in response or "arrangement" in response


def test_response_node_compliance_mode_stays_verification_focused() -> None:
    node = _build_node()
    memory = WorkingMemory(
        session_id="response-compliance-tone",
        state={
            "active_customer_name": "Aditi",
            "active_case_id": "COLL-1001",
            "active_overdue_amount": 1200.0,
            "identity_verified": False,
            "active_verification_required_fields": ["dob"],
            "verification_missing_fields": ["dob"],
            "verification_entities": {},
        },
    )
    state = {
        "user_input": "Can you tell me the dues?",
        "memory": memory,
        "plan_proposal": {
            "target": "customer",
            "intent": "generic_plan",
            "response_directive": {
                "conversation_objective": "collect_verification",
                "dialogue_action": "ask_verification",
                "response_mode": "compliance",
                "required_response_elements": ["ask_verification"],
                "forbidden_dialogue_actions": ["disclose_dues_before_verification"],
                "allowed_dialogue_actions": ["ask_verification"],
                "customer_facing_goal": "Ask only for the missing verification detail.",
                "handoff_target": None,
            },
        },
    }

    response = node.execute(state)["response"].lower()

    assert "sorry" not in response
    assert "overdue amount" not in response
    assert "date of birth" in response


def test_response_node_uses_recent_conversation_window_from_history() -> None:
    node = _build_node()
    memory = WorkingMemory(
        session_id="response-recent-conversation",
        state={
            "active_customer_name": "Aditi",
            "conversation_history": [
                {"role": "customer", "content": "turn1 customer"},
                {"role": "agent", "content": "turn1 agent"},
                {"role": "customer", "content": "turn2 customer"},
                {"role": "agent", "content": "turn2 agent"},
                {"role": "customer", "content": "turn3 customer"},
                {"role": "agent", "content": "turn3 agent"},
                {"role": "customer", "content": "turn4 customer"},
                {"role": "agent", "content": "turn4 agent"},
            ],
        },
    )
    state = {
        "user_input": "current user input",
        "memory": memory,
        "plan_proposal": {"target": "customer", "intent": "generic_plan"},
    }

    context = node._resolve_render_context(state=state, proposal=state["plan_proposal"])

    assert context["recent_conversation"] == [
        {"role": "customer", "content": "turn2 customer"},
        {"role": "agent", "content": "turn2 agent"},
        {"role": "customer", "content": "turn3 customer"},
        {"role": "agent", "content": "turn3 agent"},
        {"role": "customer", "content": "turn4 customer"},
        {"role": "agent", "content": "turn4 agent"},
    ]
