from __future__ import annotations

from agents.collection_agent.nodes.collection_reflect_node import CollectionReflectNode


class FakeLLM:
    model_name = "fake-collection-reflect-llm"

    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.output


def _build_state(*, identity_verified: bool, plan_proposal: dict, conversation_plan: dict | None = None) -> dict:
    return {
        "user_input": "hello",
        "routing_context": {"plan_origin": "pre_plan_intent"},
        "plan_proposal": plan_proposal,
        "conversation_plan": conversation_plan or {},
        "response_target": plan_proposal.get("target", "customer"),
        "identity_verified": identity_verified,
        "verification_missing_fields": [] if identity_verified else ["dob", "phone"],
        "verification_entities": {},
        "customer_payment_posture": "unknown",
        "discount_stage": "none",
        "customer_payment_capacity": None,
        "customer_payment_capacity_pct": None,
        "hardship_context": {
            "hardship_detected": False,
            "hardship_reason": None,
            "confidence": 0.0,
        },
        "plan_signals": {},
        "extracted_entities_turn": {},
    }


def test_collection_reflect_node_retries_on_invalid_json() -> None:
    node = CollectionReflectNode(
        llm=FakeLLM("not json"),
        system_prompt="validate",
        user_prompt="User input: {user_input}\nObservation: {observation}\nDecision: {decision}",
    )

    update = node.execute(
        _build_state(
            identity_verified=True,
            plan_proposal={
                "target": "customer",
                "intent": "generic_plan",
                "draft_response": "What monthly amount would work for you?",
                "next_actions": ["collect_arrangement_preference"],
                "plan_tree_update": {"operation": "advance", "selected_next_node_id": "resolve_outcome"},
            },
            conversation_plan={"current_node_id": "resolve_outcome"},
        )
    )

    assert update["failure_type"] == "invalid_json"
    assert update["reflection_complete"] is False
    assert update["retry_target"] == "plan_proposal"
    assert node.route(update) == "retry_plan_proposal"


def test_collection_reflect_node_forces_completion_on_unsafe_disclosure() -> None:
    node = CollectionReflectNode(
        llm=FakeLLM('{"reason":"looks valid","is_complete":true,"failure_type":"none"}'),
        system_prompt="validate",
        user_prompt="User input: {user_input}\nObservation: {observation}\nDecision: {decision}",
    )

    update = node.execute(
        _build_state(
            identity_verified=False,
            plan_proposal={
                "target": "customer",
                "intent": "generic_plan",
                "draft_response": "Your overdue amount is INR 1200.00.",
                "next_actions": ["collect_arrangement_preference"],
                "plan_tree_update": {"operation": "advance", "selected_next_node_id": "resolve_outcome"},
            },
            conversation_plan={"current_node_id": "verify_identity"},
        )
    )

    assert update["failure_type"] == "unsafe_disclosure"
    assert update["reflection_complete"] is True
    assert update["retry_target"] == "none"
    assert node.route(update) == "complete"


def test_collection_reflect_node_forces_completion_on_missing_required_action() -> None:
    node = CollectionReflectNode(
        llm=FakeLLM('{"reason":"looks valid","is_complete":true,"failure_type":"none"}'),
        system_prompt="validate",
        user_prompt="User input: {user_input}\nObservation: {observation}\nDecision: {decision}",
    )

    update = node.execute(
        _build_state(
            identity_verified=True,
            plan_proposal={
                "target": "customer",
                "intent": "generic_plan",
                "draft_response": "Please let me know what would work for you.",
                "next_actions": [],
                "plan_tree_update": {"operation": "advance", "selected_next_node_id": ""},
            },
            conversation_plan={"current_node_id": "collect_payment_intent"},
        )
    )

    assert update["failure_type"] == "missing_required_action"
    assert update["reflection_complete"] is True
    assert update["retry_target"] == "none"
    assert node.route(update) == "complete"


def test_collection_reflect_node_forces_completion_on_invalid_negotiation_state() -> None:
    node = CollectionReflectNode(
        llm=FakeLLM('{"reason":"looks valid","is_complete":true,"failure_type":"none"}'),
        system_prompt="validate",
        user_prompt="User input: {user_input}\nObservation: {observation}\nDecision: {decision}",
    )

    state = _build_state(
        identity_verified=True,
        plan_proposal={
            "target": "customer",
            "intent": "generic_plan",
            "draft_response": "Please share what amount would work for you.",
            "next_actions": ["collect_arrangement_preference"],
            "plan_tree_update": {"operation": "advance", "selected_next_node_id": "resolve_outcome"},
        },
        conversation_plan={"current_node_id": "resolve_outcome"},
    )
    state["customer_payment_posture"] = "bad_value"
    state["discount_stage"] = "bad_stage"

    update = node.execute(state)

    assert update["failure_type"] == "invalid_state_claim"
    assert update["reflection_complete"] is True
    assert update["retry_target"] == "none"


def test_collection_reflect_node_flags_missing_capacity_when_present_in_turn() -> None:
    node = CollectionReflectNode(
        llm=FakeLLM('{"reason":"looks valid","is_complete":true,"failure_type":"none"}'),
        system_prompt="validate",
        user_prompt="User input: {user_input}\nObservation: {observation}\nDecision: {decision}",
    )

    state = _build_state(
        identity_verified=True,
        plan_proposal={
            "target": "customer",
            "intent": "generic_plan",
            "draft_response": "Please share what amount would work for you.",
            "next_actions": ["collect_arrangement_preference"],
            "plan_tree_update": {"operation": "advance", "selected_next_node_id": "resolve_outcome"},
        },
        conversation_plan={"current_node_id": "resolve_outcome"},
    )
    state["user_input"] = "I can pay 2000 today"
    state["customer_payment_posture"] = "partial_now"
    state["discount_stage"] = "requested"
    state["extracted_entities_turn"] = {"customer_payment_capacity": 2000.0}

    update = node.execute(state)

    assert update["failure_type"] == "invalid_state_claim"
    assert update["reflection_complete"] is True
    assert update["retry_target"] == "none"


def test_collection_reflect_node_flags_missing_discount_specialist_route() -> None:
    node = CollectionReflectNode(
        llm=FakeLLM('{"reason":"looks valid","is_complete":true,"failure_type":"none"}'),
        system_prompt="validate",
        user_prompt="User input: {user_input}\nObservation: {observation}\nDecision: {decision}",
    )

    state = _build_state(
        identity_verified=True,
        plan_proposal={
            "target": "customer",
            "intent": "generic_plan",
            "draft_response": "Let me know what you can manage.",
            "next_actions": ["collect_arrangement_preference"],
            "plan_tree_update": {"operation": "advance", "selected_next_node_id": "resolve_outcome"},
        },
        conversation_plan={"current_node_id": "resolve_outcome"},
    )
    state["user_input"] = "Can you settle this if I pay 2000 today?"
    state["customer_payment_posture"] = "partial_now"
    state["discount_stage"] = "requested"
    state["customer_payment_capacity"] = 2000.0
    state["plan_signals"] = {"needs_discount_specialist": True}

    update = node.execute(state)

    assert update["failure_type"] == "policy_violation"
    assert update["reflection_complete"] is True
    assert update["retry_target"] == "none"


def test_collection_reflect_node_flags_invalid_discount_handoff_payload() -> None:
    node = CollectionReflectNode(
        llm=FakeLLM('{"reason":"looks valid","is_complete":true,"failure_type":"none"}'),
        system_prompt="validate",
        user_prompt="User input: {user_input}\nObservation: {observation}\nDecision: {decision}",
    )

    state = _build_state(
        identity_verified=True,
        plan_proposal={
            "target": "discount_planning_agent",
            "intent": "discount_handoff",
            "draft_response": "",
            "next_actions": ["invoke_discount_planning"],
            "handoff_payload": {"case_id": "COLL-1001"},
            "plan_tree_update": {"operation": "advance", "selected_next_node_id": "resolve_outcome"},
        },
        conversation_plan={"current_node_id": "resolve_outcome"},
    )
    state["response_target"] = "discount_planning_agent"
    state["customer_payment_posture"] = "partial_now"
    state["discount_stage"] = "requested"

    update = node.execute(state)

    assert update["failure_type"] == "missing_required_action"
    assert update["reflection_complete"] is True
    assert update["retry_target"] == "none"
