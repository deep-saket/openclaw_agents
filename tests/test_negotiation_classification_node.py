from __future__ import annotations

from agents.collection_agent.nodes.negotiation_classification_node import NegotiationClassificationNode
from src.memory.types import WorkingMemory


def _build_node() -> NegotiationClassificationNode:
    return NegotiationClassificationNode(
        llm=None,
        system_prompt="",
        user_prompt="",
        strict_llm_mode=False,
    )


def test_negotiation_classification_detects_hardship_and_persists_state() -> None:
    memory = WorkingMemory(
        session_id="negotiation-hardship",
        state={
            "identity_verified": True,
            "conversation_history": [
                {"role": "agent", "content": "How would you like to handle the dues today?"},
            ],
        },
    )
    node = _build_node()

    update = node.execute(
        {
            "user_input": "I lost my job and cannot afford the full payment right now.",
            "memory": memory,
            "steps": 0,
            "extracted_entities": {},
            "extracted_entities_turn": {},
            "identity_verified": True,
            "verification_missing_fields": [],
            "verification_verified_fields": ["dob", "phone"],
        }
    )

    assert update["conversation_mode"] == "hardship_negotiation"
    assert update["negotiation_stage"] == "discovering_hardship"
    assert update["customer_payment_posture"] == "cannot_pay"
    assert update["response_mode"] == "empathetic"
    assert update["active_dialogue_owner"] == "plan_proposal"
    assert update["hardship_context"]["hardship_detected"] is True
    assert update["hardship_context"]["hardship_reason"] == "job_loss"
    assert update["customer_payment_willingness"] == 0.2
    assert memory.state["conversation_mode"] == "hardship_negotiation"
    assert memory.state["response_mode"] == "empathetic"
    assert memory.state["active_dialogue_owner"] == "plan_proposal"
    assert memory.state["mode"] == "hardship_negotiation"
    assert memory.state["hardship_reason"] == "job_loss"


def test_negotiation_classification_preserves_hardship_continuity_on_follow_up_turn() -> None:
    memory = WorkingMemory(
        session_id="negotiation-follow-up",
        state={
            "identity_verified": True,
            "conversation_mode": "hardship_negotiation",
            "negotiation_stage": "discovering_hardship",
            "customer_payment_posture": "cannot_pay",
            "customer_payment_posture_history": ["cannot_pay"],
            "hardship_context": {
                "hardship_detected": True,
                "hardship_reason": "job_loss",
                "confidence": 0.96,
            },
            "response_mode": "empathetic",
            "active_dialogue_owner": "plan_proposal",
            "mode": "hardship_negotiation",
            "conversation_history": [
                {"role": "user", "content": "I lost my job last month."},
                {"role": "agent", "content": "I am sorry to hear that. What amount could be manageable?"},
            ],
        },
    )
    node = _build_node()

    update = node.execute(
        {
            "user_input": "I may be able to do 5000 monthly from next month.",
            "memory": memory,
            "steps": 0,
            "extracted_entities": {},
            "extracted_entities_turn": {"promised_amount": "5000"},
            "identity_verified": True,
            "verification_missing_fields": [],
            "verification_verified_fields": ["dob", "phone"],
        }
    )

    assert update["conversation_mode"] == "promise_capture"
    assert update["hardship_context"]["hardship_detected"] is True
    assert update["hardship_context"]["hardship_reason"] == "job_loss"
    assert update["active_dialogue_owner"] == "promise_capture"
    assert update["negotiation_stage"] == "negotiating_plan"
    assert update["customer_payment_posture"] == "promise_to_pay"
    assert "customer_payment_posture_history" not in update
    assert memory.state["conversation_mode"] == "promise_capture"
    assert memory.state["active_dialogue_owner"] == "promise_capture"
    assert memory.state["mode"] == "strict_collections"


def test_negotiation_classification_tracks_discount_request_and_counter_offer() -> None:
    memory = WorkingMemory(
        session_id="negotiation-discount",
        state={
            "identity_verified": True,
            "conversation_mode": "collections",
            "negotiation_stage": "evaluating_options",
            "customer_payment_posture": "negotiating",
            "discount_stage": "offered",
            "discount_requested": True,
            "discount_offered": True,
            "response_mode": "negotiation",
            "active_dialogue_owner": "plan_proposal",
        },
    )
    node = _build_node()

    update = node.execute(
        {
            "user_input": "What if I pay 2000 instead as settlement?",
            "memory": memory,
            "steps": 0,
            "extracted_entities": {},
            "extracted_entities_turn": {"customer_payment_capacity": "2000"},
            "identity_verified": True,
            "verification_missing_fields": [],
            "verification_verified_fields": ["dob", "phone"],
        }
    )

    assert update["customer_payment_posture"] == "negotiating"
    assert update["discount_stage"] == "counter_offer"
    assert update["counter_offer_present"] is True
    assert update["discount_requested"] is True
    assert update["discount_offered"] is True


def test_negotiation_classification_does_not_mutate_capacity_or_history_ownership() -> None:
    memory = WorkingMemory(
        session_id="negotiation-posture-history",
        state={
            "identity_verified": True,
            "conversation_mode": "hardship_negotiation",
            "negotiation_stage": "assessing_capacity",
            "customer_payment_posture": "cannot_pay",
            "customer_payment_posture_history": ["cannot_pay"],
            "customer_payment_capacity": 3000.0,
            "hardship_context": {
                "hardship_detected": True,
                "hardship_reason": "job_loss",
                "confidence": 0.96,
            },
        },
    )
    node = _build_node()

    update = node.execute(
        {
            "user_input": "I can pay 3000 today.",
            "memory": memory,
            "steps": 0,
            "extracted_entities": {},
            "extracted_entities_turn": {"customer_payment_capacity": "3000"},
            "identity_verified": True,
            "verification_missing_fields": [],
            "verification_verified_fields": ["dob", "phone"],
            "turn_index": 5,
        }
    )

    assert update["customer_payment_posture"] == "partial_now"
    assert "customer_payment_capacity" not in update
    assert "customer_payment_capacity_pct" not in update
    assert "customer_payment_posture_history" not in update
    assert memory.state["customer_payment_capacity"] == 3000.0
    assert memory.state["customer_payment_posture_history"] == ["cannot_pay"]


class _RateLimitedLLM:
    def generate(self, system_prompt: str, prompt: str) -> str:
        del system_prompt, prompt
        raise RuntimeError(
            "Error code: 429 - {'error': {'message': 'Rate limit reached for model `llama-3.1-8b-instant`', "
            "'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
        )


def test_negotiation_classification_rate_limit_falls_back_in_strict_mode() -> None:
    memory = WorkingMemory(
        session_id="negotiation-rate-limit",
        state={
            "identity_verified": True,
            "conversation_mode": "collections",
            "negotiation_stage": "none",
            "customer_payment_posture": "unknown",
        },
    )
    node = NegotiationClassificationNode(
        llm=_RateLimitedLLM(),
        system_prompt="Classify negotiation state.",
        user_prompt="User input: {user_input}",
        strict_llm_mode=True,
    )

    update = node.execute(
        {
            "user_input": "I lost my job and cannot afford the full payment right now.",
            "memory": memory,
            "steps": 0,
            "extracted_entities": {},
            "extracted_entities_turn": {},
            "identity_verified": True,
            "verification_missing_fields": [],
            "verification_verified_fields": ["dob", "phone"],
        }
    )

    assert update["conversation_mode"] == "hardship_negotiation"
    assert update["customer_payment_posture"] == "cannot_pay"
    assert update["llm_error"]
    assert memory.state["conversation_mode"] == "hardship_negotiation"
