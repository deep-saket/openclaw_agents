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
    assert update["customer_payment_posture"] == "needs_arrangement"
    assert update["response_mode"] == "empathetic"
    assert update["active_dialogue_owner"] == "plan_proposal"
    assert update["hardship_context"]["hardship_detected"] is True
    assert update["hardship_context"]["hardship_reason"] == "job_loss"
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
            "customer_payment_posture": "needs_arrangement",
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

    assert update["conversation_mode"] == "hardship_negotiation"
    assert update["hardship_context"]["hardship_detected"] is True
    assert update["hardship_context"]["hardship_reason"] == "job_loss"
    assert update["active_dialogue_owner"] == "plan_proposal"
    assert update["negotiation_stage"] == "negotiating_plan"
    assert memory.state["conversation_mode"] == "hardship_negotiation"
    assert memory.state["active_dialogue_owner"] == "plan_proposal"
    assert memory.state["mode"] == "hardship_negotiation"
