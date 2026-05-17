"""Collection-agent specific graph state contract."""

from __future__ import annotations

from typing import Any, TypedDict

from src.nodes.types import AgentState


class CollectionGraphState(AgentState, total=False):
    """Agent-specific state keys used by collection graph nodes.

    This keeps the shared framework state available while making collection
    routing fields explicit and non-overlapping.
    """

    # Session/user context
    user_id: str
    case_id: str
    channel: str
    message_source: str

    # Namespaced intent outputs to avoid key overwrite between stages
    relevance_intent: dict[str, Any]
    pre_plan_intent: dict[str, Any]
    execution_path_intent: dict[str, Any]
    post_memory_plan_intent: dict[str, Any]

    # Per-turn diagnostics and orchestration helpers
    node_history: list[str]
    previous_node: str
    next_node: str | list[str]
    conversation_phase: str
    tool_errors: list[dict[str, Any]]
    response_metadata: dict[str, Any]
    plan_proposal: dict[str, Any]
    conversation_plan: dict[str, Any]
    extracted_entities: dict[str, Any]
    extracted_entity_descriptions: dict[str, Any]
    verification_entities: dict[str, Any]
    verification_missing_fields: list[str]
    identity_verified: bool


CollectionNodeUpdate = CollectionGraphState


__all__ = ["CollectionGraphState", "CollectionNodeUpdate"]
