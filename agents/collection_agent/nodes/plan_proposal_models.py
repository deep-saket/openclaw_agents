"""Structured payload models for split plan-proposal nodes."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PlanTreeNode(BaseModel):
    id: str
    label: str
    owner: str = "collection_agent"
    status: str = "pending"


class PlanTreeEdge(BaseModel):
    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    condition: str | None = None


class PlanTreeUpdate(BaseModel):
    operation: str = "advance"
    current_node_id: str | None = None
    selected_next_node_id: str | None = None
    new_nodes: list[PlanTreeNode] = Field(default_factory=list)
    new_edges: list[PlanTreeEdge] = Field(default_factory=list)
    remove_node_ids: list[str] = Field(default_factory=list)
    mark_done: list[str] = Field(default_factory=list)
    mark_skipped: list[str] = Field(default_factory=list)
    mark_blocked: list[str] = Field(default_factory=list)
    status: str | None = None


class PlanProposalPayload(BaseModel):
    target: str = "customer"
    response_target: str | None = None
    intent: str = "generic_plan"
    conversation_objective: str | None = None
    dialogue_action: str | None = None
    response_mode: str | None = None
    required_response_elements: list[str] = Field(default_factory=list)
    forbidden_dialogue_actions: list[str] = Field(default_factory=list)
    allowed_dialogue_actions: list[str] = Field(default_factory=list)
    customer_facing_goal: str | None = None
    handoff_target: str | None = None
    response_directive: dict[str, Any] | None = None
    handoff_payload: dict[str, Any] | None = None
    plan_outline: str
    draft_response: str | None = None
    next_actions: list[str] = Field(default_factory=list)
    plan_tree_update: PlanTreeUpdate | None = None


class PlanSignalPayload(BaseModel):
    needs_discount_specialist: bool = False
    is_plan_request: bool = False
    is_plan_rejection: bool = False
    hardship_signal: bool = False
    hardship_reason: str = "income_reduction"
    suggested_plan_mode: str = "strict_collections"
    customer_payment_posture: Literal[
        "unknown",
        "pay_now",
        "partial_now",
        "promise_to_pay",
        "cannot_pay",
        "refuses_to_pay",
        "negotiating",
    ] | None = None
    customer_payment_capacity: float | None = None
    customer_payment_capacity_pct: float | None = None
    discount_stage: Literal[
        "none",
        "requested",
        "planning",
        "offered",
        "accepted",
        "rejected",
        "counter_offer",
        "closed",
    ] | None = None
    discount_requested: bool | None = None
    counter_offer_present: bool | None = None
    reason: str | None = None
