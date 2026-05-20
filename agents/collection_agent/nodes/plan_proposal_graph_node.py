"""Plan proposal graph mutation node for collections planning."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from agents.collection_agent.nodes.plan_proposal_utils import (
    effective_mode,
    fresh_debug_state,
    get_existing_conversation_plan,
    is_conversation_termination,
    latest_observation,
    overlay_negotiation_state_from_graph,
    overlay_verification_state_from_graph,
)
from src.nodes.base import BaseGraphNode
from src.nodes.types import AgentState, NodeUpdate


@dataclass(slots=True)
class PlanProposalGraphNode(BaseGraphNode):
    """Owns conversation plan graph mutation and step-marker reconciliation."""

    llm: Any | None = None
    system_prompt: str = ""
    user_prompt: str = ""
    classifier_system_prompt: str = ""
    classifier_user_prompt: str = ""
    strict_llm_mode: bool = True
    max_json_chars: int = 900
    last_debug: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    plan_graph_debug: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def execute(self, state: AgentState) -> NodeUpdate:
        self._record_llm_usage(state, node_name="plan_proposal_graph")
        self.last_debug = fresh_debug_state()
        memory = state.get("memory")
        memory_state = dict(getattr(memory, "state", {})) if memory is not None else {}
        prepared_memory_state = (
            dict(state.get("plan_prepared_memory_state"))
            if isinstance(state.get("plan_prepared_memory_state"), dict)
            else overlay_negotiation_state_from_graph(
                state=state,
                memory_state=overlay_verification_state_from_graph(state=state, memory_state=memory_state),
            )
        )
        existing_plan = get_existing_conversation_plan(state=state, memory_state=prepared_memory_state)
        plan_signals = state.get("plan_signals") if isinstance(state.get("plan_signals"), dict) else {}
        plan_mode = str(
            state.get(
                "plan_mode",
                effective_mode(
                    memory_state=prepared_memory_state,
                    default=str(memory_state.get("mode", "strict_collections")),
                ),
            )
        ).strip()
        plan_origin = str(state.get("plan_origin", "react")).strip() or "react"
        user_input = str(state.get("user_input", ""))
        observation = latest_observation(state)
        observed_tool = str(state.get("observed_tool", "")).strip()
        observed_tool_output = (
            dict(state.get("observed_tool_output"))
            if isinstance(state.get("observed_tool_output"), dict)
            else (observation.get("output", {}) if isinstance(observation, dict) and isinstance(observation.get("output"), dict) else {})
        )
        route = str(state.get("route", "continue")).strip().lower() or "continue"
        response_target = str(state.get("response_target", "customer")).strip().lower() or "customer"

        plan = self._build_or_update_conversation_plan(
            existing_plan=existing_plan,
            user_input=user_input,
            memory_state=prepared_memory_state,
            mode=plan_mode,
            plan_origin=plan_origin,
            response_target=response_target,
            route=route,
            observed_tool=observed_tool,
            proposal={},
            plan_signals=plan_signals,
        )
        if memory is not None:
            memory.set_state(active_conversation_plan=plan)
        self.plan_graph_debug = {
            "plan_id": plan.get("plan_id"),
            "version": plan.get("version"),
            "current_node_id": plan.get("current_node_id"),
            "next_node_ids": plan.get("next_node_ids"),
            "route": route,
            "plan_mode": plan_mode,
            "plan_origin": plan_origin,
            "observed_tool": observed_tool,
            "observed_tool_output": observed_tool_output,
            "plan_signals": plan_signals,
        }
        return {
            "route": "continue",
            "response_target": response_target,
            "conversation_plan": plan,
            "plan_tree_context": self._compact_conversation_plan(plan),
            "plan_graph_debug": dict(self.plan_graph_debug),
        }

    def route(self, state: AgentState) -> str:
        return str(state.get("route", "continue")).strip().lower() or "continue"

    @staticmethod
    def _compact_conversation_plan(plan: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(plan, dict):
            return {}
        nodes_raw = plan.get("nodes") if isinstance(plan.get("nodes"), list) else []
        edges_raw = plan.get("edges") if isinstance(plan.get("edges"), list) else []
        markers = plan.get("step_markers") if isinstance(plan.get("step_markers"), dict) else {}
        nodes = []
        for node in nodes_raw[:10]:
            if not isinstance(node, dict):
                continue
            nodes.append(
                {
                    "id": str(node.get("id", "")).strip(),
                    "label": str(node.get("label", "")).strip(),
                    "status": str(node.get("status", "")).strip(),
                    "owner": str(node.get("owner", "")).strip(),
                }
            )
        edges = []
        for edge in edges_raw[:16]:
            if not isinstance(edge, dict):
                continue
            edges.append(
                {
                    "from": str(edge.get("from", "")).strip(),
                    "to": str(edge.get("to", "")).strip(),
                    "condition": str(edge.get("condition", "")).strip(),
                }
            )
        marker_view = {}
        for key, raw in markers.items():
            if len(marker_view) >= 12:
                break
            if not isinstance(raw, dict):
                continue
            marker_view[str(key)] = str(raw.get("state", "pending")).strip()
        return {
            "plan_id": str(plan.get("plan_id", "")).strip(),
            "version": int(plan.get("version", 1) or 1),
            "status": str(plan.get("status", "active")).strip(),
            "current_node_id": str(plan.get("current_node_id", "")).strip(),
            "next_node_ids": [str(x).strip() for x in (plan.get("next_node_ids") or []) if str(x).strip()][:6],
            "nodes": nodes,
            "edges": edges,
            "step_markers": marker_view,
        }

    def _build_or_update_conversation_plan(
        self,
        *,
        existing_plan: dict[str, Any],
        user_input: str,
        memory_state: dict[str, Any],
        mode: str,
        plan_origin: str,
        response_target: str,
        route: str,
        observed_tool: str,
        proposal: dict[str, Any],
        plan_signals: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        plan = self._create_initial_plan_graph(memory_state=memory_state, mode=mode) if not existing_plan else dict(existing_plan)
        plan.setdefault("nodes", [])
        plan.setdefault("edges", [])
        plan.setdefault("timeline", [])
        plan.setdefault("revision_log", [])
        plan.setdefault("status", "active")
        plan.setdefault("version", 1)
        plan.setdefault("objective", "Drive compliant collections conversation toward payment resolution.")
        plan.setdefault("plan_id", f"plan-{str(memory_state.get('active_case_id', 'COLL-1001')).strip().upper()}")
        plan.setdefault("root_node_id", "open_and_context")
        plan.setdefault("step_markers", {})

        previous_current = str(plan.get("current_node_id", "")) or str(plan.get("root_node_id", "open_and_context"))
        plan_update = proposal.get("plan_tree_update") if isinstance(proposal.get("plan_tree_update"), dict) else {}
        proposal_context = proposal.get("context") if isinstance(proposal.get("context"), dict) else {}
        observed_output = (
            proposal_context.get("observed_tool_output")
            if isinstance(proposal_context.get("observed_tool_output"), dict)
            else {}
        )
        inferred_next = self._infer_current_node_id(
            user_input=user_input,
            observed_tool=observed_tool,
            response_target=response_target,
            route=route,
            proposal=proposal,
            plan_update=plan_update,
            previous_current=previous_current,
        )
        self._apply_plan_tree_update(plan=plan, plan_update=plan_update)
        markers = self._init_or_reconcile_step_markers(plan=plan)
        self._apply_step_marker_updates(
            plan=plan,
            plan_update=plan_update,
            previous_current=previous_current,
            candidate=inferred_next,
            user_input=user_input,
            observed_tool=observed_tool,
            observed_output=(observed_output if isinstance(observed_output, dict) else {}),
            memory_identity_verified=bool(memory_state.get("identity_verified", False)),
        )
        self._enforce_verification_marker_consistency(
            plan=plan,
            memory_identity_verified=bool(memory_state.get("identity_verified", False)),
            observed_tool=observed_tool,
            observed_output=(observed_output if isinstance(observed_output, dict) else {}),
        )
        self._remove_verify_identity_node_if_verified(
            plan=plan,
            identity_verified=bool(memory_state.get("identity_verified", False)),
        )
        markers = self._init_or_reconcile_step_markers(plan=plan)
        next_current = self._resolve_next_current_node(
            plan=plan,
            previous_current=previous_current,
            candidate=inferred_next,
            markers=markers,
        )
        self._prune_disconnected_nodes(plan=plan, keep_ids={next_current})
        self._enforce_status_consistency(
            plan=plan,
            current_node_id=next_current,
            response_target=response_target,
            markers=markers,
        )
        markers = self._init_or_reconcile_step_markers(plan=plan)

        lowered = user_input.lower()
        signals = plan_signals or {}
        should_revise = (
            bool(signals.get("is_plan_rejection", False))
            or bool(signals.get("needs_discount_specialist", False))
            or bool(signals.get("hardship_signal", False))
            or "hardship" in lowered
            or "cannot pay" in lowered
            or response_target != "customer"
            or bool(plan_update.get("new_nodes"))
            or bool(plan_update.get("remove_node_ids"))
        )
        if should_revise:
            plan["version"] = int(plan.get("version", 1)) + 1
            plan["revision_log"].append(
                {
                    "revision": int(plan.get("version", 1)),
                    "reason": f"context shift: target={response_target}, route={route}, origin={plan_origin}",
                    "at_utc": datetime.now(UTC).isoformat(),
                }
            )

        plan["current_node_id"] = next_current
        plan["previous_node_id"] = previous_current or None
        plan["next_node_ids"] = self._next_nodes_from_edges(nodes=plan.get("edges", []), current_node_id=next_current)
        plan["updated_from"] = plan_origin or "react"
        plan["last_response_target"] = response_target
        status_override = str(plan_update.get("status", "")).strip().lower() if isinstance(plan_update, dict) else ""
        if status_override in {"active", "completed"}:
            plan["status"] = status_override
        else:
            plan["status"] = "completed" if is_conversation_termination(user_input) else "active"
        self._append_timeline_snapshot(
            plan=plan,
            update={
                "origin": plan_origin,
                "route": route,
                "response_target": response_target,
                "previous_node_id": previous_current,
                "current_node_id": next_current,
                "plan_outline": str(proposal.get("plan_outline", "")).strip(),
                "operation": str(plan_update.get("operation", "advance")) if isinstance(plan_update, dict) else "advance",
            },
        )
        return plan

    def _remove_verify_identity_node_if_verified(self, *, plan: dict[str, Any], identity_verified: bool) -> None:
        if not identity_verified:
            return
        # Keep completed steps visible in topology: do not remove verify_identity.
        nodes = [dict(node) for node in plan.get("nodes", []) if isinstance(node, dict)]
        for node in nodes:
            if str(node.get("id", "")).strip() == "verify_identity":
                node["status"] = "done"
        plan["nodes"] = nodes
        markers = plan.get("step_markers")
        if isinstance(markers, dict):
            prior = markers.get("verify_identity") if isinstance(markers.get("verify_identity"), dict) else {}
            markers["verify_identity"] = {
                "state": "done",
                "updated_at": datetime.now(UTC).isoformat(),
                "source": "memory_identity_verified",
                "reason": str(prior.get("reason", "identity_verified_in_memory")),
            }
            plan["step_markers"] = markers

    @staticmethod
    def _create_initial_plan_graph(*, memory_state: dict[str, Any], mode: str) -> dict[str, Any]:
        case_id = str(memory_state.get("active_case_id", "COLL-1001")).strip().upper() or "COLL-1001"
        identity_verified = bool(memory_state.get("identity_verified", False))
        nodes = [
            {"id": "open_and_context", "label": "Initialize case context", "owner": "collection_agent", "status": "in_progress"},
            {"id": "verify_identity", "label": "Verify customer identity", "owner": "customer", "status": ("done" if identity_verified else "pending")},
            {"id": "explain_dues", "label": "Explain dues and policy options", "owner": "customer", "status": "pending"},
            {"id": "collect_payment_intent", "label": "Collect payment intent", "owner": "customer", "status": "pending"},
            {"id": "evaluate_assistance", "label": "Evaluate discount/restructure assistance", "owner": "collection_agent", "status": "pending"},
            {"id": "resolve_outcome", "label": "Finalize payment, promise, or follow-up", "owner": "customer", "status": "pending"},
        ]
        edges = [
            {"from": "open_and_context", "to": "verify_identity", "condition": "case_context_ready"},
            {"from": "verify_identity", "to": "explain_dues", "condition": "identity_verified"},
            {"from": "explain_dues", "to": "collect_payment_intent", "condition": "dues_explained"},
            {"from": "collect_payment_intent", "to": "resolve_outcome", "condition": "pay_now"},
            {"from": "collect_payment_intent", "to": "evaluate_assistance", "condition": "cannot_pay_full"},
            {"from": "evaluate_assistance", "to": "resolve_outcome", "condition": "assistance_ready"},
        ]
        next_node_ids = ["verify_identity"] if not identity_verified else ["explain_dues"]
        return {
            "plan_id": f"plan-{case_id}",
            "version": 1,
            "status": "active",
            "mode": mode,
            "objective": "Move borrower conversation to payment, promise-to-pay, or compliant follow-up.",
            "root_node_id": "open_and_context",
            "current_node_id": "open_and_context",
            "previous_node_id": None,
            "next_node_ids": next_node_ids,
            "nodes": nodes,
            "edges": edges,
            "timeline": [],
            "revision_log": [],
            "updated_from": "initial",
            "last_response_target": "customer",
        }

    def _apply_plan_tree_update(self, *, plan: dict[str, Any], plan_update: dict[str, Any]) -> None:
        if not isinstance(plan_update, dict) or not plan_update:
            return
        nodes = [dict(node) for node in plan.get("nodes", []) if isinstance(node, dict)]
        edges = [dict(edge) for edge in plan.get("edges", []) if isinstance(edge, dict)]
        node_map = {str(node.get("id", "")): node for node in nodes if str(node.get("id", "")).strip()}
        historical_ids = self._historical_node_ids(plan=plan)

        for node_id in [str(x).strip() for x in plan_update.get("remove_node_ids", []) if str(x).strip()]:
            if node_id in historical_ids:
                continue
            node_map.pop(node_id, None)
        if plan_update.get("remove_node_ids"):
            removed_ids = {str(x).strip() for x in plan_update.get("remove_node_ids", []) if str(x).strip()}
            removed_ids = {node_id for node_id in removed_ids if node_id not in historical_ids}
            edges = [
                edge
                for edge in edges
                if str(edge.get("from", "")).strip() not in removed_ids and str(edge.get("to", "")).strip() not in removed_ids
            ]

        for raw in plan_update.get("new_nodes", []):
            if not isinstance(raw, dict):
                continue
            node_id = str(raw.get("id", "")).strip()
            if not node_id:
                continue
            node_map[node_id] = {
                "id": node_id,
                "label": str(raw.get("label", node_id)).strip() or node_id,
                "owner": str(raw.get("owner", "collection_agent")).strip() or "collection_agent",
                "status": str(raw.get("status", "pending")).strip() or "pending",
            }

        edge_set = set()
        for edge in edges:
            src = str(edge.get("from", "")).strip()
            dst = str(edge.get("to", "")).strip()
            cond = str(edge.get("condition", "")).strip()
            if not src or not dst:
                continue
            if src not in node_map or dst not in node_map:
                continue
            edge_set.add((src, dst, cond))
        for raw in plan_update.get("new_edges", []):
            if not isinstance(raw, dict):
                continue
            src = str(raw.get("from", "")).strip()
            dst = str(raw.get("to", "")).strip()
            cond = str(raw.get("condition", "")).strip()
            if not src or not dst:
                continue
            if src not in node_map or dst not in node_map:
                continue
            edge_set.add((src, dst, cond))

        mark_done = {str(x).strip() for x in plan_update.get("mark_done", []) if str(x).strip()}
        mark_skipped = {str(x).strip() for x in plan_update.get("mark_skipped", []) if str(x).strip()}
        mark_blocked = {str(x).strip() for x in plan_update.get("mark_blocked", []) if str(x).strip()}
        for node_id, node in node_map.items():
            if node_id in mark_done:
                node["status"] = "done"
            elif node_id in mark_skipped:
                node["status"] = "skipped"
            elif node_id in mark_blocked:
                node["status"] = "blocked"

        plan["nodes"] = list(node_map.values())
        plan["edges"] = [
            {"from": src, "to": dst, "condition": cond}
            for (src, dst, cond) in sorted(edge_set)
        ]

    def _resolve_next_current_node(
        self,
        *,
        plan: dict[str, Any],
        previous_current: str,
        candidate: str,
        markers: dict[str, Any],
    ) -> str:
        node_ids = {str(node.get("id")) for node in plan.get("nodes", []) if isinstance(node, dict)}
        parent_map = self._parent_map(nodes=plan.get("edges", []))

        def is_actionable(node_id: str) -> bool:
            if node_id not in node_ids:
                return False
            if not self._is_node_unlocked(plan=plan, node_id=node_id, markers=markers):
                return False
            return self._marker_state(markers=markers, node_id=node_id) == "pending"

        def first_actionable_descendant(start_node_id: str) -> str:
            queue: list[str] = [start_node_id]
            visited: set[str] = set()
            while queue:
                nid = queue.pop(0)
                if not nid or nid in visited:
                    continue
                visited.add(nid)
                if is_actionable(nid):
                    return nid
                children = self._next_nodes_from_edges(nodes=plan.get("edges", []), current_node_id=nid)
                for child in children:
                    if child not in visited:
                        queue.append(child)
            return ""

        def nearest_unlocked(node_id: str) -> str:
            cursor = node_id
            visited: set[str] = set()
            while cursor and cursor not in visited:
                visited.add(cursor)
                if cursor in node_ids and self._is_node_unlocked(
                    plan=plan,
                    node_id=cursor,
                    markers=markers,
                ):
                    return cursor
                cursor = parent_map.get(cursor, "")
            root_candidate = str(plan.get("root_node_id", "open_and_context"))
            return root_candidate if root_candidate in node_ids else node_id

        if not previous_current or previous_current not in node_ids:
            root = str(plan.get("root_node_id", "open_and_context"))
            if is_actionable(candidate):
                return candidate
            first = first_actionable_descendant(root)
            if first:
                return first
            if candidate in node_ids and self._is_node_unlocked(plan=plan, node_id=candidate, markers=markers):
                return candidate
            return root if root in node_ids else candidate

        previous_current = nearest_unlocked(previous_current)
        allowed_next = self._next_nodes_from_edges(nodes=plan.get("edges", []), current_node_id=previous_current)
        if not allowed_next:
            if is_actionable(candidate):
                return candidate
            if candidate in node_ids and self._is_node_unlocked(plan=plan, node_id=candidate, markers=markers):
                return candidate
            first = first_actionable_descendant(previous_current)
            if first:
                return first
            return previous_current

        if candidate in allowed_next and is_actionable(candidate):
            return candidate

        # Enforce sequential progression with marker gates.
        for node_id in allowed_next:
            if is_actionable(node_id):
                return node_id
        for node_id in allowed_next:
            first = first_actionable_descendant(node_id)
            if first:
                return first
        return previous_current

    @staticmethod
    def _parent_map(*, nodes: list[Any]) -> dict[str, str]:
        parent_map: dict[str, str] = {}
        for edge in nodes:
            if not isinstance(edge, dict):
                continue
            src = str(edge.get("from", "")).strip()
            dst = str(edge.get("to", "")).strip()
            if not src or not dst:
                continue
            if dst not in parent_map:
                parent_map[dst] = src
        return parent_map

    def _enforce_status_consistency(
        self,
        *,
        plan: dict[str, Any],
        current_node_id: str,
        response_target: str,
        markers: dict[str, Any],
    ) -> None:
        nodes = [node for node in plan.get("nodes", []) if isinstance(node, dict)]
        node_map = {str(node.get("id", "")).strip(): node for node in nodes if str(node.get("id", "")).strip()}
        if current_node_id not in node_map:
            return

        for node_id, node in node_map.items():
            marker_state = self._marker_state(markers=markers, node_id=node_id)
            if node_id == current_node_id:
                if marker_state in {"done", "skipped", "blocked"}:
                    node["status"] = marker_state
                else:
                    node["status"] = "in_progress"
                    node["owner"] = "collection_agent" if response_target == "self" else node.get("owner", "customer")
                continue
            if marker_state in {"done", "skipped", "blocked"}:
                node["status"] = marker_state
            else:
                node["status"] = "pending"

    @staticmethod
    def _marker_state(*, markers: dict[str, Any], node_id: str) -> str:
        raw = markers.get(node_id)
        if isinstance(raw, dict):
            state = str(raw.get("state", "pending")).strip().lower()
        else:
            state = str(raw or "pending").strip().lower()
        if state in {"done", "skipped", "blocked", "pending"}:
            return state
        return "pending"

    def _init_or_reconcile_step_markers(self, *, plan: dict[str, Any]) -> dict[str, Any]:
        existing = plan.get("step_markers") if isinstance(plan.get("step_markers"), dict) else {}
        markers: dict[str, Any] = dict(existing)
        has_existing_markers = bool(existing)
        root_id = str(plan.get("root_node_id", "open_and_context")).strip() or "open_and_context"
        now = datetime.now(UTC).isoformat()

        node_ids: set[str] = set()
        for node in plan.get("nodes", []):
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id", "")).strip()
            if not node_id:
                continue
            node_ids.add(node_id)
            status = str(node.get("status", "pending")).strip().lower()
            previous = markers.get(node_id) if isinstance(markers.get(node_id), dict) else {}
            previous_state = str(previous.get("state", "pending")).strip().lower()
            if previous_state not in {"done", "skipped", "blocked", "pending"}:
                previous_state = "pending"

            if previous_state in {"done", "skipped", "blocked"}:
                state = previous_state
            elif (not has_existing_markers) and status in {"done", "skipped", "blocked"}:
                # Bootstrap marker states from node status for initial sessions.
                state = status
            elif node_id == root_id and not existing:
                state = "pending"
            else:
                state = "pending"

            markers[node_id] = {
                "state": state,
                "updated_at": str(previous.get("updated_at", now)),
                "source": str(previous.get("source", "reconciler")),
                "reason": str(previous.get("reason", "")),
            }

        for marker_id in list(markers.keys()):
            if marker_id not in node_ids:
                markers.pop(marker_id, None)

        plan["step_markers"] = markers
        return markers

    def _apply_step_marker_updates(
        self,
        *,
        plan: dict[str, Any],
        plan_update: dict[str, Any],
        previous_current: str,
        candidate: str,
        user_input: str,
        observed_tool: str,
        observed_output: dict[str, Any],
        memory_identity_verified: bool,
    ) -> None:
        markers = plan.get("step_markers") if isinstance(plan.get("step_markers"), dict) else {}
        now = datetime.now(UTC).isoformat()
        if not isinstance(plan_update, dict):
            plan["step_markers"] = markers
            return

        mark_done = {str(x).strip() for x in plan_update.get("mark_done", []) if str(x).strip()}
        mark_skipped = {str(x).strip() for x in plan_update.get("mark_skipped", []) if str(x).strip()}
        mark_blocked = {str(x).strip() for x in plan_update.get("mark_blocked", []) if str(x).strip()}

        def set_state(node_id: str, state: str, reason: str) -> None:
            prior = markers.get(node_id) if isinstance(markers.get(node_id), dict) else {}
            markers[node_id] = {
                "state": state,
                "updated_at": now,
                "source": "plan_tree_update",
                "reason": reason or str(prior.get("reason", "")),
            }

        for node_id in sorted(mark_done):
            if self._can_accept_done_marker(
                plan=plan,
                node_id=node_id,
                user_input=user_input,
                observed_tool=observed_tool,
                observed_output=observed_output,
                memory_identity_verified=memory_identity_verified,
            ):
                set_state(node_id, "done", "mark_done")
        for node_id in sorted(mark_skipped):
            if self._can_accept_skip_marker(
                plan=plan,
                node_id=node_id,
                previous_current=previous_current,
                user_input=user_input,
                memory_identity_verified=memory_identity_verified,
                markers=markers,
            ):
                set_state(node_id, "skipped", "mark_skipped")
        for node_id in sorted(mark_blocked):
            set_state(node_id, "blocked", "mark_blocked")

        if memory_identity_verified:
            # Deterministic safety: if identity is already verified in memory state,
            # keep the marker aligned even when the model omits mark_done.
            set_state("verify_identity", "done", "memory_identity_verified")

        root_id = str(plan.get("root_node_id", "open_and_context")).strip() or "open_and_context"
        if (
            previous_current == root_id
            and candidate
            and candidate != root_id
            and self._marker_state(markers=markers, node_id=root_id) == "pending"
        ):
            # Root step completes only when planner advances to next actionable step.
            set_state(root_id, "done", "case_context_ready_transition")

        if (
            previous_current
            and candidate
            and candidate != previous_current
            and self._marker_state(markers=markers, node_id=previous_current) == "pending"
        ):
            # Require explicit completion/skip marker before moving from previous step.
            set_state(previous_current, "pending", "awaiting_completion_or_skip_marker")

        plan["step_markers"] = markers

    def _enforce_verification_marker_consistency(
        self,
        *,
        plan: dict[str, Any],
        memory_identity_verified: bool,
        observed_tool: str,
        observed_output: dict[str, Any],
    ) -> None:
        markers = plan.get("step_markers") if isinstance(plan.get("step_markers"), dict) else {}
        verify_raw = markers.get("verify_identity")
        if not isinstance(verify_raw, dict):
            return
        verify_state = str(verify_raw.get("state", "pending")).strip().lower()
        if verify_state != "done":
            return

        if memory_identity_verified:
            return

        verify_raw["state"] = "pending"
        verify_raw["source"] = "reconciler"
        verify_raw["reason"] = "identity_not_verified_in_current_state"
        verify_raw["updated_at"] = datetime.now(UTC).isoformat()
        markers["verify_identity"] = verify_raw
        plan["step_markers"] = markers

    @staticmethod
    def _node_owner(*, plan: dict[str, Any], node_id: str) -> str:
        for node in plan.get("nodes", []):
            if not isinstance(node, dict):
                continue
            if str(node.get("id", "")).strip() != node_id:
                continue
            owner = str(node.get("owner", "collection_agent")).strip().lower()
            return owner or "collection_agent"
        return "collection_agent"

    @staticmethod
    def _node_label_by_id(*, plan: dict[str, Any], node_id: str) -> str:
        for node in plan.get("nodes", []):
            if not isinstance(node, dict):
                continue
            if str(node.get("id", "")).strip() != node_id:
                continue
            return str(node.get("label", node_id)).strip() or node_id
        return node_id

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9]+", str(text).lower())
        stop = {
            "and",
            "or",
            "the",
            "a",
            "an",
            "to",
            "of",
            "for",
            "with",
            "customer",
            "collection",
            "agent",
            "call",
            "context",
            "details",
            "outcome",
            "finalize",
        }
        return {token for token in tokens if token and token not in stop}

    def _tool_matches_node(self, *, node_id: str, node_label: str, observed_tool: str) -> bool:
        tool_tokens = self._tokenize(observed_tool)
        if not tool_tokens:
            return False
        node_tokens = self._tokenize(node_id) | self._tokenize(node_label)
        if not node_tokens:
            return False
        return bool(tool_tokens.intersection(node_tokens))

    def _can_accept_done_marker(
        self,
        *,
        plan: dict[str, Any],
        node_id: str,
        user_input: str,
        observed_tool: str,
        observed_output: dict[str, Any],
        memory_identity_verified: bool,
    ) -> bool:
        owner = self._node_owner(plan=plan, node_id=node_id)
        if node_id == str(plan.get("root_node_id", "open_and_context")).strip():
            return True
        if node_id == "verify_identity":
            # Identity completion is gated by effective verification state,
            # not by single-tool success in isolation.
            return bool(memory_identity_verified)
        if observed_tool:
            tool_status = str(observed_output.get("status", "")).strip().lower()
            if tool_status in {"failed", "locked", "error", "rejected", "denied", "invalid"}:
                return False
            owner = self._node_owner(plan=plan, node_id=node_id)
            if owner in {"customer", "borrower"}:
                label = self._node_label_by_id(plan=plan, node_id=node_id)
                return self._tool_matches_node(node_id=node_id, node_label=label, observed_tool=observed_tool)
            return True
        lowered = user_input.lower()
        if any(token in lowered for token in ["skip", "skipped", "defer", "later"]):
            return True
        if owner in {"customer", "borrower"}:
            # Customer-owned steps must be grounded in an observed tool result.
            return False
        return True

    def _can_accept_skip_marker(
        self,
        *,
        plan: dict[str, Any],
        node_id: str,
        previous_current: str,
        user_input: str,
        memory_identity_verified: bool,
        markers: dict[str, Any],
    ) -> bool:
        root_id = str(plan.get("root_node_id", "open_and_context")).strip() or "open_and_context"
        if node_id == root_id:
            return False
        if node_id == "verify_identity" and not memory_identity_verified:
            return False

        lowered = str(user_input).lower()
        explicit_skip = any(
            token in lowered
            for token in [
                "skip",
                "skipped",
                "defer",
                "later",
                "not now",
                "can't talk",
                "cannot talk",
                "call me later",
            ]
        )
        if not explicit_skip:
            return False

        current = str(previous_current or "").strip()
        if not current:
            return False
        if node_id == current:
            return True

        allowed_next = self._next_nodes_from_edges(nodes=plan.get("edges", []), current_node_id=current)
        if node_id not in allowed_next:
            return False
        return self._is_node_unlocked(plan=plan, node_id=node_id, markers=markers)

    def _is_node_unlocked(self, *, plan: dict[str, Any], node_id: str, markers: dict[str, Any]) -> bool:
        parents: list[str] = []
        for edge in plan.get("edges", []):
            if not isinstance(edge, dict):
                continue
            src = str(edge.get("from", "")).strip()
            dst = str(edge.get("to", "")).strip()
            if dst == node_id and src:
                parents.append(src)
        if not parents:
            return True
        for parent_id in parents:
            parent_state = self._marker_state(markers=markers, node_id=parent_id)
            if parent_state not in {"done", "skipped"}:
                return False
        return True

    def _prune_disconnected_nodes(self, *, plan: dict[str, Any], keep_ids: set[str]) -> None:
        nodes = [node for node in plan.get("nodes", []) if isinstance(node, dict)]
        node_map = {str(node.get("id", "")).strip(): node for node in nodes if str(node.get("id", "")).strip()}
        if not node_map:
            return
        root_id = str(plan.get("root_node_id", "")).strip() or next(iter(node_map.keys()))
        if root_id not in node_map:
            root_id = next(iter(node_map.keys()))
            plan["root_node_id"] = root_id

        forward_adj: dict[str, list[str]] = {node_id: [] for node_id in node_map}
        edges = [edge for edge in plan.get("edges", []) if isinstance(edge, dict)]
        filtered_edges: list[dict[str, Any]] = []
        for edge in edges:
            src = str(edge.get("from", "")).strip()
            dst = str(edge.get("to", "")).strip()
            if src not in node_map or dst not in node_map:
                continue
            filtered_edges.append({"from": src, "to": dst, "condition": str(edge.get("condition", "")).strip()})
            forward_adj[src].append(dst)

        reachable: set[str] = set()
        queue: list[str] = [root_id]
        while queue:
            nid = queue.pop(0)
            if nid in reachable:
                continue
            reachable.add(nid)
            for child in forward_adj.get(nid, []):
                if child not in reachable:
                    queue.append(child)

        historical_ids = self._historical_node_ids(plan=plan)
        keep = set(reachable) | {kid for kid in keep_ids if kid in node_map} | {hid for hid in historical_ids if hid in node_map}
        plan["nodes"] = [node_map[node_id] for node_id in node_map if node_id in keep]
        plan["edges"] = [edge for edge in filtered_edges if edge["from"] in keep and edge["to"] in keep]

    @staticmethod
    def _historical_node_ids(*, plan: dict[str, Any]) -> set[str]:
        markers = plan.get("step_markers") if isinstance(plan.get("step_markers"), dict) else {}
        historical: set[str] = set()
        for node_id, raw in markers.items():
            node_key = str(node_id).strip()
            if not node_key:
                continue
            if isinstance(raw, dict):
                state = str(raw.get("state", "pending")).strip().lower()
            else:
                state = str(raw or "pending").strip().lower()
            if state in {"done", "skipped", "blocked"}:
                historical.add(node_key)
        return historical

    def _infer_current_node_id(
        self,
        *,
        user_input: str,
        observed_tool: str,
        response_target: str,
        route: str,
        proposal: dict[str, Any],
        plan_update: dict[str, Any],
        previous_current: str,
    ) -> str:
        selected_next = str(plan_update.get("selected_next_node_id", "")).strip()
        if selected_next:
            return selected_next
        current_override = str(plan_update.get("current_node_id", "")).strip()
        if current_override:
            return current_override

        lowered = user_input.lower()
        proposal_intent = str(proposal.get("intent", "")).strip().lower() if isinstance(proposal, dict) else ""
        if self.strict_llm_mode:
            if proposal_intent == "conversation_termination" or is_conversation_termination(user_input):
                return "resolve_outcome"
            return str(previous_current or "").strip()
        if proposal_intent == "conversation_termination" or is_conversation_termination(user_input):
            return "resolve_outcome"
        if observed_tool in {"verify_dob", "verify_mobile"} or "verify" in lowered:
            return "verify_identity"
        if observed_tool in {"dues_explain_build", "loan_policy_lookup"} or any(
            token in lowered for token in ["dues", "overdue", "emi", "policy", "amount due"]
        ):
            return "explain_dues"
        if observed_tool in {"payment_link_create", "pay_by_phone_collect", "payment_status_check"} or any(
            token in lowered for token in ["pay now", "payment", "link", "settle"]
        ):
            return "resolve_outcome"
        if any(token in lowered for token in ["cannot pay", "hardship", "discount", "waiver", "restructure", "settlement"]):
            return "evaluate_assistance"
        if response_target == "self":
            return "evaluate_assistance"
        return "collect_payment_intent"

    @staticmethod
    def _next_nodes_from_edges(*, nodes: list[Any], current_node_id: str) -> list[str]:
        next_nodes: list[str] = []
        seen: set[str] = set()
        for edge in nodes:
            if not isinstance(edge, dict):
                continue
            if str(edge.get("from", "")) != current_node_id:
                continue
            to = str(edge.get("to", "")).strip()
            if to and to not in seen:
                next_nodes.append(to)
                seen.add(to)
        return next_nodes

    @staticmethod
    def _node_label(plan: dict[str, Any], node_id: str) -> str:
        for node in plan.get("nodes", []):
            if isinstance(node, dict) and str(node.get("id", "")) == node_id:
                return str(node.get("label", node_id)).strip() or node_id
        return node_id

    @staticmethod
    def _append_timeline_snapshot(*, plan: dict[str, Any], update: dict[str, Any]) -> None:
        timeline = plan.get("timeline")
        entries = list(timeline) if isinstance(timeline, list) else []
        timestamp = datetime.now(UTC).isoformat()
        entries.append(
            {
                "at_utc": timestamp,
                "version": int(plan.get("version", 1)),
                "status": str(plan.get("status", "active")),
                "current_node_id": str(plan.get("current_node_id", "")),
                "next_node_ids": list(plan.get("next_node_ids", [])) if isinstance(plan.get("next_node_ids"), list) else [],
                "update": update,
            }
        )
        plan["timeline"] = entries[-40:]

        # Persist full plan snapshots across turns so the UI can render a
        # conversation-level plan timeline (prev/next over historical plan states).
        snapshot_plan = {
            "plan_id": str(plan.get("plan_id", "")),
            "version": int(plan.get("version", 1)),
            "status": str(plan.get("status", "active")),
            "mode": str(plan.get("mode", "strict_collections")),
            "objective": str(plan.get("objective", "")),
            "root_node_id": str(plan.get("root_node_id", "")),
            "current_node_id": str(plan.get("current_node_id", "")),
            "previous_node_id": plan.get("previous_node_id"),
            "next_node_ids": list(plan.get("next_node_ids", [])) if isinstance(plan.get("next_node_ids"), list) else [],
            "nodes": [dict(node) for node in plan.get("nodes", []) if isinstance(node, dict)],
            "edges": [dict(edge) for edge in plan.get("edges", []) if isinstance(edge, dict)],
            "step_markers": dict(plan.get("step_markers", {})) if isinstance(plan.get("step_markers"), dict) else {},
            "updated_from": str(plan.get("updated_from", "")),
            "last_response_target": str(plan.get("last_response_target", "")),
        }
        snapshots_raw = plan.get("timeline_snapshots")
        snapshots = list(snapshots_raw) if isinstance(snapshots_raw, list) else []
        snapshots.append(
            {
                "at_utc": timestamp,
                "version": int(plan.get("version", 1)),
                "status": str(plan.get("status", "active")),
                "current_node_id": str(plan.get("current_node_id", "")),
                "update": dict(update) if isinstance(update, dict) else {},
                "plan": snapshot_plan,
            }
        )
        plan["timeline_snapshots"] = snapshots[-80:]
