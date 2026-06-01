"""Generate the collection-agent v3 trajectory evaluation dataset."""

from __future__ import annotations

import csv
import json
import random
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EVAL_DIR = ROOT / "eval_dataset"
DATASET_PATH = EVAL_DIR / "collection_agent_golden_dataset_v3.jsonl"
MANIFEST_PATH = EVAL_DIR / "collection_agent_manifest.csv"
EDA_PATH = EVAL_DIR / "collection_agent_eda.csv"


@dataclass(frozen=True, slots=True)
class CanonicalSpec:
    canonical_id: str
    category: str
    flow: str
    difficulty: str
    verification_pattern: str
    customer_style: str
    hardship_reason: str | None = None
    posture_path: tuple[str, ...] = ()
    resolution: str = "conversation_continues"
    followup: bool = False
    counter_offer: bool = False
    dispute_reason: str | None = None


PAYMENT_POSTURES = {
    "pay_now",
    "partial_now",
    "promise_to_pay",
    "cannot_pay",
    "refuses_to_pay",
    "negotiating",
}

DISCOUNT_STAGES = {
    "none",
    "requested",
    "planning",
    "offered",
    "accepted",
    "rejected",
    "counter_offer",
    "closed",
}

NEGOTIATION_STAGES = {
    "none",
    "discovering_hardship",
    "assessing_capacity",
    "evaluating_options",
    "negotiating_plan",
    "confirming_commitment",
    "awaiting_customer_decision",
}

TODAY = date(2026, 6, 1)


def _json_load(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_fixture_pool() -> dict[str, Any]:
    customers = _json_load(DATA_DIR / "customers.json")
    cases = _json_load(DATA_DIR / "cases.json")
    policies = _json_load(DATA_DIR / "policies.json")
    profiles = _json_load(DATA_DIR / "customer_profile.json")
    payments = _json_load(DATA_DIR / "payment_history.json")
    offers = _json_load(DATA_DIR / "offer_history.json")
    programs = _json_load(DATA_DIR / "assistance_programs.json")

    cases_by_customer = {row["customer_id"]: row for row in cases}
    policies_by_loan = {row["loan_id"]: row for row in policies}
    profiles_by_customer = {row["customer_id"]: row for row in profiles}
    payments_by_customer = {row["customer_id"]: row for row in payments}
    offers_by_case = {row["case_id"]: row for row in offers}

    return {
        "customers": customers,
        "cases_by_customer": cases_by_customer,
        "policies_by_loan": policies_by_loan,
        "profiles_by_customer": profiles_by_customer,
        "payments_by_customer": payments_by_customer,
        "offers_by_case": offers_by_case,
        "programs": programs,
    }


def _round_money(value: float) -> int:
    return int(round(value / 50.0) * 50)


def _fmt_money(amount: int | float | None) -> str:
    if amount is None:
        return "INR 0"
    return f"INR {int(round(amount)):,}"


def _phone_for(script_number: int) -> str:
    seed = 8800000000 + script_number
    return str(seed)[-10:]


def _date_str(offset_days: int) -> str:
    return (TODAY + timedelta(days=offset_days)).isoformat()


def build_context(
    *,
    script_number: int,
    spec: CanonicalSpec,
    variant_index: int,
    fixtures: dict[str, Any],
    rnd: random.Random,
) -> dict[str, Any]:
    customer_seed = fixtures["customers"][(script_number + variant_index) % len(fixtures["customers"])]
    case_seed = fixtures["cases_by_customer"][customer_seed["customer_id"]]
    policy_seed = fixtures["policies_by_loan"][case_seed["loan_id"]]
    profile_seed = fixtures["profiles_by_customer"][customer_seed["customer_id"]]
    payment_seed = fixtures["payments_by_customer"][customer_seed["customer_id"]]
    offer_seed = fixtures["offers_by_case"][case_seed["case_id"]]

    script_suffix = f"{script_number:04d}"
    phone = _phone_for(script_number)
    customer = deepcopy(customer_seed)
    customer["customer_id"] = f"CUST-V3-{script_suffix}"
    customer["phone"] = f"+91{phone}"
    customer["email"] = f"{customer['name'].lower().replace(' ', '.')}+{script_suffix}@example.com"
    customer["challenge"] = deepcopy(customer_seed["challenge"])
    customer["challenge"]["phone"] = phone

    case = deepcopy(case_seed)
    case["case_id"] = f"COLL-V3-{script_suffix}"
    case["customer_id"] = customer["customer_id"]
    case["loan_id"] = f"LOAN-V3-{script_suffix}"
    case["portfolio_id"] = f"{case_seed['portfolio_id']}-V3"
    case["dpd"] = max(15, int(round(case_seed["dpd"] * rnd.uniform(0.75, 1.95))))
    case["emi_amount"] = _round_money(case_seed["emi_amount"] * rnd.uniform(0.9, 1.25))
    case["overdue_amount"] = _round_money(case_seed["overdue_amount"] * rnd.uniform(0.85, 1.4))
    case["late_fee"] = _round_money(case_seed["late_fee"] * rnd.uniform(0.7, 1.6))
    case["last_contact_date"] = _date_str(-rnd.randint(2, 20))
    case["contact_attempts"] = max(1, int(round(case_seed["contact_attempts"] * rnd.uniform(0.7, 1.8))))
    case["promise_to_pay_date"] = None
    case["promise_to_pay_amount"] = None

    policy = deepcopy(policy_seed)
    policy["loan_id"] = case["loan_id"]
    if spec.flow in {"partial_payment", "partial_to_full", "settlement_request", "settlement_counter", "multi_turn_negotiation"}:
        policy["allow_partial_payment"] = True
        policy["allow_counter_offer"] = True

    profile = deepcopy(profile_seed)
    profile["customer_id"] = customer["customer_id"]
    profile["risk_score"] = round(min(0.98, max(0.05, profile_seed["risk_score"] * rnd.uniform(0.85, 1.3))), 2)
    profile["previous_broken_promises"] = max(0, int(round(profile_seed["previous_broken_promises"] * rnd.uniform(0.5, 1.5))))

    payment_history = deepcopy(payment_seed)
    payment_history["customer_id"] = customer["customer_id"]
    payment_history["payments"] = [
        {
            "date": _date_str(-30 - (30 * idx)),
            "amount": _round_money(case["emi_amount"] * rnd.uniform(0.9, 1.05)),
        }
        for idx in range(3)
    ]

    offer_history = deepcopy(offer_seed)
    offer_history["case_id"] = case["case_id"]

    assistance_programs = [
        deepcopy(item)
        for item in fixtures["programs"]
        if case["product"] in item.get("eligible_products", [])
    ]

    return {
        "customer": customer,
        "case": case,
        "policy": policy,
        "customer_profile": profile,
        "payment_history": payment_history,
        "offer_history": offer_history,
        "assistance_programs": assistance_programs,
    }


def make_initial_state(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "verification_status": "not_started",
        "verification_missing_fields": ["dob", "mobile"],
        "verification_verified_fields": [],
        "verified_dob": False,
        "verified_mobile": False,
        "identity_verified": False,
        "conversation_mode": "collections",
        "negotiation_stage": "none",
        "customer_payment_posture": "negotiating",
        "customer_payment_posture_history": ["negotiating"],
        "customer_payment_capacity": None,
        "customer_payment_capacity_pct": None,
        "customer_payment_willingness": 0.5,
        "discount_stage": "none",
        "discount_requested": False,
        "discount_offered": False,
        "discount_accepted": False,
        "discount_rejected": False,
        "counter_offer_present": False,
        "response_target": "customer",
        "hardship_reason": None,
        "hardship_context": {
            "hardship_detected": False,
            "hardship_reason": None,
            "confidence": 0.0,
        },
        "plan_signals": {
            "needs_discount_specialist": False,
            "discount_requested": False,
            "customer_payment_posture": "negotiating",
            "customer_payment_capacity": None,
            "customer_payment_capacity_pct": None,
            "discount_stage": "none",
            "hardship_signal": False,
            "hardship_reason": None,
        },
        "needs_discount_specialist": False,
        "promise_to_pay_date": context["case"].get("promise_to_pay_date"),
        "followup_date": None,
    }


def _update_posture_history(state: dict[str, Any], old_posture: str, new_posture: str) -> None:
    if new_posture and new_posture != old_posture:
        history = list(state.get("customer_payment_posture_history") or [])
        if not history or history[-1] != new_posture:
            history.append(new_posture)
        state["customer_payment_posture_history"] = history


def sync_derived_state(state: dict[str, Any]) -> None:
    posture = str(state.get("customer_payment_posture") or "negotiating")
    discount_stage = str(state.get("discount_stage") or "none")
    hardship_context = deepcopy(state.get("hardship_context") or {})
    hardship_reason = hardship_context.get("hardship_reason") or state.get("hardship_reason")
    hardship_signal = bool(hardship_context.get("hardship_detected")) or bool(hardship_reason)
    state["hardship_reason"] = hardship_reason
    needs_discount = bool(
        state.get("discount_requested")
        or posture in {"partial_now", "cannot_pay"}
        or discount_stage in {"requested", "counter_offer"}
        or state.get("counter_offer_present")
    )
    state["needs_discount_specialist"] = needs_discount
    state["plan_signals"] = {
        "needs_discount_specialist": needs_discount,
        "discount_requested": bool(state.get("discount_requested")),
        "customer_payment_posture": posture,
        "customer_payment_capacity": state.get("customer_payment_capacity"),
        "customer_payment_capacity_pct": state.get("customer_payment_capacity_pct"),
        "discount_stage": discount_stage,
        "hardship_signal": hardship_signal,
        "hardship_reason": hardship_reason,
    }


def compute_state_transitions(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    transitions: dict[str, Any] = {}
    for key in sorted(set(before) | set(after)):
        if before.get(key) != after.get(key):
            transitions[key] = {
                "old": deepcopy(before.get(key)),
                "new": deepcopy(after.get(key)),
            }
    return transitions


def make_node_path(
    *,
    state: dict[str, Any],
    extracted: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    response_target: str,
    selected_next_node_id: str,
    include_discount_agent: bool,
    offer_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = [
        {"node": "relevance_intent", "expected_output": {"is_relevant": True, "route": "relevant"}},
        {
            "node": "entity_extract",
            "expected_output": {
                "extracted_entities_turn": deepcopy(extracted or {}),
                "verification_entities": deepcopy((extracted or {}).get("verification_entities", {})),
            },
        },
        {
            "node": "negotiation_classification",
            "expected_output": {
                "conversation_mode": state["conversation_mode"],
                "negotiation_stage": state["negotiation_stage"],
                "customer_payment_posture": state["customer_payment_posture"],
                "customer_payment_willingness": state["customer_payment_willingness"],
                "discount_stage": state["discount_stage"],
                "hardship_context": state["hardship_context"],
                "response_mode": "empathetic" if state["conversation_mode"] == "hardship_negotiation" else "negotiation",
                "active_dialogue_owner": "plan_proposal",
            },
        },
        {
            "node": "pre_plan_intent",
            "expected_output": {
                "intent": "collections",
                "verification_required": not state["identity_verified"],
            },
        },
    ]

    verify_tools = [call for call in tool_calls if call["tool"] in {"verify_dob", "verify_mobile"}]
    action_tools = [call for call in tool_calls if call["tool"] not in {"verify_dob", "verify_mobile"}]
    if verify_tools:
        nodes.extend(
            [
                {
                    "node": "execution_path_intent",
                    "expected_output": {"requires_tools": True, "tool_domain": "verification"},
                },
                {
                    "node": "verification_react",
                    "expected_output": {
                        "queued_tools": [call["tool"] for call in verify_tools],
                        "identity_verified": state["identity_verified"],
                        "verification_missing_fields": state["verification_missing_fields"],
                    },
                },
                {
                    "node": "tool_execution",
                    "expected_output": {"observed_tools": [call["tool"] for call in verify_tools]},
                },
                {
                    "node": "post_verification_intent",
                    "expected_output": {"identity_verified": state["identity_verified"], "route": "planning"},
                },
            ]
        )
    elif action_tools:
        nodes.extend(
            [
                {
                    "node": "execution_path_intent",
                    "expected_output": {"requires_tools": True, "tool_domain": "action"},
                },
                {
                    "node": "react",
                    "expected_output": {"queued_tools": [call["tool"] for call in action_tools]},
                },
                {
                    "node": "tool_execution",
                    "expected_output": {"observed_tools": [call["tool"] for call in action_tools]},
                },
            ]
        )

    nodes.extend(
        [
            {
                "node": "plan_proposal_state",
                "expected_output": {
                    "plan_mode": "hardship_negotiation"
                    if state["conversation_mode"] == "hardship_negotiation"
                    else "strict_collections",
                    "plan_signals": deepcopy(state["plan_signals"]),
                },
            },
            {
                "node": "plan_proposal_graph",
                "expected_output": {
                    "current_node_id": selected_next_node_id,
                    "selected_next_node_id": selected_next_node_id,
                },
            },
            {
                "node": "plan_proposal_directive",
                "expected_output": {
                    "response_target": response_target,
                    "conversation_objective": selected_next_node_id,
                    "handoff_payload": deepcopy(offer_payload) if response_target == "discount_planning_agent" else None,
                },
            },
            {"node": "reflect", "expected_output": {"is_complete": True, "failure_type": "none"}},
        ]
    )

    if include_discount_agent:
        nodes.append(
            {
                "node": "discount_planning_agent",
                "expected_output": deepcopy(offer_payload) if offer_payload else {"recommended_offer": None},
            }
        )

    nodes.append({"node": "relevant_response", "expected_output": {"response_target": "customer"}})
    return nodes


def make_payment_link_tool(case: dict[str, Any], amount: int, script_id: str) -> dict[str, Any]:
    return {
        "tool": "payment_link_create",
        "input": {"case_id": case["case_id"], "amount": amount},
        "output": {
            "status": "created",
            "payment_link": f"https://pay.example.com/{script_id.lower()}",
            "amount": amount,
        },
    }


def make_promise_tool(case: dict[str, Any], amount: int, promise_date: str) -> dict[str, Any]:
    return {
        "tool": "promise_capture",
        "input": {"case_id": case["case_id"], "amount": amount, "promise_date": promise_date},
        "output": {
            "status": "captured",
            "promise_amount": amount,
            "promise_date": promise_date,
        },
    }


def make_escalation_tool(case: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "tool": "human_escalation",
        "input": {"case_id": case["case_id"], "reason": reason},
        "output": {"status": "queued", "queue": "collections_supervisor", "reason": reason},
    }


def make_verification_tools(
    *,
    customer: dict[str, Any],
    include_dob: bool,
    include_mobile: bool,
    dob_value: str | None = None,
    mobile_value: str | None = None,
) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    if include_dob:
        provided = dob_value or customer["challenge"]["dob"]
        status = "verified" if provided == customer["challenge"]["dob"] else "failed"
        tools.append(
            {
                "tool": "verify_dob",
                "input": {"dob": provided},
                "output": {"status": status, "field": "dob", "verified_dob": status == "verified"},
            }
        )
    if include_mobile:
        provided = mobile_value or customer["challenge"]["phone"]
        status = "verified" if provided == customer["challenge"]["phone"] else "failed"
        tools.append(
            {
                "tool": "verify_mobile",
                "input": {"mobile": provided},
                "output": {"status": status, "field": "mobile", "verified_mobile": status == "verified"},
            }
        )
    return tools


class TrajectoryBuilder:
    def __init__(self, *, script_id: str, context: dict[str, Any], state: dict[str, Any], rnd: random.Random) -> None:
        self.script_id = script_id
        self.context = context
        self.state = state
        self.turns: list[dict[str, Any]] = []
        self.rnd = rnd

    def add_turn(
        self,
        *,
        customer_text: str,
        agent_text: str,
        updates: dict[str, Any] | None = None,
        extracted: dict[str, Any] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        response_target: str = "customer",
        selected_next_node_id: str = "continue_collections",
        include_discount_agent: bool = False,
        offer_payload: dict[str, Any] | None = None,
    ) -> None:
        before = deepcopy(self.state)
        old_posture = str(self.state.get("customer_payment_posture") or "")
        for key, value in (updates or {}).items():
            self.state[key] = deepcopy(value)
        new_posture = str(self.state.get("customer_payment_posture") or "")
        _update_posture_history(self.state, old_posture, new_posture)
        self.state["response_target"] = response_target
        sync_derived_state(self.state)
        transitions = compute_state_transitions(before, self.state)
        self.turns.append(
            {
                "turn_id": len(self.turns) + 1,
                "customer": customer_text,
                "agent": agent_text,
                "nodes_traversed": make_node_path(
                    state=self.state,
                    extracted=extracted or {},
                    tool_calls=tool_calls or [],
                    response_target=response_target,
                    selected_next_node_id=selected_next_node_id,
                    include_discount_agent=include_discount_agent,
                    offer_payload=offer_payload,
                ),
                "tool_calls": deepcopy(tool_calls or []),
                "state_transitions": transitions,
            }
        )


def _agent_intro_for_verification() -> str:
    return "Thanks for confirming. For security, please share your date of birth and registered mobile number."


def _agent_due_prompt(context: dict[str, Any]) -> str:
    total_due = context["case"]["overdue_amount"] + context["case"]["late_fee"]
    return (
        f"Thank you for verifying. Your overdue amount is {_fmt_money(total_due)}. "
        "How would you like to address this today?"
    )


def _make_offer_payload(
    *,
    context: dict[str, Any],
    state: dict[str, Any],
    amount: int | None = None,
    amount_pct: int | None = None,
    hardship_reason: str | None = None,
    offer_type: str = "arrangement",
) -> dict[str, Any]:
    total_due = context["case"]["overdue_amount"] + context["case"]["late_fee"]
    recommended_amount = amount or max(_round_money(total_due * 0.6), context["case"]["emi_amount"])
    discount_pct = 0
    if offer_type == "settlement":
        discount_pct = min(context["policy"]["max_waiver_pct"], 18)
        recommended_amount = _round_money(total_due * (1 - (discount_pct / 100.0)))
    installments = max(2, min(6, int(round(total_due / max(recommended_amount, 1)))))
    return {
        "recommended_offer": {
            "offer_type": offer_type,
            "recommended_amount": recommended_amount,
            "discount_pct": discount_pct,
            "installments": installments,
        },
        "offer_variants": [
            {"amount": recommended_amount, "months": installments},
            {"amount": _round_money(recommended_amount * 1.1), "months": max(2, installments - 1)},
        ],
        "input_context": {
            "case_id": context["case"]["case_id"],
            "customer_payment_posture": state["customer_payment_posture"],
            "customer_payment_capacity": amount,
            "customer_payment_capacity_pct": amount_pct,
            "discount_stage": state["discount_stage"],
            "hardship_reason": hardship_reason,
        },
        "rationale": "Recommended based on payment capacity, hardship, and policy bounds.",
        "compliance_flags": [],
        "confidence": 0.82,
        "next_action_hint": "present_offer_to_customer",
    }


def _verification_success_sequence(
    *,
    builder: TrajectoryBuilder,
    pattern: str,
) -> None:
    customer = builder.context["customer"]
    if pattern == "split":
        builder.add_turn(
            customer_text=f"Yes, this is {customer['name'].split()[0]}.",
            agent_text=_agent_intro_for_verification(),
            updates={"verification_status": "in_progress"},
            selected_next_node_id="verify_identity",
        )
        builder.add_turn(
            customer_text=f"My date of birth is {customer['challenge']['dob']}.",
            agent_text="I have your date of birth. Please confirm the registered mobile number as well.",
            extracted={"verification_entities": {"dob": customer["challenge"]["dob"]}},
            tool_calls=make_verification_tools(customer=customer, include_dob=True, include_mobile=False),
            updates={
                "verification_status": "in_progress",
                "verified_dob": True,
                "verification_verified_fields": ["dob"],
                "verification_missing_fields": ["mobile"],
            },
            selected_next_node_id="verify_identity",
        )
        builder.add_turn(
            customer_text=f"The registered mobile is {customer['challenge']['phone']}.",
            agent_text=_agent_due_prompt(builder.context),
            extracted={"verification_entities": {"mobile": customer["challenge"]["phone"]}},
            tool_calls=make_verification_tools(customer=customer, include_dob=False, include_mobile=True),
            updates={
                "verification_status": "verified",
                "verified_mobile": True,
                "verification_verified_fields": ["dob", "mobile"],
                "verification_missing_fields": [],
                "identity_verified": True,
            },
            selected_next_node_id="explain_dues",
        )
        return

    if pattern == "failed_once":
        wrong_dob = customer["challenge"]["dob"][:-2] + "11"
        builder.add_turn(
            customer_text=f"Yes, this is {customer['name'].split()[0]}.",
            agent_text=_agent_intro_for_verification(),
            updates={"verification_status": "in_progress"},
            selected_next_node_id="verify_identity",
        )
        builder.add_turn(
            customer_text=f"My date of birth is {wrong_dob} and my phone is {customer['challenge']['phone']}.",
            agent_text="I could verify the mobile number, but the date of birth did not match. Please confirm the correct date of birth.",
            extracted={
                "verification_entities": {
                    "dob": wrong_dob,
                    "mobile": customer["challenge"]["phone"],
                }
            },
            tool_calls=make_verification_tools(
                customer=customer,
                include_dob=True,
                include_mobile=True,
                dob_value=wrong_dob,
            ),
            updates={
                "verification_status": "in_progress",
                "verified_mobile": True,
                "verified_dob": False,
                "verification_verified_fields": ["mobile"],
                "verification_missing_fields": ["dob"],
            },
            selected_next_node_id="verify_identity",
        )
        builder.add_turn(
            customer_text=f"Sorry, the correct date of birth is {customer['challenge']['dob']}.",
            agent_text=_agent_due_prompt(builder.context),
            extracted={"verification_entities": {"dob": customer["challenge"]["dob"]}},
            tool_calls=make_verification_tools(customer=customer, include_dob=True, include_mobile=False),
            updates={
                "verification_status": "verified",
                "verified_dob": True,
                "verification_verified_fields": ["mobile", "dob"],
                "verification_missing_fields": [],
                "identity_verified": True,
            },
            selected_next_node_id="explain_dues",
        )
        return

    builder.add_turn(
        customer_text=f"Yes, this is {customer['name'].split()[0]}.",
        agent_text=_agent_intro_for_verification(),
        updates={"verification_status": "in_progress"},
        selected_next_node_id="verify_identity",
    )
    builder.add_turn(
        customer_text=(
            f"My phone number is {customer['challenge']['phone']} and my date of birth is "
            f"{customer['challenge']['dob']}."
        ),
        agent_text=_agent_due_prompt(builder.context),
        extracted={
            "verification_entities": {
                "dob": customer["challenge"]["dob"],
                "mobile": customer["challenge"]["phone"],
            }
        },
        tool_calls=make_verification_tools(customer=customer, include_dob=True, include_mobile=True),
        updates={
            "verification_status": "verified",
            "verified_dob": True,
            "verified_mobile": True,
            "verification_verified_fields": ["dob", "mobile"],
            "verification_missing_fields": [],
            "identity_verified": True,
        },
        selected_next_node_id="explain_dues",
    )


def _generate_pay_now(builder: TrajectoryBuilder, reluctant: bool) -> str:
    context = builder.context
    total_due = context["case"]["overdue_amount"] + context["case"]["late_fee"]
    amount = total_due
    if reluctant:
        builder.add_turn(
            customer_text="That is a sizeable amount. Can you tell me whether there are any extra charges if I delay this by a few days?",
            agent_text=(
                f"The current dues are {_fmt_money(total_due)} including late fees. "
                "Paying today will help avoid additional collection pressure."
            ),
            updates={
                "customer_payment_posture": "negotiating",
                "customer_payment_willingness": 0.62,
            },
            selected_next_node_id="collect_payment_intent",
        )
    else:
        builder.add_turn(
            customer_text="Please tell me the exact amount due today.",
            agent_text=f"The amount due today is {_fmt_money(total_due)}. If you can clear it now, I can send a payment link immediately.",
            updates={
                "customer_payment_posture": "negotiating",
                "customer_payment_willingness": 0.8,
            },
            selected_next_node_id="collect_payment_intent",
        )
    builder.add_turn(
        customer_text="I can pay the full amount today.",
        agent_text="That works. If you are ready, I can send the payment link now.",
        updates={"customer_payment_posture": "pay_now", "customer_payment_willingness": 0.97},
        selected_next_node_id="collect_payment",
    )
    builder.add_turn(
        customer_text="Yes, please send me the payment link.",
        agent_text="I have generated the payment link. Please complete the payment and let me know once done.",
        tool_calls=[make_payment_link_tool(context["case"], amount, builder.script_id)],
        updates={"customer_payment_posture": "pay_now", "customer_payment_willingness": 1.0},
        selected_next_node_id="collect_payment",
    )
    builder.add_turn(
        customer_text="I will make the payment in the next few minutes.",
        agent_text="Thank you. I will note that you are paying today. Please reply once it is completed.",
        updates={"customer_payment_posture": "pay_now", "customer_payment_willingness": 1.0},
        selected_next_node_id="confirm_commitment",
    )
    builder.add_turn(
        customer_text="The payment is done from my side.",
        agent_text="Thank you. We will update the account once the payment is posted. Is there anything else you want me to note?",
        updates={"customer_payment_posture": "pay_now"},
        selected_next_node_id="close_conversation",
    )
    builder.add_turn(
        customer_text="No, that is all.",
        agent_text="Understood. Thank you for your time today.",
        updates={"discount_stage": "closed"},
        selected_next_node_id="close_conversation",
    )
    return "payment_link_sent"


def _generate_promise(builder: TrajectoryBuilder, with_followup: bool) -> str:
    context = builder.context
    promise_amount = context["case"]["overdue_amount"]
    promise_date = _date_str(3 if with_followup else 2)
    builder.add_turn(
        customer_text=f"I cannot pay today, but I can pay {_fmt_money(promise_amount)} on {promise_date}.",
        agent_text="I can note that arrangement. Please confirm the date and amount you can commit to.",
        updates={
            "customer_payment_posture": "promise_to_pay",
            "customer_payment_willingness": 0.74,
            "negotiation_stage": "confirming_commitment",
            "promise_to_pay_date": promise_date,
        },
        selected_next_node_id="capture_promise",
    )
    builder.add_turn(
        customer_text=f"Yes, I will pay {_fmt_money(promise_amount)} on {promise_date}.",
        agent_text="Thank you. I have captured your commitment.",
        tool_calls=[make_promise_tool(context["case"], promise_amount, promise_date)],
        updates={
            "customer_payment_posture": "promise_to_pay",
            "customer_payment_willingness": 0.78,
        },
        selected_next_node_id="capture_promise",
    )
    builder.add_turn(
        customer_text="Please make a note that I may need a reminder on the same day.",
        agent_text=(
            f"I have noted your request. We will use {promise_date} as the expected payment date."
            if with_followup
            else "Noted. Please ensure the payment is completed on the promised date."
        ),
        updates={"followup_date": promise_date if with_followup else None},
        selected_next_node_id="confirm_commitment",
    )
    builder.add_turn(
        customer_text="That is fine with me.",
        agent_text="Thank you. We will look for the promised payment.",
        updates={},
        selected_next_node_id="close_conversation",
    )
    builder.add_turn(
        customer_text="Okay.",
        agent_text="Thanks for your cooperation today.",
        updates={"discount_stage": "closed"},
        selected_next_node_id="close_conversation",
    )
    builder.add_turn(
        customer_text="Goodbye.",
        agent_text="Goodbye.",
        updates={},
        selected_next_node_id="close_conversation",
    )
    return "promise_captured_followup" if with_followup else "promise_captured"


def _generate_partial(
    builder: TrajectoryBuilder,
    *,
    to_full: bool,
    settlement: bool,
    counter_offer: bool,
) -> str:
    context = builder.context
    total_due = context["case"]["overdue_amount"] + context["case"]["late_fee"]
    partial_amount = _round_money(total_due * builder.rnd.uniform(0.28, 0.55))
    offer_type = "settlement" if settlement else "arrangement"
    hardship_reason = builder.state["hardship_reason"]
    builder.add_turn(
        customer_text=f"I can pay {_fmt_money(partial_amount)} today, but not the full amount.",
        agent_text="Let me see what structured option fits that amount.",
        extracted={"customer_payment_capacity": partial_amount, "customer_payment_capacity_pct": None},
        updates={
            "customer_payment_posture": "partial_now",
            "customer_payment_capacity": partial_amount,
            "customer_payment_capacity_pct": None,
            "customer_payment_willingness": 0.67,
            "negotiation_stage": "evaluating_options",
            "discount_requested": True,
            "discount_stage": "requested",
        },
        response_target="discount_planning_agent",
        selected_next_node_id="present_arrangement_options",
        include_discount_agent=True,
        offer_payload=_make_offer_payload(
            context=context,
            state=builder.state,
            amount=partial_amount,
            hardship_reason=hardship_reason,
            offer_type=offer_type,
        ),
    )
    builder.add_turn(
        customer_text="What arrangement can you offer based on that?",
        agent_text=(
            f"We can look at a structured plan anchored around {_fmt_money(partial_amount)} today and staged follow-up payments."
            if not settlement
            else f"We can explore a settlement-style closure anchored around {_fmt_money(partial_amount)}."
        ),
        updates={
            "discount_stage": "offered",
            "discount_offered": True,
            "conversation_mode": "hardship_negotiation" if hardship_reason else "collections",
            "negotiation_stage": "negotiating_plan",
        },
        selected_next_node_id="present_arrangement_options",
    )
    if counter_offer:
        revised_amount = _round_money(partial_amount * 1.18)
        builder.add_turn(
            customer_text=f"I cannot go above {_fmt_money(partial_amount)}, and I want the account closed if possible.",
            agent_text="I understand. I will note that as your counter-offer and review whether there is any flexibility.",
            updates={
                "counter_offer_present": True,
                "discount_stage": "counter_offer",
                "customer_payment_willingness": 0.61,
            },
            response_target="discount_planning_agent",
            selected_next_node_id="negotiate_installment",
            include_discount_agent=True,
            offer_payload=_make_offer_payload(
                context=context,
                state=builder.state,
                amount=revised_amount,
                hardship_reason=hardship_reason,
                offer_type=offer_type,
            ),
        )
        builder.add_turn(
            customer_text="Can you tell me the best final option you have?",
            agent_text=f"The revised option requires at least {_fmt_money(revised_amount)} to proceed.",
            updates={
                "discount_stage": "offered",
                "discount_offered": True,
                "negotiation_stage": "awaiting_customer_decision",
            },
            selected_next_node_id="present_arrangement_options",
        )
    if to_full:
        full_amount = total_due
        builder.add_turn(
            customer_text="If that is the case, I will clear the full amount today instead.",
            agent_text="Understood. I can move this to a full payment flow right away.",
            updates={
                "customer_payment_posture": "pay_now",
                "customer_payment_capacity": full_amount,
                "customer_payment_willingness": 0.93,
                "discount_stage": "closed",
                "discount_requested": False,
            },
            selected_next_node_id="collect_payment",
        )
        builder.add_turn(
            customer_text="Please share the link for the full amount.",
            agent_text="I have generated the payment link for the full balance.",
            tool_calls=[make_payment_link_tool(context["case"], full_amount, builder.script_id)],
            updates={"customer_payment_posture": "pay_now", "customer_payment_willingness": 1.0},
            selected_next_node_id="collect_payment",
        )
        builder.add_turn(
            customer_text="I will make that payment now.",
            agent_text="Thank you. Please confirm once it is completed.",
            updates={},
            selected_next_node_id="confirm_commitment",
        )
        builder.add_turn(
            customer_text="Done.",
            agent_text="Thank you for resolving this today.",
            updates={"discount_stage": "closed"},
            selected_next_node_id="close_conversation",
        )
        return "partial_to_full_payment"

    plan_date = _date_str(4)
    builder.add_turn(
        customer_text=f"I can proceed if you note the balance for later this week, by {plan_date}.",
        agent_text="That works. I will record this as a structured arrangement.",
        tool_calls=[make_promise_tool(context["case"], partial_amount, plan_date)],
        updates={
            "customer_payment_posture": "partial_now",
            "promise_to_pay_date": plan_date,
            "customer_payment_willingness": 0.72,
            "discount_stage": "accepted" if settlement else "offered",
        },
        selected_next_node_id="confirm_commitment",
    )
    builder.add_turn(
        customer_text="Okay, please note that.",
        agent_text="Noted. We will expect the agreed amount and then follow up on the remaining balance.",
        updates={
            "discount_accepted": settlement,
            "discount_stage": "closed" if settlement else "offered",
        },
        selected_next_node_id="close_conversation",
    )
    builder.add_turn(
        customer_text="Please send a reminder before that date.",
        agent_text="I have noted the reminder request against the arrangement.",
        updates={"followup_date": plan_date},
        selected_next_node_id="close_conversation",
    )
    builder.add_turn(
        customer_text="Thank you.",
        agent_text="You are welcome. We will follow the agreed plan from here.",
        updates={"discount_stage": "closed"},
        selected_next_node_id="close_conversation",
    )
    return "settlement_accepted" if settlement else "partial_payment_arrangement"


def _generate_hardship(
    builder: TrajectoryBuilder,
    *,
    hardship_reason: str,
    accepted: bool | None,
    rejected: bool,
) -> str:
    context = builder.context
    hardship_text = {
        "job_loss": "I lost my job recently and I cannot manage the full EMI right now.",
        "medical_emergency": "I had a medical emergency at home and my finances are under pressure.",
        "family_emergency": "There is a family crisis right now, so I cannot pay the full amount.",
    }[hardship_reason]
    builder.add_turn(
        customer_text=hardship_text,
        agent_text="I am sorry to hear that. Let us look at what may be manageable given the situation.",
        updates={
            "conversation_mode": "hardship_negotiation",
            "negotiation_stage": "discovering_hardship",
            "customer_payment_posture": "cannot_pay",
            "customer_payment_willingness": 0.28,
            "hardship_reason": hardship_reason,
            "hardship_context": {
                "hardship_detected": True,
                "hardship_reason": hardship_reason,
                "confidence": 0.95,
            },
        },
        selected_next_node_id="assess_affordability",
    )
    builder.add_turn(
        customer_text="At best I can manage a much smaller amount this month.",
        agent_text="Thank you. I will use that to explore whether a hardship arrangement can be discussed.",
        updates={
            "negotiation_stage": "assessing_capacity",
            "discount_requested": True,
            "discount_stage": "requested",
        },
        response_target="discount_planning_agent",
        selected_next_node_id="present_arrangement_options",
        include_discount_agent=True,
        offer_payload=_make_offer_payload(
            context=context,
            state=builder.state,
            amount=_round_money((context["case"]["overdue_amount"] + context["case"]["late_fee"]) * 0.35),
            hardship_reason=hardship_reason,
            offer_type="settlement" if accepted is not None else "arrangement",
        ),
    )
    builder.add_turn(
        customer_text="Please tell me what support may be possible.",
        agent_text="Based on the review, I can discuss a hardship-aligned option with you.",
        updates={"discount_stage": "offered", "discount_offered": True, "negotiation_stage": "evaluating_options"},
        selected_next_node_id="present_arrangement_options",
    )
    if accepted is True:
        plan_date = _date_str(5)
        offer_amount = _round_money((context["case"]["overdue_amount"] + context["case"]["late_fee"]) * 0.55)
        builder.add_turn(
            customer_text=f"If that option is available, I can commit to {_fmt_money(offer_amount)} by {plan_date}.",
            agent_text="Thank you. I can document that commitment under the discussed hardship option.",
            tool_calls=[make_promise_tool(context["case"], offer_amount, plan_date)],
            updates={
                "customer_payment_posture": "partial_now",
                "customer_payment_capacity": offer_amount,
                "customer_payment_willingness": 0.66,
                "discount_stage": "accepted",
                "discount_accepted": True,
                "promise_to_pay_date": plan_date,
                "negotiation_stage": "confirming_commitment",
            },
            selected_next_node_id="confirm_commitment",
        )
        builder.add_turn(
            customer_text="Yes, please note that arrangement.",
            agent_text="It has been noted. We will watch for that payment on the agreed date.",
            updates={"discount_stage": "closed"},
            selected_next_node_id="close_conversation",
        )
        builder.add_turn(
            customer_text="Thank you for working with me.",
            agent_text="You are welcome. Please keep to the committed date if at all possible.",
            updates={},
            selected_next_node_id="close_conversation",
        )
        return "hardship_discount_accepted"
    if rejected:
        builder.add_turn(
            customer_text="That still does not work for me. I cannot agree to that option.",
            agent_text="Understood. I will note that the proposed hardship option was declined.",
            updates={
                "discount_stage": "rejected",
                "discount_rejected": True,
                "customer_payment_willingness": 0.22,
            },
            selected_next_node_id="awaiting_customer_decision",
        )
        builder.add_turn(
            customer_text="Then I need more time or someone senior to review this.",
            agent_text="I can route this for a human review so the team can decide the next step.",
            tool_calls=[make_escalation_tool(context["case"], "hardship_offer_rejected")],
            updates={"followup_date": _date_str(2)},
            selected_next_node_id="human_escalation",
        )
        builder.add_turn(
            customer_text="Okay, please do that.",
            agent_text="I have logged the escalation request.",
            updates={"discount_stage": "closed"},
            selected_next_node_id="close_conversation",
        )
        return "hardship_discount_rejected"

    builder.add_turn(
        customer_text="Please help me with the smallest workable payment plan.",
        agent_text="We can continue reviewing practical installment options based on your situation.",
        updates={"customer_payment_posture": "negotiating", "customer_payment_willingness": 0.44, "negotiation_stage": "negotiating_plan"},
        selected_next_node_id="present_arrangement_options",
    )
    builder.add_turn(
        customer_text="I may be able to start with a small amount next week.",
        agent_text="That is helpful. We can keep working from that starting point.",
        updates={"customer_payment_posture": "partial_now", "customer_payment_capacity": _round_money(context["case"]["emi_amount"] * 0.4)},
        selected_next_node_id="confirm_commitment",
    )
    builder.add_turn(
        customer_text="Please note that for now.",
        agent_text="Noted. We will continue from this hardship discussion on the next contact.",
        updates={"followup_date": _date_str(3)},
        selected_next_node_id="close_conversation",
    )
    return "hardship_arrangement_in_progress"


def _generate_verification_failure(builder: TrajectoryBuilder) -> str:
    customer = builder.context["customer"]
    wrong_mobile = "9" + customer["challenge"]["phone"][1:-1] + "0"
    wrong_dob = customer["challenge"]["dob"][:-2] + "01"
    builder.add_turn(
        customer_text=f"Yes, I am {customer['name'].split()[0]}.",
        agent_text=_agent_intro_for_verification(),
        updates={"verification_status": "in_progress"},
        selected_next_node_id="verify_identity",
    )
    builder.add_turn(
        customer_text=f"My date of birth is {wrong_dob}.",
        agent_text="That date of birth does not match. Please try again, and also confirm the registered mobile number.",
        extracted={"verification_entities": {"dob": wrong_dob}},
        tool_calls=make_verification_tools(customer=customer, include_dob=True, include_mobile=False, dob_value=wrong_dob),
        updates={"verification_status": "failed", "verification_missing_fields": ["dob", "mobile"]},
        selected_next_node_id="verify_identity",
    )
    builder.add_turn(
        customer_text=f"The mobile is {wrong_mobile}.",
        agent_text="That mobile number also did not match. I still cannot verify the account.",
        extracted={"verification_entities": {"mobile": wrong_mobile}},
        tool_calls=make_verification_tools(customer=customer, include_dob=False, include_mobile=True, mobile_value=wrong_mobile),
        updates={"verification_status": "failed", "verification_missing_fields": ["dob", "mobile"]},
        selected_next_node_id="verify_identity",
    )
    builder.add_turn(
        customer_text="I do not remember the exact details right now.",
        agent_text="Without verification I cannot discuss the account, but I can arrange a callback or human follow-up.",
        updates={"customer_payment_posture": "negotiating", "customer_payment_willingness": 0.3},
        selected_next_node_id="verify_identity",
    )
    builder.add_turn(
        customer_text="Please have someone call me later.",
        agent_text="I will escalate this for assisted follow-up.",
        tool_calls=[make_escalation_tool(builder.context["case"], "verification_failed")],
        updates={"followup_date": _date_str(1)},
        selected_next_node_id="human_escalation",
    )
    builder.add_turn(
        customer_text="Okay.",
        agent_text="Thank you. A member of the team will follow up after reviewing the case.",
        updates={"discount_stage": "closed"},
        selected_next_node_id="close_conversation",
    )
    builder.add_turn(
        customer_text="All right.",
        agent_text="Take care.",
        updates={},
        selected_next_node_id="close_conversation",
    )
    builder.add_turn(
        customer_text="Goodbye.",
        agent_text="Goodbye.",
        updates={},
        selected_next_node_id="close_conversation",
    )
    return "verification_failed"


def _generate_wrong_contact(builder: TrajectoryBuilder) -> str:
    builder.add_turn(
        customer_text="You have the wrong number. There is no Aditi here.",
        agent_text="Thank you for letting me know. I will not discuss the account. Can you confirm whether you know the customer at all?",
        updates={"verification_status": "not_started", "customer_payment_willingness": 0.0},
        selected_next_node_id="verify_identity",
    )
    builder.add_turn(
        customer_text="No, I do not know that person.",
        agent_text="Understood. I will mark this as a wrong contact so the team can update the record.",
        updates={"customer_payment_posture": "refuses_to_pay", "customer_payment_willingness": 0.0},
        selected_next_node_id="human_escalation",
    )
    builder.add_turn(
        customer_text="Please stop calling this number.",
        agent_text="I understand. I am recording the number as a wrong contact.",
        tool_calls=[make_escalation_tool(builder.context["case"], "wrong_contact")],
        updates={"followup_date": None},
        selected_next_node_id="human_escalation",
    )
    builder.add_turn(
        customer_text="Thank you.",
        agent_text="You are welcome. No account details have been disclosed.",
        updates={},
        selected_next_node_id="close_conversation",
    )
    builder.add_turn(
        customer_text="Okay.",
        agent_text="We will update the records accordingly.",
        updates={},
        selected_next_node_id="close_conversation",
    )
    builder.add_turn(
        customer_text="Fine.",
        agent_text="Thank you for your time.",
        updates={},
        selected_next_node_id="close_conversation",
    )
    builder.add_turn(
        customer_text="Bye.",
        agent_text="Goodbye.",
        updates={},
        selected_next_node_id="close_conversation",
    )
    builder.add_turn(
        customer_text="Understood.",
        agent_text="The wrong-contact closure is complete.",
        updates={"discount_stage": "closed"},
        selected_next_node_id="close_conversation",
    )
    return "wrong_contact_marked"


def _generate_suspicious(builder: TrajectoryBuilder) -> str:
    builder.add_turn(
        customer_text="How do I know this is not a scam call?",
        agent_text="I understand the concern. I will not discuss the account until you verify, and I can arrange a supervisor callback if you prefer.",
        updates={"verification_status": "in_progress", "customer_payment_willingness": 0.12},
        selected_next_node_id="verify_identity",
    )
    builder.add_turn(
        customer_text="I am not sharing personal details unless I speak to an official number.",
        agent_text="That is fine. I can route this to a verified callback process.",
        updates={"customer_payment_posture": "negotiating"},
        selected_next_node_id="human_escalation",
    )
    builder.add_turn(
        customer_text="Please do that.",
        agent_text="I have submitted the callback request.",
        tool_calls=[make_escalation_tool(builder.context["case"], "suspicious_customer")],
        updates={"followup_date": _date_str(1)},
        selected_next_node_id="human_escalation",
    )
    builder.add_turn(
        customer_text="When should I expect the call?",
        agent_text="The team should try within one business day.",
        updates={},
        selected_next_node_id="close_conversation",
    )
    builder.add_turn(
        customer_text="Okay.",
        agent_text="Please verify only with the official callback team.",
        updates={},
        selected_next_node_id="close_conversation",
    )
    builder.add_turn(
        customer_text="Fine.",
        agent_text="Thank you.",
        updates={},
        selected_next_node_id="close_conversation",
    )
    builder.add_turn(
        customer_text="Goodbye.",
        agent_text="Goodbye.",
        updates={},
        selected_next_node_id="close_conversation",
    )
    builder.add_turn(
        customer_text="Understood.",
        agent_text="The suspicious-customer flow is complete.",
        updates={"discount_stage": "closed"},
        selected_next_node_id="close_conversation",
    )
    return "supervisor_callback_requested"


def _generate_bad_actor(builder: TrajectoryBuilder) -> str:
    builder.add_turn(
        customer_text="I am not paying anything, and do not call me again.",
        agent_text="I understand your position. I will keep the conversation professional. If you want, I can connect you with a supervisor.",
        updates={
            "customer_payment_posture": "refuses_to_pay",
            "customer_payment_willingness": 0.0,
            "verification_status": "in_progress",
        },
        selected_next_node_id="verify_identity",
    )
    builder.add_turn(
        customer_text="I said I am not talking to you.",
        agent_text="Understood. I will route this to a human collections specialist.",
        tool_calls=[make_escalation_tool(builder.context["case"], "hostile_customer")],
        updates={"followup_date": _date_str(1)},
        selected_next_node_id="human_escalation",
    )
    for customer_text, agent_text in [
        ("Do whatever you want.", "I have recorded your request for human follow-up."),
        ("Fine.", "A supervisor will review the case."),
        ("Okay.", "Thank you."),
        ("Bye.", "Goodbye."),
        ("Understood.", "The escalation record is complete."),
        ("All right.", "The conversation is now closed."),
    ]:
        builder.add_turn(customer_text=customer_text, agent_text=agent_text, updates={}, selected_next_node_id="close_conversation")
    builder.state["discount_stage"] = "closed"
    sync_derived_state(builder.state)
    return "hostile_customer_escalated"


def _generate_debt_dispute(builder: TrajectoryBuilder, dispute_reason: str) -> str:
    context = builder.context
    builder.add_turn(
        customer_text="I do not think this debt is correct.",
        agent_text=_agent_due_prompt(context),
        updates={"verification_status": "verified", "identity_verified": True, "verified_dob": True, "verified_mobile": True, "verification_verified_fields": ["dob", "mobile"], "verification_missing_fields": []},
        selected_next_node_id="explain_dues",
    )
    builder.add_turn(
        customer_text=f"I am disputing it because {dispute_reason}.",
        agent_text="I understand. I can note the dispute and arrange a manual review.",
        updates={"customer_payment_posture": "negotiating", "customer_payment_willingness": 0.35},
        selected_next_node_id="human_escalation",
    )
    builder.add_turn(
        customer_text="Please do that.",
        agent_text="I have logged the dispute for review.",
        tool_calls=[make_escalation_tool(context["case"], "debt_dispute")],
        updates={"followup_date": _date_str(2)},
        selected_next_node_id="human_escalation",
    )
    for customer_text, agent_text in [
        ("When will I hear back?", "The review team should respond within two business days."),
        ("Okay.", "No additional dues discussion will happen until that review."),
        ("Fine.", "Thank you."),
        ("Bye.", "Goodbye."),
        ("Understood.", "The dispute workflow is complete."),
    ]:
        builder.add_turn(customer_text=customer_text, agent_text=agent_text, updates={}, selected_next_node_id="close_conversation")
    return "debt_dispute_escalated"


def _generate_human_escalation(builder: TrajectoryBuilder) -> str:
    builder.add_turn(
        customer_text="I want to speak to a human supervisor.",
        agent_text="Certainly. I can route this to a human collections specialist.",
        updates={"customer_payment_posture": "negotiating", "customer_payment_willingness": 0.4},
        selected_next_node_id="human_escalation",
    )
    builder.add_turn(
        customer_text="Please arrange that today.",
        agent_text="I have placed the escalation request.",
        tool_calls=[make_escalation_tool(builder.context["case"], "customer_requested_human")],
        updates={"followup_date": _date_str(1)},
        selected_next_node_id="human_escalation",
    )
    for customer_text, agent_text in [
        ("Will they call this number?", "Yes, the team will use the registered number unless you advise otherwise."),
        ("That is fine.", "Thank you. The request is logged."),
        ("Okay.", "Take care."),
        ("Bye.", "Goodbye."),
        ("Understood.", "The human-escalation workflow is complete."),
        ("All right.", "Conversation closed."),
    ]:
        builder.add_turn(customer_text=customer_text, agent_text=agent_text, updates={}, selected_next_node_id="close_conversation")
    return "human_escalation_requested"


def _generate_multi_turn_negotiation(builder: TrajectoryBuilder) -> str:
    context = builder.context
    total_due = context["case"]["overdue_amount"] + context["case"]["late_fee"]
    first_amount = _round_money(total_due * 0.28)
    second_amount = _round_money(total_due * 0.4)
    final_amount = _round_money(total_due * 0.52)
    builder.add_turn(
        customer_text="I lost my job and I cannot pay the full amount right now.",
        agent_text="I am sorry to hear that. Let us work through practical options.",
        updates={
            "conversation_mode": "hardship_negotiation",
            "negotiation_stage": "discovering_hardship",
            "customer_payment_posture": "cannot_pay",
            "customer_payment_willingness": 0.24,
            "hardship_reason": "job_loss",
            "hardship_context": {"hardship_detected": True, "hardship_reason": "job_loss", "confidence": 0.96},
        },
        selected_next_node_id="assess_affordability",
    )
    builder.add_turn(
        customer_text=f"I might manage {_fmt_money(first_amount)} this month.",
        agent_text="I will use that as a starting point for planning.",
        extracted={"customer_payment_capacity": first_amount},
        updates={
            "customer_payment_posture": "partial_now",
            "customer_payment_capacity": first_amount,
            "customer_payment_willingness": 0.42,
            "discount_requested": True,
            "discount_stage": "requested",
            "negotiation_stage": "assessing_capacity",
        },
        response_target="discount_planning_agent",
        selected_next_node_id="present_arrangement_options",
        include_discount_agent=True,
        offer_payload=_make_offer_payload(
            context=context,
            state=builder.state,
            amount=first_amount,
            hardship_reason="job_loss",
            offer_type="arrangement",
        ),
    )
    builder.add_turn(
        customer_text="What does that mean for the remaining balance?",
        agent_text="The review suggests a staged arrangement, but the initial amount may need to be higher.",
        updates={"discount_stage": "offered", "discount_offered": True, "negotiation_stage": "evaluating_options"},
        selected_next_node_id="present_arrangement_options",
    )
    builder.add_turn(
        customer_text=f"I can stretch to {_fmt_money(second_amount)} if it helps.",
        agent_text="Thank you. I will update the arrangement request with that revised amount.",
        extracted={"customer_payment_capacity": second_amount},
        updates={
            "customer_payment_posture": "partial_now",
            "customer_payment_capacity": second_amount,
            "customer_payment_willingness": 0.55,
            "counter_offer_present": True,
            "discount_stage": "counter_offer",
            "negotiation_stage": "negotiating_plan",
        },
        response_target="discount_planning_agent",
        selected_next_node_id="negotiate_installment",
        include_discount_agent=True,
        offer_payload=_make_offer_payload(
            context=context,
            state=builder.state,
            amount=second_amount,
            hardship_reason="job_loss",
            offer_type="arrangement",
        ),
    )
    builder.add_turn(
        customer_text="Please tell me the best final arrangement you can make.",
        agent_text=f"The updated review indicates that {_fmt_money(final_amount)} would make the arrangement workable.",
        updates={
            "discount_stage": "offered",
            "discount_offered": True,
            "customer_payment_capacity": final_amount,
            "negotiation_stage": "awaiting_customer_decision",
        },
        selected_next_node_id="present_arrangement_options",
    )
    promise_date = _date_str(4)
    builder.add_turn(
        customer_text=f"I can commit to {_fmt_money(final_amount)} by {promise_date}.",
        agent_text="That works. I will capture that commitment under the reviewed arrangement.",
        tool_calls=[make_promise_tool(context["case"], final_amount, promise_date)],
        updates={
            "customer_payment_posture": "promise_to_pay",
            "customer_payment_willingness": 0.74,
            "promise_to_pay_date": promise_date,
            "discount_stage": "accepted",
            "discount_accepted": True,
            "negotiation_stage": "confirming_commitment",
        },
        selected_next_node_id="confirm_commitment",
    )
    builder.add_turn(
        customer_text="Please send me the details by message as well.",
        agent_text="I have noted that. We will expect the agreed payment on the committed date.",
        updates={"followup_date": promise_date},
        selected_next_node_id="close_conversation",
    )
    builder.add_turn(
        customer_text="Thank you.",
        agent_text="You are welcome. Please keep us updated if anything changes before the promised date.",
        updates={"discount_stage": "closed"},
        selected_next_node_id="close_conversation",
    )
    return "multi_turn_negotiation_arrangement"


def generate_script(spec: CanonicalSpec, variant_index: int, script_number: int, fixtures: dict[str, Any]) -> dict[str, Any]:
    seed = 10_000 + script_number
    rnd = random.Random(seed)
    script_id = f"COLL_TRAJ_{script_number:04d}"
    context = build_context(
        script_number=script_number,
        spec=spec,
        variant_index=variant_index,
        fixtures=fixtures,
        rnd=rnd,
    )
    state = make_initial_state(context)
    builder = TrajectoryBuilder(script_id=script_id, context=context, state=state, rnd=rnd)

    if spec.flow == "wrong_contact":
        expected_outcome = _generate_wrong_contact(builder)
    elif spec.flow == "suspicious_customer":
        expected_outcome = _generate_suspicious(builder)
    elif spec.flow == "bad_actor":
        expected_outcome = _generate_bad_actor(builder)
    elif spec.flow == "debt_dispute":
        expected_outcome = _generate_debt_dispute(builder, spec.dispute_reason or "the amount does not match my records")
    elif spec.flow == "human_escalation":
        _verification_success_sequence(builder=builder, pattern="direct")
        expected_outcome = _generate_human_escalation(builder)
    elif spec.flow == "verification_failure":
        expected_outcome = _generate_verification_failure(builder)
    else:
        _verification_success_sequence(builder=builder, pattern=spec.verification_pattern)
        if spec.flow == "pay_now":
            expected_outcome = _generate_pay_now(builder, reluctant=False)
        elif spec.flow == "pay_now_reluctance":
            expected_outcome = _generate_pay_now(builder, reluctant=True)
        elif spec.flow == "promise_to_pay":
            expected_outcome = _generate_promise(builder, with_followup=False)
        elif spec.flow == "promise_followup":
            expected_outcome = _generate_promise(builder, with_followup=True)
        elif spec.flow == "partial_payment":
            expected_outcome = _generate_partial(builder, to_full=False, settlement=False, counter_offer=False)
        elif spec.flow == "partial_to_full":
            expected_outcome = _generate_partial(builder, to_full=True, settlement=False, counter_offer=False)
        elif spec.flow == "settlement_request":
            expected_outcome = _generate_partial(builder, to_full=False, settlement=True, counter_offer=False)
        elif spec.flow == "settlement_counter":
            expected_outcome = _generate_partial(builder, to_full=False, settlement=True, counter_offer=True)
        elif spec.flow == "hardship_job_loss":
            expected_outcome = _generate_hardship(builder, hardship_reason="job_loss", accepted=None, rejected=False)
        elif spec.flow == "hardship_medical":
            expected_outcome = _generate_hardship(builder, hardship_reason="medical_emergency", accepted=None, rejected=False)
        elif spec.flow == "hardship_family":
            expected_outcome = _generate_hardship(builder, hardship_reason="family_emergency", accepted=None, rejected=False)
        elif spec.flow == "hardship_discount_accepted":
            expected_outcome = _generate_hardship(builder, hardship_reason=spec.hardship_reason or "job_loss", accepted=True, rejected=False)
        elif spec.flow == "hardship_discount_rejected":
            expected_outcome = _generate_hardship(builder, hardship_reason=spec.hardship_reason or "medical_emergency", accepted=False, rejected=True)
        elif spec.flow == "customer_changes_mind":
            expected_outcome = _generate_customer_changes_mind(builder, spec.posture_path)
        elif spec.flow == "verification_success":
            expected_outcome = _generate_verification_success_resolution(builder)
        elif spec.flow == "multi_turn_negotiation":
            expected_outcome = _generate_multi_turn_negotiation(builder)
        else:
            raise ValueError(f"Unsupported flow: {spec.flow}")

    if not (8 <= len(builder.turns) <= 15):
        raise ValueError(f"{script_id} generated invalid turn count {len(builder.turns)} for {spec.flow}")

    final_state = deepcopy(builder.state)
    return {
        "script_id": script_id,
        "category": spec.category,
        "difficulty": spec.difficulty,
        "context": context,
        "initial_state": make_initial_state(context),
        "conversation": builder.turns,
        "final_state": final_state,
        "expected_response_target": str(final_state.get("response_target", "customer")),
        "expected_outcome": expected_outcome,
    }


def _generate_verification_success_resolution(builder: TrajectoryBuilder) -> str:
    context = builder.context
    plan_date = _date_str(2)
    builder.add_turn(
        customer_text="Thanks. I needed to understand the dues before deciding.",
        agent_text="Of course. Now that verification is complete, we can discuss the repayment path.",
        updates={"customer_payment_posture": "negotiating", "customer_payment_willingness": 0.65},
        selected_next_node_id="collect_payment_intent",
    )
    builder.add_turn(
        customer_text=f"I can pay {_fmt_money(context['case']['emi_amount'])} by {plan_date}.",
        agent_text="Thank you. I can capture that as your commitment.",
        updates={
            "customer_payment_posture": "promise_to_pay",
            "customer_payment_willingness": 0.7,
            "promise_to_pay_date": plan_date,
            "negotiation_stage": "confirming_commitment",
        },
        selected_next_node_id="capture_promise",
    )
    builder.add_turn(
        customer_text="Please do that.",
        agent_text="The commitment has been recorded.",
        tool_calls=[make_promise_tool(context["case"], context["case"]["emi_amount"], plan_date)],
        updates={},
        selected_next_node_id="capture_promise",
    )
    for customer_text, agent_text in [
        ("Okay.", "We will expect the payment on the agreed date."),
        ("Thanks.", "Thank you."),
        ("Bye.", "Goodbye."),
    ]:
        builder.add_turn(customer_text=customer_text, agent_text=agent_text, updates={}, selected_next_node_id="close_conversation")
    return "verification_success_promise"


def _generate_customer_changes_mind(builder: TrajectoryBuilder, posture_path: tuple[str, ...]) -> str:
    context = builder.context
    total_due = context["case"]["overdue_amount"] + context["case"]["late_fee"]
    partial_amount = _round_money(total_due * 0.4)
    if posture_path == ("cannot_pay", "partial_now", "pay_now"):
        builder.add_turn(
            customer_text="I cannot pay anything today because my cash flow is blocked.",
            agent_text="I understand. Let us see whether there is a smaller amount you can manage to start with.",
            updates={
                "conversation_mode": "hardship_negotiation",
                "negotiation_stage": "assessing_capacity",
                "customer_payment_posture": "cannot_pay",
                "customer_payment_willingness": 0.2,
                "hardship_reason": "cashflow_issue",
                "hardship_context": {"hardship_detected": True, "hardship_reason": "cashflow_issue", "confidence": 0.88},
            },
            selected_next_node_id="assess_affordability",
        )
        builder.add_turn(
            customer_text=f"Actually I may be able to arrange {_fmt_money(partial_amount)} by evening.",
            agent_text="That helps. I will treat this as a partial-payment discussion.",
            extracted={"customer_payment_capacity": partial_amount},
            updates={
                "customer_payment_posture": "partial_now",
                "customer_payment_capacity": partial_amount,
                "customer_payment_willingness": 0.52,
                "discount_requested": True,
                "discount_stage": "requested",
            },
            response_target="discount_planning_agent",
            selected_next_node_id="present_arrangement_options",
            include_discount_agent=True,
            offer_payload=_make_offer_payload(
                context=context,
                state=builder.state,
                amount=partial_amount,
                hardship_reason="cashflow_issue",
                offer_type="arrangement",
            ),
        )
        builder.add_turn(
            customer_text="If I can borrow some money, I may clear the full amount instead.",
            agent_text="Understood. If you can pay in full today, I can switch this to a pay-now flow.",
            updates={
                "customer_payment_posture": "pay_now",
                "customer_payment_capacity": total_due,
                "customer_payment_willingness": 0.9,
                "discount_stage": "closed",
                "discount_requested": False,
            },
            selected_next_node_id="collect_payment",
        )
        builder.add_turn(
            customer_text="Yes, please send the full payment link.",
            agent_text="I have generated the full payment link.",
            tool_calls=[make_payment_link_tool(context["case"], total_due, builder.script_id)],
            updates={"customer_payment_posture": "pay_now", "customer_payment_willingness": 1.0},
            selected_next_node_id="collect_payment",
        )
        for customer_text, agent_text in [
            ("I will complete it shortly.", "Thank you. Please confirm once done."),
            ("Done.", "Thank you for resolving it today."),
            ("Okay.", "Goodbye."),
        ]:
            builder.add_turn(customer_text=customer_text, agent_text=agent_text, updates={}, selected_next_node_id="close_conversation")
        return "posture_transition_to_full_payment"

    builder.add_turn(
        customer_text="I am not paying anything right now.",
        agent_text="I understand you are not ready to commit. If you want, we can still discuss options.",
        updates={"customer_payment_posture": "refuses_to_pay", "customer_payment_willingness": 0.05},
        selected_next_node_id="collect_payment_intent",
    )
    builder.add_turn(
        customer_text="What options are you talking about?",
        agent_text="We can discuss arrangements if you want to work something out.",
        updates={"customer_payment_posture": "negotiating", "customer_payment_willingness": 0.36},
        selected_next_node_id="present_arrangement_options",
    )
    builder.add_turn(
        customer_text=f"I may be able to arrange {_fmt_money(partial_amount)} this week.",
        agent_text="That gives us something concrete to work with.",
        extracted={"customer_payment_capacity": partial_amount},
        updates={
            "customer_payment_posture": "partial_now",
            "customer_payment_capacity": partial_amount,
            "customer_payment_willingness": 0.57,
            "discount_requested": True,
            "discount_stage": "requested",
        },
        response_target="discount_planning_agent",
        selected_next_node_id="present_arrangement_options",
        include_discount_agent=True,
        offer_payload=_make_offer_payload(
            context=context,
            state=builder.state,
            amount=partial_amount,
            hardship_reason=None,
            offer_type="arrangement",
        ),
    )
    for customer_text, agent_text in [
        ("Please note that and tell me the next step.", "I will note it and prepare the next arrangement step."),
        ("Okay.", "Thank you."),
        ("Bye.", "Goodbye."),
    ]:
        builder.add_turn(customer_text=customer_text, agent_text=agent_text, updates={}, selected_next_node_id="close_conversation")
    return "posture_transition_to_partial"


def build_canonical_specs() -> list[CanonicalSpec]:
    distribution = [
        ("Pay Now", "pay_now", 3),
        ("Pay Now After Reluctance", "pay_now_reluctance", 3),
        ("Promise To Pay", "promise_to_pay", 3),
        ("Promise To Pay With Followup", "promise_followup", 2),
        ("Partial Payment", "partial_payment", 4),
        ("Partial Payment To Full Payment", "partial_to_full", 3),
        ("Hardship Job Loss", "hardship_job_loss", 3),
        ("Hardship Medical Emergency", "hardship_medical", 2),
        ("Hardship Family Crisis", "hardship_family", 2),
        ("Hardship Discount Accepted", "hardship_discount_accepted", 3),
        ("Hardship Discount Rejected", "hardship_discount_rejected", 2),
        ("Settlement Request", "settlement_request", 3),
        ("Settlement Request With Counter Offer", "settlement_counter", 3),
        ("Customer Changes Mind", "customer_changes_mind", 2),
        ("Verification Success", "verification_success", 2),
        ("Verification Failure", "verification_failure", 2),
        ("Wrong Contact", "wrong_contact", 1),
        ("Suspicious Customer", "suspicious_customer", 1),
        ("Bad Actor", "bad_actor", 1),
        ("Debt Dispute", "debt_dispute", 2),
        ("Human Escalation", "human_escalation", 2),
        ("Multi Turn Negotiation", "multi_turn_negotiation", 1),
    ]
    assert sum(count for _, _, count in distribution) == 50
    difficulties = ["easy", "medium", "hard", "adversarial"]
    verification_patterns = ["direct", "split", "failed_once"]
    customer_styles = ["cooperative", "cautious", "frustrated", "practical"]
    hardship_cycle = ["job_loss", "medical_emergency", "family_emergency"]
    dispute_reasons = [
        "I already made a payment last week",
        "the amount is higher than the statement I have",
    ]

    specs: list[CanonicalSpec] = []
    counter = 1
    for category, flow, count in distribution:
        for index in range(count):
            specs.append(
                CanonicalSpec(
                    canonical_id=f"CANONICAL_{counter:02d}",
                    category=category,
                    flow=flow,
                    difficulty=difficulties[(counter + index) % len(difficulties)],
                    verification_pattern=verification_patterns[(counter + index) % len(verification_patterns)],
                    customer_style=customer_styles[(counter + index) % len(customer_styles)],
                    hardship_reason=hardship_cycle[(counter + index) % len(hardship_cycle)]
                    if "hardship" in flow
                    else None,
                    posture_path=(
                        ("cannot_pay", "partial_now", "pay_now")
                        if flow == "customer_changes_mind" and index == 0
                        else ("refuses_to_pay", "negotiating", "partial_now")
                        if flow == "customer_changes_mind"
                        else ()
                    ),
                    counter_offer=flow == "settlement_counter",
                    dispute_reason=dispute_reasons[index % len(dispute_reasons)] if flow == "debt_dispute" else None,
                )
            )
            counter += 1
    return specs


def validate_dataset(records: list[dict[str, Any]]) -> None:
    if len(records) != 1000:
        raise ValueError(f"Expected 1000 scripts, found {len(records)}")
    seen: set[str] = set()
    for record in records:
        script_id = record["script_id"]
        if script_id in seen:
            raise ValueError(f"Duplicate script_id: {script_id}")
        seen.add(script_id)
        conversation = record["conversation"]
        if not (8 <= len(conversation) <= 15):
            raise ValueError(f"{script_id} has invalid turn count {len(conversation)}")
        if not any(turn.get("state_transitions") for turn in conversation):
            raise ValueError(f"{script_id} did not contain any state transitions")
        if not any(turn.get("nodes_traversed") for turn in conversation):
            raise ValueError(f"{script_id} did not contain any node outputs")
        for turn in conversation:
            if not turn.get("nodes_traversed"):
                raise ValueError(f"{script_id} turn {turn['turn_id']} has no node outputs")
            for node in turn["nodes_traversed"]:
                if not node.get("expected_output"):
                    raise ValueError(f"{script_id} turn {turn['turn_id']} node {node['node']} has empty expected_output")
        final_state = record["final_state"]
        posture = final_state["customer_payment_posture"]
        if posture not in PAYMENT_POSTURES:
            raise ValueError(f"{script_id} invalid posture {posture}")
        discount_stage = final_state["discount_stage"]
        if discount_stage not in DISCOUNT_STAGES:
            raise ValueError(f"{script_id} invalid discount_stage {discount_stage}")
        if final_state["verification_status"] not in {"not_started", "in_progress", "verified", "failed"}:
            raise ValueError(f"{script_id} invalid verification_status")


def write_manifest(records: list[dict[str, Any]]) -> None:
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "script_id",
                "category",
                "difficulty",
                "turn_count",
                "expected_response_target",
                "expected_outcome",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "script_id": record["script_id"],
                    "category": record["category"],
                    "difficulty": record["difficulty"],
                    "turn_count": len(record["conversation"]),
                    "expected_response_target": record["expected_response_target"],
                    "expected_outcome": record["expected_outcome"],
                }
            )


def write_eda(records: list[dict[str, Any]]) -> None:
    category_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    tool_counts: dict[str, int] = {}
    response_target_counts: dict[str, int] = {}
    discount_planner_count = 0
    total_turns = 0

    for record in records:
        category_counts[record["category"]] = category_counts.get(record["category"], 0) + 1
        difficulty_counts[record["difficulty"]] = difficulty_counts.get(record["difficulty"], 0) + 1
        response_target_counts[record["expected_response_target"]] = response_target_counts.get(record["expected_response_target"], 0) + 1
        total_turns += len(record["conversation"])
        for turn in record["conversation"]:
            if any(node["node"] == "discount_planning_agent" for node in turn["nodes_traversed"]):
                discount_planner_count += 1
            for tool in turn["tool_calls"]:
                tool_name = tool["tool"]
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

    rows: list[dict[str, Any]] = []
    for key, value in sorted(category_counts.items()):
        rows.append({"metric": "category_count", "dimension": key, "value": value})
    for key, value in sorted(difficulty_counts.items()):
        rows.append({"metric": "difficulty_count", "dimension": key, "value": value})
    rows.append({"metric": "average_turns", "dimension": "overall", "value": round(total_turns / max(len(records), 1), 2)})
    for key, value in sorted(tool_counts.items()):
        rows.append({"metric": "tool_frequency", "dimension": key, "value": value})
    for key, value in sorted(response_target_counts.items()):
        rows.append({"metric": "response_target_frequency", "dimension": key, "value": value})
    rows.append({"metric": "discount_planner_frequency", "dimension": "turns_with_discount_planner", "value": discount_planner_count})

    with EDA_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "dimension", "value"])
        writer.writeheader()
        writer.writerows(rows)


def generate_dataset() -> list[dict[str, Any]]:
    fixtures = load_fixture_pool()
    specs = build_canonical_specs()
    records: list[dict[str, Any]] = []
    script_number = 1
    for spec in specs:
        for variant_index in range(20):
            records.append(generate_script(spec, variant_index, script_number, fixtures))
            script_number += 1
    validate_dataset(records)
    return records


def write_dataset(records: list[dict[str, Any]]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with DATASET_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")
    write_manifest(records)
    write_eda(records)


def main() -> None:
    records = generate_dataset()
    write_dataset(records)
    print(f"Wrote {len(records)} records to {DATASET_PATH}")


if __name__ == "__main__":
    main()
