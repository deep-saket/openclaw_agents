"""Discount planning specialist agent (standalone)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class DiscountPlanningAgent:
    """Returns policy-aware discount planning recommendations.

    This is intentionally small for demo use. It is called by collection_agent
    through a handoff tool and returns only agent-to-agent payload.
    """

    llm: Any | None = None

    def run(self, handoff_payload: dict[str, Any]) -> dict[str, Any]:
        case_id = str(handoff_payload.get("case_id", "UNKNOWN"))
        hardship_reason = str(handoff_payload.get("hardship_reason", "income_reduction"))
        target_emi = handoff_payload.get("target_monthly_emi")
        customer_payment_capacity = handoff_payload.get("customer_payment_capacity")
        customer_payment_capacity_pct = handoff_payload.get("customer_payment_capacity_pct")
        customer_payment_posture = str(handoff_payload.get("customer_payment_posture", "unknown"))
        discount_stage = str(handoff_payload.get("discount_stage", "none"))
        customer_payment_willingness = handoff_payload.get("customer_payment_willingness")
        posture_history = handoff_payload.get("customer_payment_posture_history", [])
        previous_discount_offers = handoff_payload.get("previous_discount_offers", [])
        counter_offer_present = bool(handoff_payload.get("counter_offer_present", False))

        if isinstance(customer_payment_capacity, (int, float)) and float(customer_payment_capacity) > 0:
            monthly_emi = round(float(customer_payment_capacity), 2)
            tenure = 24 if counter_offer_present else 18
        elif isinstance(target_emi, (int, float)) and float(target_emi) > 0:
            monthly_emi = round(float(target_emi), 2)
            tenure = 24
        else:
            monthly_emi = 1500.0
            tenure = 18

        recommended_offer = {
            "case_id": case_id,
            "offer_type": "restructure",
            "waiver_pct": 0.0,
            "tenure_months": tenure,
            "monthly_emi": monthly_emi,
            "hardship_reason": hardship_reason,
            "customer_payment_posture": customer_payment_posture,
            "discount_stage": discount_stage,
        }
        return {
            "recommended_offer": recommended_offer,
            "offer_variants": [
                {**recommended_offer, "tenure_months": tenure + 6, "monthly_emi": round(monthly_emi * 0.85, 2)},
                {**recommended_offer, "tenure_months": max(12, tenure - 6), "monthly_emi": round(monthly_emi * 1.2, 2)},
            ],
            "input_context": {
                "customer_payment_capacity": customer_payment_capacity,
                "customer_payment_capacity_pct": customer_payment_capacity_pct,
                "customer_payment_willingness": customer_payment_willingness,
                "posture_history": posture_history,
                "previous_discount_offers": previous_discount_offers,
                "counter_offer_present": counter_offer_present,
            },
            "rationale": "Selected baseline restructure from hardship reason, posture, and payment-capacity handoff context.",
            "compliance_flags": ["demo_policy_check_pending"],
            "confidence": 0.62,
            "next_action_hint": "Present recommended_offer first, then use offer_variants if rejected.",
        }
