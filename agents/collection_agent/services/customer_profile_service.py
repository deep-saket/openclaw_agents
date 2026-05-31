"""Helpers for customer profile data already loaded into graph state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CustomerProfileService:
    """Returns customer-profile state slices and compact summaries."""

    @staticmethod
    def get_profile(source: dict[str, Any]) -> dict[str, Any]:
        profile = source.get("customer_profile")
        return dict(profile) if isinstance(profile, dict) else {}

    @staticmethod
    def summarize(profile: dict[str, Any] | None) -> dict[str, Any]:
        payload = dict(profile) if isinstance(profile, dict) else {}
        return {
            "employment_type": str(payload.get("employment_type", "")).strip() or None,
            "customer_segment": str(payload.get("customer_segment", "")).strip() or None,
            "risk_score": payload.get("risk_score"),
            "tenure_years": payload.get("tenure_years"),
            "previous_hardship_count": int(payload.get("previous_hardship_count", 0) or 0),
            "previous_settlement_count": int(payload.get("previous_settlement_count", 0) or 0),
            "previous_broken_promises": int(payload.get("previous_broken_promises", 0) or 0),
            "preferred_language": str(payload.get("preferred_language", "")).strip() or None,
            "preferred_channel": str(payload.get("preferred_channel", "")).strip() or None,
            "vulnerable_customer": bool(payload.get("vulnerable_customer", False)),
        }
