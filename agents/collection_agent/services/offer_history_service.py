"""Helpers for loaded offer-history state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class OfferHistoryService:
    """Summarizes prior concession / settlement offer history from state."""

    @staticmethod
    def get_history(source: dict[str, Any]) -> dict[str, Any]:
        history = source.get("offer_history")
        return dict(history) if isinstance(history, dict) else {}

    @staticmethod
    def summarize(history: dict[str, Any] | None) -> dict[str, Any]:
        payload = dict(history) if isinstance(history, dict) else {}
        offers = payload.get("offers") if isinstance(payload.get("offers"), list) else []
        normalized = [item for item in offers if isinstance(item, dict)]
        rejected_count = 0
        accepted_count = 0
        counter_offer_count = 0
        previous_discount_offers: list[dict[str, Any]] = []
        last_offer_status = None
        last_offer_pct = None
        for offer in normalized:
            status = str(offer.get("status", "")).strip().lower()
            offer_pct = offer.get("offer_pct")
            if status == "rejected":
                rejected_count += 1
            if status == "accepted":
                accepted_count += 1
            if status == "counter_offer":
                counter_offer_count += 1
            if offer_pct not in {None, ""}:
                previous_discount_offers.append(
                    {
                        "offer_pct": offer_pct,
                        "status": status or None,
                    }
                )
                last_offer_pct = offer_pct
            if status:
                last_offer_status = status
        return {
            "offer_count": len(normalized),
            "rejected_count": rejected_count,
            "accepted_count": accepted_count,
            "counter_offer_count": counter_offer_count,
            "last_offer_status": last_offer_status,
            "last_offer_pct": last_offer_pct,
            "previous_discount_offers": previous_discount_offers,
        }
