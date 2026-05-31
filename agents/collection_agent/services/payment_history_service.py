"""Helpers for loaded payment-history state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PaymentHistoryService:
    """Summarizes customer payment behavior from loaded state."""

    @staticmethod
    def get_history(source: dict[str, Any]) -> dict[str, Any]:
        history = source.get("payment_history")
        return dict(history) if isinstance(history, dict) else {}

    @staticmethod
    def summarize(history: dict[str, Any] | None) -> dict[str, Any]:
        payload = dict(history) if isinstance(history, dict) else {}
        payments = payload.get("payments") if isinstance(payload.get("payments"), list) else []
        normalized = [item for item in payments if isinstance(item, dict)]
        total_paid = 0.0
        last_payment_date = None
        last_payment_amount = None
        for payment in normalized:
            try:
                amount = float(payment.get("amount", 0.0) or 0.0)
            except Exception:
                amount = 0.0
            total_paid += amount
            date = str(payment.get("date", "")).strip() or None
            if date:
                last_payment_date = date
                last_payment_amount = amount
        return {
            "payment_count": len(normalized),
            "total_paid": round(total_paid, 2),
            "last_payment_date": last_payment_date,
            "last_payment_amount": last_payment_amount,
        }
