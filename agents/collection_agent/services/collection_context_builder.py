"""Builds aggregated collection context before graph execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.collection_agent.services.assistance_program_service import AssistanceProgramService
from agents.collection_agent.services.customer_profile_service import CustomerProfileService
from agents.collection_agent.services.offer_history_service import OfferHistoryService
from agents.collection_agent.services.payment_history_service import PaymentHistoryService
from agents.collection_agent.tools.data_store import CollectionDataStore


@dataclass(slots=True)
class CollectionContextBuilder:
    """Aggregates all local collection datasets into one active context object."""

    store: CollectionDataStore

    def build(self, *, customer_id: str, case_id: str) -> dict[str, Any]:
        case_row = self.store.get_case(case_id=case_id, customer_id=customer_id) or {}
        resolved_customer_id = str(case_row.get("customer_id", customer_id)).strip() or customer_id
        customer = self.store.get_customer(resolved_customer_id) or {}
        loan_id = str(case_row.get("loan_id", "")).strip()
        policy = self.store.get_policy(loan_id) if loan_id else {}
        customer_profile = self.store.get_customer_profile(resolved_customer_id) or {}
        payment_history = self.store.get_payment_history(resolved_customer_id) or {}
        offer_history = self.store.get_offer_history(case_id) or {}
        assistance_programs = self.store.get_assistance_programs(
            product=str(case_row.get("product", "")).strip() or None
        )

        return {
            "customer": dict(customer),
            "case": dict(case_row),
            "policy": dict(policy) if isinstance(policy, dict) else {},
            "customer_profile": dict(customer_profile),
            "payment_history": dict(payment_history),
            "offer_history": dict(offer_history),
            "assistance_programs": [dict(item) for item in assistance_programs if isinstance(item, dict)],
        }

    def build_memory_updates(self, *, customer_id: str, case_id: str) -> dict[str, Any]:
        context = self.build(customer_id=customer_id, case_id=case_id)
        customer_profile = CustomerProfileService.get_profile(context)
        payment_history = PaymentHistoryService.get_history(context)
        offer_history = OfferHistoryService.get_history(context)
        assistance_programs = AssistanceProgramService.get_programs(context)
        return {
            "customer_profile": customer_profile,
            "payment_history": payment_history,
            "offer_history": offer_history,
            "assistance_programs": assistance_programs,
            "active_collection_context": context,
            "customer_profile_summary": CustomerProfileService.summarize(customer_profile),
            "payment_history_summary": PaymentHistoryService.summarize(payment_history),
            "offer_history_summary": OfferHistoryService.summarize(offer_history),
        }
