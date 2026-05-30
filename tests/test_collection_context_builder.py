from __future__ import annotations

from pathlib import Path

from agents.collection_agent.services.collection_context_builder import CollectionContextBuilder
from agents.collection_agent.tools.data_store import CollectionDataStore


def test_collection_context_builder_aggregates_local_datasets() -> None:
    store = CollectionDataStore(base_dir=Path("agents/collection_agent"))
    builder = CollectionContextBuilder(store=store)

    context = builder.build(customer_id="CUST-2001", case_id="COLL-1001")
    updates = builder.build_memory_updates(customer_id="CUST-2001", case_id="COLL-1001")

    assert context["customer"]["customer_id"] == "CUST-2001"
    assert context["case"]["case_id"] == "COLL-1001"
    assert context["policy"]["loan_id"] == "LOAN-3001"
    assert context["customer_profile"]["customer_id"] == "CUST-2001"
    assert context["payment_history"]["customer_id"] == "CUST-2001"
    assert context["offer_history"]["case_id"] == "COLL-1001"
    assert context["assistance_programs"]

    assert updates["active_collection_context"]["case"]["case_id"] == "COLL-1001"
    assert updates["customer_profile_summary"]["preferred_channel"] == "phone"
    assert updates["payment_history_summary"]["payment_count"] >= 1
    assert updates["offer_history_summary"]["offer_count"] >= 1
