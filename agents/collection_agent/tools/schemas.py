"""Schemas for Collection Agent tools."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class CaseRecord(BaseModel):
    case_id: str
    customer_id: str
    loan_id: str
    portfolio_id: str
    product: str
    dpd: int
    emi_amount: float
    overdue_amount: float
    late_fee: float
    status: str
    risk_band: str


class CaseFetchInput(BaseModel):
    case_id: str | None = None
    customer_id: str | None = None
    portfolio_id: str | None = None
    dpd_min: int | None = None
    dpd_max: int | None = None
    status: str | None = None
    limit: int = 20


class CaseFetchOutput(BaseModel):
    total: int
    cases: list[CaseRecord]


class CasePrioritizeInput(BaseModel):
    case_ids: list[str] = Field(default_factory=list)
    portfolio_id: str | None = None
    top_k: int = 20


class PrioritizedCase(BaseModel):
    case_id: str
    priority_score: float
    reason: str


class CasePrioritizeOutput(BaseModel):
    total: int
    queue: list[PrioritizedCase]


class ContactAttemptInput(BaseModel):
    case_id: str
    channel: Literal["voice", "whatsapp", "email", "sms"] = "voice"
    template_id: str | None = None
    reached: bool = False
    notes: str | None = None


class ContactAttemptOutput(BaseModel):
    attempt_id: str
    case_id: str
    channel: str
    status: Literal["reached", "not_reached"]
    created_at: datetime


class VerifyDOBInput(BaseModel):
    case_id: str | None = None
    customer_id: str | None = None
    dob: str


class VerifyDOBOutput(BaseModel):
    customer_id: str
    status: Literal["verified", "failed", "locked"]
    field: Literal["dob"] = "dob"
    failed_attempts: int


class VerifyMobileInput(BaseModel):
    case_id: str | None = None
    customer_id: str | None = None
    phone: str


class VerifyMobileOutput(BaseModel):
    customer_id: str
    status: Literal["verified", "failed", "locked"]
    field: Literal["phone"] = "phone"
    failed_attempts: int


class LoanPolicyLookupInput(BaseModel):
    case_id: str | None = None
    loan_id: str | None = None


class LoanPolicyLookupOutput(BaseModel):
    loan_id: str
    product: str
    max_promise_days: int
    waiver_allowed: bool
    max_waiver_pct: float
    restructure_allowed: bool
    notes: str


class DuesExplainBuildInput(BaseModel):
    case_id: str
    locale: str = "en-IN"


class DuesExplainBuildOutput(BaseModel):
    case_id: str
    customer_id: str
    total_due: float
    explanation: str


class OfferEligibilityInput(BaseModel):
    case_id: str
    hardship_flag: bool = False
    requested_waiver_pct: float | None = None


class OfferEligibilityOutput(BaseModel):
    case_id: str
    allowed: bool
    offer_type: Literal["none", "waiver", "restructure"]
    approved_waiver_pct: float
    reason_codes: list[str]
    recommended_next: str


class PaymentLinkCreateInput(BaseModel):
    case_id: str
    amount: float
    channel: Literal["whatsapp", "email", "sms"] = "whatsapp"
    expiry_minutes: int = 60


class PaymentLinkCreateOutput(BaseModel):
    payment_reference_id: str
    case_id: str
    amount: float
    payment_url: str
    expires_at: datetime


class PaymentStatusCheckInput(BaseModel):
    payment_reference_id: str
    simulate_status: Literal["pending", "success", "failed"] | None = None


class PaymentStatusCheckOutput(BaseModel):
    payment_reference_id: str
    status: Literal["pending", "success", "failed"]
    amount: float
    needs_additional_action: bool = False


class PromiseCaptureInput(BaseModel):
    case_id: str
    promised_date: str
    promised_amount: float
    channel: Literal["voice", "whatsapp", "email", "sms"] = "voice"


class PromiseCaptureOutput(BaseModel):
    promise_id: str
    case_id: str
    promised_date: str
    promised_amount: float
    status: str


class FollowupScheduleInput(BaseModel):
    case_id: str
    scheduled_for: str
    preferred_channel: Literal["voice", "whatsapp", "email", "sms"] = "voice"
    reason: str = "promise_to_pay"


class FollowupScheduleOutput(BaseModel):
    schedule_id: str
    case_id: str
    scheduled_for: str
    preferred_channel: str
    reason: str


class DispositionUpdateInput(BaseModel):
    case_id: str
    disposition_code: str
    notes: str


class DispositionUpdateOutput(BaseModel):
    case_id: str
    disposition_code: str
    audit_id: str
    updated_at: datetime


class HumanEscalationInput(BaseModel):
    case_id: str
    reason: str
    evidence_summary: str | None = None


class HumanEscalationOutput(BaseModel):
    escalation_id: str
    case_id: str
    queue: str
    priority: Literal["low", "medium", "high"]
    status: str


class ChannelSwitchInput(BaseModel):
    case_id: str
    from_channel: Literal["sms", "voice", "email", "whatsapp"] = "sms"
    to_channel: Literal["sms", "voice", "email", "whatsapp"] = "voice"
    reason: str = "customer_requested"


class ChannelSwitchOutput(BaseModel):
    switch_id: str
    case_id: str
    from_channel: str
    to_channel: str
    reason: str
    carried_context_summary: str


class PayByPhoneCollectInput(BaseModel):
    case_id: str
    amount: float
    consent_confirmed: bool = True
    simulate_status: Literal["success", "failed", "partial"] = "success"


class PayByPhoneCollectOutput(BaseModel):
    payment_id: str
    case_id: str
    collected_amount: float
    status: Literal["success", "failed", "partial"]
    receipt_reference: str


class PlanProposeInput(BaseModel):
    case_id: str
    hardship_reason: str = "income_reduction"
    revision_index: int = 0
    max_installment_amount: float | None = None


class PlanProposeOutput(BaseModel):
    plan_id: str
    case_id: str
    hardship_reason: str
    months: int
    monthly_amount: float
    first_due_date: str
    rationale: str
    status: Literal["proposed", "revised"]


class EntityExtractInput(BaseModel):
    text: str


class EntityExtractOutput(BaseModel):
    entities: dict[str, str] = Field(default_factory=dict)
    entity_keys: list[str] = Field(default_factory=list)


class VerificationEntityExtractInput(BaseModel):
    text: str
    required_fields: list[str] = Field(default_factory=list)
    include_name: bool = True


class VerificationEntityExtractOutput(BaseModel):
    entities: dict[str, str] = Field(default_factory=dict)
    required_fields: list[str] = Field(default_factory=list)
    detected_fields: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)


class VerificationMemoryVerifyInput(BaseModel):
    entities: dict[str, str] = Field(default_factory=dict)
    expected_challenge: dict[str, str] = Field(default_factory=dict)
    required_fields: list[str] = Field(default_factory=list)
    require_name_match: bool = False
    expected_name: str | None = None


class VerificationMemoryVerifyOutput(BaseModel):
    status: Literal["verified", "failed", "insufficient"]
    matched: bool
    missing_fields: list[str] = Field(default_factory=list)
    mismatched_fields: list[str] = Field(default_factory=list)
    required_fields: list[str] = Field(default_factory=list)
    compared_fields: list[str] = Field(default_factory=list)


class StrictScriptInput(BaseModel):
    scenario: str
    placeholders: dict[str, str] = Field(default_factory=dict)


class StrictScriptOutput(BaseModel):
    scenario: str
    message: str
    generated_at: datetime


class GenericSummaryOutput(BaseModel):
    summary: str
    metadata: dict[str, Any] = Field(default_factory=dict)
