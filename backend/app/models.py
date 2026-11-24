"""Typed contracts: domain records, LLM output schemas and API responses."""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Role(str, Enum):
    SDR = "SDR"
    AE = "AE"
    MANAGER = "MANAGER"


Sensitivity = Literal["public", "internal", "confidential", "restricted"]
SENSITIVITY_LEVEL: dict[str, int] = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}
RecordType = Literal["account", "contact", "opportunity", "activity", "email", "meeting", "product_doc"]


class User(BaseModel):
    id: str
    name: str
    role: Role
    territories: list[str]


class Account(BaseModel):
    id: str
    name: str
    industry: str
    territory: str
    tier: str
    employees: int
    city: str
    owner_id: str
    owner_name: str
    use_case: str
    sensitivity: Sensitivity = "internal"
    synthetic: bool = True


class Contact(BaseModel):
    id: str
    account_id: str
    name: str
    title: str
    email: str
    phone: str


class Opportunity(BaseModel):
    id: str
    account_id: str
    territory: str
    name: str
    stage: str
    amount_usd: int
    close_date: str
    probability: int
    products: list[str]
    competitor: str | None
    next_step: str
    sensitivity: Sensitivity


class Activity(BaseModel):
    id: str
    account_id: str
    territory: str
    type: str
    date: str
    author: str
    text: str
    sensitivity: Sensitivity


class Email(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: str
    account_id: str
    territory: str
    date: str
    sender: str = Field(alias="from")
    to: str
    subject: str
    body: str
    sensitivity: Sensitivity


class Meeting(BaseModel):
    id: str
    account_id: str
    territory: str
    title: str
    start: str
    duration_minutes: int
    attendee_contact_ids: list[str]
    owner_id: str
    agenda: str
    sensitivity: Sensitivity


class DocSection(BaseModel):
    title: str
    text: str


class ProductDoc(BaseModel):
    id: str
    name: str
    territory: str = "GLOBAL"
    sensitivity: Sensitivity
    sections: list[DocSection]


class Evidence(BaseModel):
    """One citable unit handed to the model: a tool result or a retrieved chunk."""
    source_id: str
    record_id: str
    source_type: RecordType
    account_id: str | None
    territory: str
    sensitivity: Sensitivity
    title: str
    text: str
    score: float | None = None
    quarantined: bool = False


# ---------- LLM output schemas (validated with Pydantic) ----------

class Claim(BaseModel):
    text: str = Field(min_length=3, max_length=600)
    source_ids: list[str] = Field(min_length=1, max_length=6)


class AccountBrief(BaseModel):
    summary: str = Field(min_length=10, max_length=1200)
    summary_source_ids: list[str] = Field(min_length=1)
    key_facts: list[Claim] = Field(max_length=8)
    recent_activity: list[Claim] = Field(max_length=6)
    risks: list[Claim] = Field(max_length=6)


class Attendee(BaseModel):
    name: str
    title: str


class MeetingPrep(BaseModel):
    objectives: list[Claim] = Field(min_length=1, max_length=5)
    talking_points: list[Claim] = Field(max_length=6)
    questions_to_ask: list[str] = Field(max_length=6)
    risks: list[Claim] = Field(max_length=5)


ActionType = Literal["follow_up", "schedule_meeting", "share_content", "involve_specialist",
                     "escalate_internal", "update_crm"]


class NextBestAction(BaseModel):
    action: str = Field(min_length=5, max_length=300)
    action_type: ActionType
    priority: Literal["high", "medium", "low"]
    rationale: str = Field(min_length=5, max_length=500)
    source_ids: list[str] = Field(min_length=1, max_length=6)


class NextBestActions(BaseModel):
    actions: list[NextBestAction] = Field(min_length=1, max_length=5)


class Answer(BaseModel):
    answer: str = Field(min_length=1, max_length=2000)
    claims: list[Claim] = Field(max_length=8)


class ToolCall(BaseModel):
    name: Literal["crm_lookup", "opportunity_data", "product_catalogue", "calendar"]
    args: dict[str, str] = Field(default_factory=dict)


class ToolPlan(BaseModel):
    account_id: str | None = None
    tool_calls: list[ToolCall] = Field(max_length=6)
    search_query: str


# ---------- API envelopes ----------

class Out(BaseModel):
    """Response models: fields with defaults are still always present, so mark them required in OpenAPI."""
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)


class Citation(Out):
    source_id: str
    record_id: str
    source_type: RecordType
    title: str
    snippet: str


class TokenUsage(Out):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated: bool = False


class GuardrailEvent(Out):
    type: Literal["prompt_injection", "pii_redaction", "invalid_citation", "validation_retry", "fallback"]
    source_id: str | None = None
    detail: str


class ToolCallRecord(Out):
    name: str
    args: dict[str, str]
    result_ids: list[str]
    ms: float


class AccessScope(Out):
    role: Role
    territories: list[str]
    record_types: list[str]
    max_sensitivity: Sensitivity


class ResponseMeta(Out):
    trace_id: str
    provider: str
    model: str
    fallback_reason: str | None = None
    stages_ms: dict[str, float]
    tokens: TokenUsage | None = None
    redactions: int = 0
    guardrail_events: list[GuardrailEvent] = Field(default_factory=list)
    groundedness: float
    ungrounded_claims: list[str] = Field(default_factory=list)
    retrieved_ids: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    access_scope: AccessScope


class AccountSummary(Out):
    id: str
    name: str
    industry: str
    territory: str
    tier: str
    city: str
    owner_name: str
    use_case: str
    synthetic: bool = True


class BriefResponse(Out):
    account: AccountSummary
    brief: AccountBrief
    citations: list[Citation]
    meta: ResponseMeta


class MeetingInfo(Out):
    id: str
    title: str
    start: str
    duration_minutes: int
    agenda: str
    attendees: list[Attendee]


class MeetingPrepResponse(Out):
    account: AccountSummary
    meeting: MeetingInfo | None
    prep: MeetingPrep
    citations: list[Citation]
    meta: ResponseMeta


class NextBestActionsResponse(Out):
    account: AccountSummary
    result: NextBestActions
    citations: list[Citation]
    meta: ResponseMeta


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    account_id: str | None = None


class AskResponse(Out):
    question: str
    result: Answer
    citations: list[Citation]
    meta: ResponseMeta


class MeetingPrepRequest(BaseModel):
    meeting_id: str | None = None


class SourceRecord(Out):
    source_id: str
    record_id: str
    source_type: RecordType
    title: str
    text: str
    account_id: str | None
    territory: str
    sensitivity: Sensitivity
    flagged_injection: bool


class AuditEntry(Out):
    ts: str
    trace_id: str
    user_id: str
    role: str
    action: str
    account_id: str | None = None
    provider: str | None = None
    record_ids: list[str] = Field(default_factory=list)
    guardrail_events: list[str] = Field(default_factory=list)
    outcome: str
