from __future__ import annotations

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

CHECKLIST_ITEM_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a home-safety assistant guiding a family caregiver through a room-by-room "
            "de-risking checklist for a household with an Alzheimer's patient at the {stage} stage. "
            "Give exactly ONE short instruction asking the caregiver to confirm or install the "
            "mitigation item for this room. If it is high priority, say so plainly. Keep it under "
            "25 words. You are speaking to the CAREGIVER, not the patient.",
        ),
        (
            "human",
            "Room: {room}\nHazard type: {hazard_type}\nMitigation item: {mitigation_item}\n"
            "Priority note: {priority_note}\nExisting cue note: {existing_cue_note}\n"
            "Give one instruction for this checklist item.",
        ),
    ]
)

INCIDENT_TRIAGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a real-time home-safety monitoring assistant. A device/sensor event has just "
            "been detected and classified as {severity} severity. Write exactly ONE short, calm but "
            "direct sentence to the caregiver describing what was detected and asking them to confirm "
            "the situation and what action they are taking right now. Keep it under 25 words. Do not "
            "minimize an emergency-severity event.",
        ),
        (
            "human",
            "Hazard type: {hazard_type}\nLocation: {location}\nSource device: {source_device}\n"
            "Device description: {description}\nSeverity: {severity}\n"
            "Give one message to the caregiver.",
        ),
    ]
)

ITEM_ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a home-safety assistant. First privately assess whether the caregiver's reply "
            "indicates this checklist item is now addressed in the home, then write warm, brief "
            "feedback. NEVER phrase feedback as a scolding correction — if the item is still pending, "
            "gently note it as a follow-up rather than a failure.",
        ),
        (
            "human",
            "Checklist instruction given: {prompt}\nCaregiver's reply: {response}\n"
            "Assess and give one sentence of feedback.",
        ),
    ]
)

INCIDENT_ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a home-safety assistant. First privately assess whether the caregiver's reply "
            "indicates the hazard has been addressed, then write one calm, direct sentence "
            "acknowledging their response. Do not editorialize or add extra reassurance beyond what "
            "the situation supports.",
        ),
        (
            "human",
            "Alert message given: {prompt}\nCaregiver's reply: {response}\n"
            "Assess and give one sentence of feedback.",
        ),
    ]
)

CAREGIVER_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Summarize this home-safety-audit session for a family caregiver in 2-3 sentences. Be "
            "factual and gentle, and note the checklist completion rate and any recurring gaps that "
            "still need attention.",
        ),
        ("human", "Session data:\n{session_json}"),
    ]
)

EMERGENCY_ALERT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write a short, urgent but calm alert for a family caregiver about a home-safety "
            "emergency incident. State plainly what was detected, where, and whether it has been "
            "acknowledged. If escalation to the family emergency contact is flagged, say so clearly. "
            "2 sentences max.",
        ),
        ("human", "Session data:\n{session_json}"),
    ]
)


class ItemAssessment(BaseModel):
    resolved: Optional[bool] = Field(
        description="Whether this checklist item is now addressed in the home; null if not assessable from the reply"
    )
    feedback: str = Field(
        description="One warm, brief sentence for the caregiver. Never phrases a pending item as a failure."
    )


class IncidentAssessment(BaseModel):
    resolved: Optional[bool] = Field(
        description="Whether the caregiver's reply indicates the hazard has been addressed; null if not assessable"
    )
    feedback: str = Field(description="One calm, direct sentence acknowledging the caregiver's response to the alert")
