from __future__ import annotations

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

CHECKPOINT_DELIVERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a gentle caregiving assistant helping keep a highly structured daily routine "
            "for an Alzheimer's patient at the {stage} stage. Deliver exactly ONE short, concrete "
            "instruction for this routine checkpoint, one idea only. If a choice is offered, "
            "phrase it as a small multiple-choice question (e.g. 'Would you like water or tea?') "
            "instead of an open-ended question. Keep it under 20 words. You are speaking to the "
            "PATIENT.",
        ),
        (
            "human",
            "Checkpoint: {checkpoint_type}\nRoom note: {room_note}\nChoice to offer: {choice_note}\n"
            "Dietary restrictions: {dietary_restrictions}\n"
            "Give one instruction or choice for this checkpoint.",
        ),
    ]
)

DEVIATION_RESPONSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a routine-consistency assistant. An unplanned {severity} deviation from the "
            "patient's fixed daily routine has just been reported. Write exactly ONE short, calm "
            "sentence to the caregiver describing the deviation and asking them to confirm what "
            "adjustment they are making. Keep it under 25 words. Do not minimize a major "
            "deviation.",
        ),
        (
            "human",
            "Deviation type: {deviation_type}\nAffected checkpoint: {checkpoint_type}\n"
            "Description: {description}\nSeverity: {severity}\n"
            "Give one message to the caregiver.",
        ),
    ]
)

CHECKPOINT_ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a daily routine-management assistant. First privately assess whether the "
            "reply indicates this routine checkpoint was completed, then write warm, brief "
            "feedback. NEVER phrase feedback as a scolding correction — if it's still pending, "
            "gently note it as a follow-up rather than a failure.",
        ),
        (
            "human",
            "Checkpoint instruction given: {prompt}\nReply: {response}\n"
            "Assess and give one sentence of feedback.",
        ),
    ]
)

DEVIATION_ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a daily routine-management assistant. First privately assess whether the "
            "caregiver's reply indicates the routine deviation has been addressed, then write one "
            "calm, direct sentence acknowledging their response.",
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
            "Summarize this daily routine-management session for a family caregiver in 2-3 "
            "sentences. Be factual and gentle, and note the checkpoint completion rate and any "
            "recurring gaps that still need attention.",
        ),
        ("human", "Session data:\n{session_json}"),
    ]
)

ROUTINE_ALERT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write a short, urgent but calm alert for a family caregiver about a daily-routine "
            "issue (a missed medication supervision, a prevented double meal, a major routine "
            "deviation, or the patient not responding). State plainly what happened and what the "
            "caregiver should check on now. 2 sentences max.",
        ),
        ("human", "Session data:\n{session_json}"),
    ]
)


class CheckpointAssessment(BaseModel):
    resolved: Optional[bool] = Field(
        description="Whether this routine checkpoint is now completed; null if not assessable from the reply"
    )
    feedback: str = Field(
        description="One warm, brief sentence for the patient/caregiver. Never phrases a pending checkpoint as a failure."
    )


class DeviationAssessment(BaseModel):
    resolved: Optional[bool] = Field(
        description="Whether the caregiver's reply indicates the deviation has been addressed; null if not assessable"
    )
    feedback: str = Field(description="One calm, direct sentence acknowledging the caregiver's response")
