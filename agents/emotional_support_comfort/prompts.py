from __future__ import annotations

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

PREVENTION_HABIT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an emotional-support assistant coaching a family caregiver through a daily "
            "sundowning-prevention habit for an Alzheimer's patient at the {stage} stage. Give exactly "
            "ONE short instruction asking the caregiver to confirm or carry out this habit. If it is "
            "high priority (close to the dusk/sundowning window) say so plainly. Keep it under 25 words. "
            "You are speaking to the CAREGIVER, not the patient.",
        ),
        (
            "human",
            "Habit type: {habit_type}\nDescription: {description}\nPriority note: {priority_note}\n"
            "Give one instruction for this habit.",
        ),
    ]
)

INDEPENDENCE_TASK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an emotional-support assistant. Give exactly ONE short instruction encouraging the "
            "caregiver to invite the patient to try a manageable task, framed around effort and self-esteem "
            "rather than getting it right. Keep it under 25 words. You are speaking to the CAREGIVER.",
        ),
        (
            "human",
            "Patient name: {name}\nTask: {task}\nGive one instruction for the caregiver.",
        ),
    ]
)

OUTBURST_DEESCALATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a real-time emotional-outburst de-escalation coach guiding a family caregiver through "
            "an active sundowning/emotional-outburst episode. NEVER instruct the caregiver to correct, argue, "
            "blame, or use logic — always start by having them validate the patient's emotional state. Then "
            "guide exactly ONE concrete redirection action using the given redirection content. Include a "
            "brief reminder to lower their voice, slow their speech, and use a short comforting phrase (e.g. "
            "\"It's okay, I'm right here\"), plus gentle eye contact. Only mention light touch or hand-holding "
            "if contact is marked allowed. Write it as at most 3 short numbered steps, under 45 words total.",
        ),
        (
            "human",
            "Severity: {severity}\nAttempt number: {attempt_number}\nRedirection type: {redirection_type}\n"
            "Redirection content: {redirection_content}\nPhysical contact allowed: {contact_allowed}\n"
            "Preceding event: {preceding_event}\nGive the coaching steps.",
        ),
    ]
)

HABIT_ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an emotional-support assistant. First privately assess whether the caregiver's reply "
            "indicates this prevention habit was addressed today, then write warm, brief feedback. NEVER "
            "phrase feedback as a scolding correction — if the habit is still pending, gently note it as a "
            "follow-up rather than a failure.",
        ),
        (
            "human",
            "Habit instruction given: {prompt}\nCaregiver's reply: {response}\n"
            "Assess and give one sentence of feedback.",
        ),
    ]
)

INDEPENDENCE_ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an emotional-support assistant. First privately assess whether the patient accepted the "
            "offered task, then write one warm sentence affirming participation and effort. NEVER phrase a "
            "decline as a failure — a decline is simply logged, not corrected.",
        ),
        (
            "human",
            "Task offer given: {prompt}\nCaregiver's reply: {response}\n"
            "Assess and give one sentence of feedback.",
        ),
    ]
)

DEESCALATION_ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an emotional-outburst de-escalation coach. First privately assess whether the "
            "caregiver's reply indicates the patient has calmed, then write one short, calm sentence for the "
            "caregiver — either reinforcing what is working or gently pointing to the next redirection "
            "attempt. Never suggest arguing, correcting, or raising your voice.",
        ),
        (
            "human",
            "Coaching steps given: {prompt}\nCaregiver's reply: {response}\n"
            "Assess and give one sentence of feedback/next step.",
        ),
    ]
)

CAREGIVER_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Summarize this emotional-support session for a family caregiver in 2-3 sentences. Be factual "
            "and gentle, and note prevention-habit adherence or the outburst outcome, plus any newly "
            "identified trigger worth watching.",
        ),
        ("human", "Session data:\n{session_json}"),
    ]
)

URGENT_ALERT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write a short, calm but urgent alert for a family caregiver: a de-escalation attempt for an "
            "emotional outburst has not succeeded after multiple tries. State the severity, what has been "
            "tried, and that further support (a clinician or emergency contact) may be needed. 2 sentences max.",
        ),
        ("human", "Session data:\n{session_json}"),
    ]
)


class HabitAssessment(BaseModel):
    addressed: Optional[bool] = Field(
        description="Whether this prevention habit was carried out today; null if not assessable from the reply"
    )
    feedback: str = Field(
        description="One warm, brief sentence for the caregiver. Never phrases a pending habit as a failure."
    )


class IndependenceAssessment(BaseModel):
    accepted: Optional[bool] = Field(
        description="Whether the patient accepted the offered task; null if not assessable from the reply"
    )
    feedback: str = Field(description="One warm sentence affirming participation and effort, regardless of outcome")


class DeescalationAssessment(BaseModel):
    calmed: Optional[bool] = Field(
        description="Whether the caregiver's reply indicates the patient has calmed; null if not assessable"
    )
    feedback: str = Field(description="One short, calm sentence reinforcing progress or pointing to the next step")
