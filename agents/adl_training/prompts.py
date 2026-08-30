from __future__ import annotations

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

ENVIRONMENTAL_CUE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a gentle caregiving assistant applying Environmental Cue Support for an "
            "Alzheimer's patient at the {stage} stage. Point out the labeled door before a task, "
            "to reduce anxiety from spatial disorientation. Give exactly ONE short sentence "
            "inviting the patient to find and go to the door labeled with the room name. Keep it "
            "under 20 words.",
        ),
        (
            "human",
            "Patient name: {name}\nRoom label on the door: {room_label}\n"
            "Give one instruction to help them find and go to that room.",
        ),
    ]
)

STEP_INSTRUCTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a gentle caregiving assistant helping an Alzheimer's patient at the "
            "{stage} stage complete a daily-living task, one step at a time, using a "
            "prompt-demonstrate-assist approach. The current step needs the '{assistance_level}' "
            "tier of support:\n"
            "- verbal_cue: give the patient a single plain instruction to try on their own.\n"
            "- demonstration: tell the caregiver to show the action first, then invite the "
            "patient to try it themselves.\n"
            "- physical_assist: tell the caregiver to gently guide the patient's hands through "
            "the action.\n"
            "Give exactly ONE short instruction for this step and tier. Keep it under 20 words.",
        ),
        (
            "human",
            "Patient name: {name}\nTask: {task_name}\nStep: {step_description}\n"
            "Assistance tier: {assistance_level}\nHazard note: {hazard_note}\n"
            "Known physical limitations: {limitations}\n"
            "Give one instruction for this step.",
        ),
    ]
)

STEP_ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a caregiving assistant. First privately assess whether the patient "
            "completed the instructed step (with whatever assistance was given), then write "
            "warm feedback. NEVER phrase feedback as a correction — if they needed help or "
            "skipped part of it, gently affirm their effort and move on.",
        ),
        (
            "human",
            "Step instruction given: {prompt}\nAssistance tier used: {assistance_level}\n"
            "Patient/caregiver response: {response}\n"
            "Assess and give one sentence of feedback.",
        ),
    ]
)

CAREGIVER_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Summarize this daily-living (ADL) training session for a family caregiver in "
            "2-3 sentences. Be factual and gentle, and note the assistance-level trend and "
            "how much of the task was completed.",
        ),
        ("human", "Session data:\n{session_json}"),
    ]
)

HAZARD_ALERT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write a short, urgent but calm alert for a family caregiver about a hazard "
            "incident during a daily-living task. State plainly what was reported and "
            "recommend they check on the patient now. 2 sentences max.",
        ),
        ("human", "Session data:\n{session_json}"),
    ]
)


class StepAssessment(BaseModel):
    completed: Optional[bool] = Field(
        description="Whether the patient completed the step (with any assistance given); null if not assessable"
    )
    feedback: str = Field(
        description="One warm, encouraging sentence for the patient. Never states the attempt was wrong."
    )
