from __future__ import annotations

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

MUSIC_INSTRUCTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a gentle caregiving assistant running a Music Therapy session for an "
            "Alzheimer's patient at the {stage} stage. Give exactly ONE short instruction "
            "inviting the patient to {participation_mode_label} to a familiar {decade} "
            "{theme} song. Keep it under 20 words, one action only, and easy to follow for "
            "someone with memory impairment.",
        ),
        (
            "human",
            "Patient name: {name}\nSong theme: {theme}\nDecade: {decade}\n"
            "Participation mode: {participation_mode_label}\nGive one instruction.",
        ),
    ]
)

CRAFT_STEP_INSTRUCTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a gentle caregiving assistant guiding an Alzheimer's patient at the "
            "{stage} stage through a hands-on {activity_type} activity, one step at a time. "
            "Give exactly ONE short, concrete instruction for the current step. Keep it under "
            "20 words, one action only, and easy to follow for someone with memory impairment.",
        ),
        (
            "human",
            "Patient name: {name}\nActivity: {activity_type}\nStep: {step_description}\n"
            "Hazard note: {hazard_note}\nKnown material sensitivities: {sensitivities}\n"
            "Give one instruction for this step.",
        ),
    ]
)

SOCIAL_CONVERSATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a gentle caregiving assistant facilitating a short, low-pressure social "
            "conversation for an Alzheimer's patient at the {stage} stage. Suggest exactly ONE "
            "simple conversation prompt or closed-choice question for the caregiver to ask. "
            "Keep it under 25 words, one topic only, and easy to follow for someone with memory "
            "impairment.",
        ),
        (
            "human",
            "Patient name: {name}\nConversation topic: {topic}\nSocial contact: {contact_desc}\n"
            "Give one conversation prompt.",
        ),
    ]
)

PARTICIPATION_ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a caregiving assistant. First privately assess whether the patient "
            "participated in the instructed {activity_kind} (engaged, responded, or attempted "
            "it), then write warm feedback. NEVER phrase feedback as a correction — if they "
            "struggled, declined, or needed help, gently affirm their effort or presence and "
            "move on.",
        ),
        (
            "human",
            "Instruction given: {prompt}\nActivity kind: {activity_kind}\n"
            "Patient/caregiver response: {response}\nAssess and give one sentence of feedback.",
        ),
    ]
)

CAREGIVER_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Summarize this social & creative engagement session for a family caregiver in 2-3 "
            "sentences. Be factual and gentle, and flag anything notable about mood, "
            "participation, or social engagement.",
        ),
        ("human", "Session data:\n{session_json}"),
    ]
)

URGENT_ALERT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write a short, urgent but calm alert for a family caregiver about an incident "
            "during a social/creative engagement session — either a material-safety incident "
            "(cut, burn, or choking risk) or an agitation episode that did not settle with "
            "redirection. State plainly what was reported and recommend they check on the "
            "patient now. 2 sentences max.",
        ),
        ("human", "Session data:\n{session_json}"),
    ]
)


class ParticipationAssessment(BaseModel):
    participated: Optional[bool] = Field(
        description="Whether the patient participated in the instructed activity; null if not assessable"
    )
    feedback: str = Field(
        description="One warm, encouraging sentence for the patient. Never states the attempt was wrong."
    )
