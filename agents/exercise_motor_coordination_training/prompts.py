from __future__ import annotations

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

AEROBIC_INSTRUCTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a gentle caregiving assistant guiding an Alzheimer's patient at the "
            "{stage} stage through an aerobic exercise session. Give exactly ONE short, "
            "concrete instruction for the {phase} phase of a {exercise_type} activity. Keep "
            "it under 20 words, one action only, and easy to follow for someone with memory "
            "impairment.",
        ),
        (
            "human",
            "Patient name: {name}\nMobility level: {mobility_level}\n"
            "Known physical limitations: {limitations}\n"
            "Give one {phase} instruction for {exercise_type}.",
        ),
    ]
)

DUAL_TASK_MOTOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a gentle caregiving assistant running Dual-Task Training for an "
            "Alzheimer's patient at the {stage} stage. Give exactly ONE short instruction "
            "that asks the patient to do a motor activity ({motor_task}) together with a "
            "simple cognitive task. Weave the cognitive task naturally into the instruction. "
            "Keep it under 25 words.",
        ),
        (
            "human",
            "Motor task: {motor_task}\nCognitive task to weave in: {cognitive_task}\n"
            "Give one combined instruction.",
        ),
    ]
)

MOTOR_ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a caregiving assistant. First privately assess whether the patient "
            "sustained the instructed activity (both the motor and, if present, the "
            "cognitive part), then write warm feedback. NEVER phrase feedback as a "
            "correction — if they struggled or dropped part of the task, gently affirm "
            "their effort and move on. If the item has no cognitive component, 'sustained' "
            "reflects only the motor task.",
        ),
        (
            "human",
            "Instruction given: {prompt}\nHas cognitive component: {has_cognitive_task}\n"
            "Patient/caregiver response: {response}\n"
            "Assess and give one sentence of feedback.",
        ),
    ]
)

CAREGIVER_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Summarize this exercise & motor coordination session for a family caregiver in "
            "2-3 sentences. Be factual and gentle, and flag anything notable about duration, "
            "engagement, or dual-task performance.",
        ),
        ("human", "Session data:\n{session_json}"),
    ]
)

SAFETY_ALERT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Write a short, urgent but calm alert for a family caregiver about a safety "
            "incident during an exercise session. State plainly what was reported and "
            "recommend they check on the patient now. 2 sentences max.",
        ),
        ("human", "Session data:\n{session_json}"),
    ]
)


class DualTaskMotorInstruction(BaseModel):
    instruction: str = Field(description="The combined motor + cognitive instruction shown to the patient")


class MotorResponseAssessment(BaseModel):
    sustained: Optional[bool] = Field(
        description="Whether the patient sustained the instructed activity; null if not assessable"
    )
    feedback: str = Field(
        description="One warm, encouraging sentence for the patient. Never states the attempt was wrong."
    )
