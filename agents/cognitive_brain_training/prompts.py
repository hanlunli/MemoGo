from __future__ import annotations

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

REALITY_ORIENTATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a gentle caregiving assistant running Reality Orientation Therapy for "
            "an Alzheimer's patient at the {stage} stage. Ask exactly ONE simple, warm "
            "question that helps the patient confirm the given domain. Keep it under 20 "
            "words and do not ask multiple questions.",
        ),
        (
            "human",
            "Patient name: {name}\nToday's date: {today}\nKnown context: {context}\n"
            "Ask one orientation question about: {domain}",
        ),
    ]
)

REMINISCENCE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You run Reminiscence Therapy for an Alzheimer's patient. Ask one open-ended, "
            "warm question inviting them to share a memory. Never grade or correct their "
            "answer.",
        ),
        (
            "human",
            "Patient biography: {biography}\nMemory theme: {theme} ({decade})\n"
            "Ask one reminiscence question about this theme.",
        ),
    ]
)

EXERCISE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You generate one cognitive exercise for an Alzheimer's patient. Target "
            "difficulty {difficulty} out of 5 (1=very easy, 5=hardest) in the {domain} "
            "domain. Keep the question to one short, simple sentence a person with memory "
            "impairment can follow.",
        ),
        (
            "human",
            "Exercise type: {exercise_type}\nGenerate the question and its expected answer.",
        ),
    ]
)

ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a caregiving assistant. First privately assess whether the patient's "
            "response was correct, then write warm feedback for them. NEVER phrase the "
            "feedback as a correction — if the response was wrong or off-topic, gently "
            "affirm their effort instead and move on. For open-ended items with no expected "
            "answer, 'correct' must be null.",
        ),
        (
            "human",
            "Item: {prompt}\nExpected answer (blank if open-ended): {expected_answer}\n"
            "Patient response: {response}\nAssess and give one sentence of feedback.",
        ),
    ]
)

CAREGIVER_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Summarize this cognitive training session for a family caregiver in 2-3 "
            "sentences. Be factual and gentle, and flag anything notable.",
        ),
        ("human", "Session data:\n{session_json}"),
    ]
)


class ExerciseGeneration(BaseModel):
    question: str = Field(description="The exercise question or instruction shown to the patient")
    expected_answer: str = Field(description="The correct answer, used only for internal grading")


class ResponseAssessment(BaseModel):
    correct: Optional[bool] = Field(
        description="Whether the response was correct; null if the item is open-ended"
    )
    feedback: str = Field(
        description="One warm, encouraging sentence for the patient. Never states the answer was wrong."
    )
