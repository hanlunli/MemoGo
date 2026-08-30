from __future__ import annotations

import random
from datetime import date
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from .models import DiseaseStage, ExerciseItem, MemoryItem, ModuleId, PatientProfile, ProgressTrend, SessionLog
from .prompts import (
    ASSESSMENT_PROMPT,
    CAREGIVER_SUMMARY_PROMPT,
    EXERCISE_PROMPT,
    REALITY_ORIENTATION_PROMPT,
    REMINISCENCE_PROMPT,
    ExerciseGeneration,
    ResponseAssessment,
)

FATIGUE_KEYWORDS = ("tired", "stop", "enough", "no more", "don't want", "leave me")

ORIENTATION_DOMAINS = (
    "today's date",
    "their current location",
    "a family member's identity",
)

STAGE_DIFFICULTY_CEILING = {
    DiseaseStage.MILD: 5,
    DiseaseStage.MODERATE: 3,
    DiseaseStage.SEVERE: 1,
}

_SCHEDULE_KEYWORD_TO_MODULE = {
    "reminiscence": ModuleId.REMINISCENCE,
    "reality orientation": ModuleId.REALITY_ORIENTATION,
    "cognitive training": ModuleId.EXERCISE,
    "targeted cognitive": ModuleId.EXERCISE,
}


class RealityOrientationEngine:
    def __init__(self, llm: BaseChatModel):
        self._chain = REALITY_ORIENTATION_PROMPT | llm

    def generate_prompt(self, patient: PatientProfile) -> tuple[str, str]:
        """Returns (prompt_text, orientation_domain) so the caller can log which domain
        (date/location/person) this check actually tested."""
        domain = random.choice(ORIENTATION_DOMAINS)
        result = self._chain.invoke(
            {
                "stage": patient.stage.value,
                "domain": domain,
                "name": patient.name,
                "today": date.today().isoformat(),
                "context": ", ".join(patient.preferences) or "none provided",
            }
        )
        return result.content, domain


class ReminiscenceSessionManager:
    def __init__(self, llm: BaseChatModel):
        self._chain = REMINISCENCE_PROMPT | llm

    def generate_prompt(self, patient: PatientProfile, memory: MemoryItem) -> str:
        result = self._chain.invoke(
            {
                "biography": patient.biography or "not provided",
                "theme": memory.theme,
                "decade": memory.decade,
            }
        )
        return result.content


class CognitiveExerciseGenerator:
    def __init__(self, llm: BaseChatModel):
        self._chain = EXERCISE_PROMPT | llm.with_structured_output(ExerciseGeneration)

    def generate(self, domain: str, difficulty: int, exercise_type: str) -> ExerciseItem:
        result: ExerciseGeneration = self._chain.invoke(
            {"difficulty": difficulty, "domain": domain, "exercise_type": exercise_type}
        )
        return ExerciseItem(
            type=exercise_type,
            domain=domain,
            difficulty=difficulty,
            content=result.question,
            answer=result.expected_answer,
        )


class FeedbackEncouragementLayer:
    def __init__(self, llm: BaseChatModel):
        self._chain = ASSESSMENT_PROMPT | llm.with_structured_output(ResponseAssessment)

    def generate(self, prompt: str, expected_answer: Optional[str], response: str) -> ResponseAssessment:
        return self._chain.invoke(
            {"prompt": prompt, "expected_answer": expected_answer or "", "response": response}
        )


class CaregiverReporter:
    def __init__(self, llm: BaseChatModel):
        self._chain = CAREGIVER_SUMMARY_PROMPT | llm

    def summarize(self, session_json: str) -> str:
        result = self._chain.invoke({"session_json": session_json})
        return result.content


def resolve_module(schedule_activity: str) -> ModuleId:
    lowered = schedule_activity.lower()
    for keyword, module in _SCHEDULE_KEYWORD_TO_MODULE.items():
        if keyword in lowered:
            return module
    return ModuleId.EXERCISE


def adjust_difficulty(
    difficulty: int, ceiling: int, correct: Optional[bool], fatigue_detected: bool
) -> int:
    if fatigue_detected:
        return max(1, difficulty - 1)
    if correct is True:
        return min(ceiling, difficulty + 1)
    if correct is False:
        return max(1, difficulty - 1)
    return difficulty


def build_progress_trends(session_log: SessionLog) -> list[ProgressTrend]:
    """Summarizes this session's turns per cognitive domain (memory/attention/executive
    function/language/orientation), since the session log is the only history this agent has
    access to — there's no cross-session store to compute a true longitudinal trend from."""
    turns_by_domain: dict[str, list] = {}
    for turn in session_log.turns:
        if turn.domain:
            turns_by_domain.setdefault(turn.domain, []).append(turn)

    trends = []
    for domain, turns in turns_by_domain.items():
        assessed = [turn.correct for turn in turns if turn.correct is not None]
        accuracy = round(sum(1 for c in assessed if c) / len(assessed), 2) if assessed else 0.0
        trends.append(
            ProgressTrend(
                domain=domain,
                rolling_accuracy=accuracy,
                rolling_engagement=session_log.engagement_score,
                timeframe="this_session",
            )
        )
    return trends


def assess_engagement(response_text: Optional[str], response_latency_s: float) -> tuple[bool, float]:
    if not response_text:
        return True, 0.0
    lowered = response_text.lower()
    keyword_hit = any(keyword in lowered for keyword in FATIGUE_KEYWORDS)
    slow_response = response_latency_s > 45
    fatigue_detected = keyword_hit or slow_response
    engagement_score = 0.3 if fatigue_detected else min(1.0, 1.2 - response_latency_s / 60)
    return fatigue_detected, round(max(0.0, engagement_score), 2)
