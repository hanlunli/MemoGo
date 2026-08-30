from __future__ import annotations

import random
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from agents.cognitive_brain_training.modules import CognitiveExerciseGenerator

from .models import DiseaseStage, IncidentFlag, ModuleId, PatientProfile
from .prompts import (
    AEROBIC_INSTRUCTION_PROMPT,
    CAREGIVER_SUMMARY_PROMPT,
    DUAL_TASK_MOTOR_PROMPT,
    MOTOR_ASSESSMENT_PROMPT,
    SAFETY_ALERT_PROMPT,
    DualTaskMotorInstruction,
    MotorResponseAssessment,
)

FATIGUE_KEYWORDS = ("tired", "stop", "enough", "no more", "don't want", "leave me", "out of breath")
# NEAR_FALL is checked before FALL: its own keywords ("almost fell", "nearly fell") contain "fell"
# as a substring, so checking FALL first would misclassify every recovered near-fall as a real fall.
SAFETY_KEYWORDS: dict[IncidentFlag, tuple[str, ...]] = {
    IncidentFlag.NEAR_FALL: ("almost fell", "stumbled", "lost balance", "nearly fell"),
    IncidentFlag.FALL: ("fell", "fall", "on the floor", "on the ground"),
    IncidentFlag.DISTRESS: ("dizzy", "chest pain", "chest hurts", "can't breathe", "cannot breathe", "in pain"),
}

AEROBIC_EXERCISE_TYPES = ("walking", "tai chi", "baduanjin", "square dancing")
MOTOR_TASKS = ("walking slowly", "marching in place", "bouncing a ball")
COGNITIVE_OVERLAY_DOMAINS = ("counting", "attention", "memory")
COGNITIVE_OVERLAY_EXERCISE_TYPES = {
    "counting": "arithmetic",
    "attention": "picture recognition",
    "memory": "word association",
}

STAGE_INTENSITY_CEILING = {
    DiseaseStage.MILD: 5,
    DiseaseStage.MODERATE: 3,
    DiseaseStage.SEVERE: 1,
}

_SCHEDULE_KEYWORD_TO_MODULE = {
    "dual-task": ModuleId.DUAL_TASK,
    "dual task": ModuleId.DUAL_TASK,
    "aerobic": ModuleId.AEROBIC,
    "outdoor exercise": ModuleId.AEROBIC,
}


class AerobicExerciseSessionEngine:
    def __init__(self, llm: BaseChatModel):
        self._chain = AEROBIC_INSTRUCTION_PROMPT | llm

    def generate_instruction(self, patient: PatientProfile, exercise_type: str, phase: str, intensity: int) -> str:
        result = self._chain.invoke(
            {
                "stage": patient.stage.value,
                "name": patient.name,
                "mobility_level": patient.mobility_level.value,
                "limitations": ", ".join(patient.physical_limitations) or "none reported",
                "exercise_type": exercise_type,
                "phase": phase,
                "intensity": intensity,
            }
        )
        return result.content


class DualTaskTrainingCoordinator:
    """Pairs a motor task with a cognitive micro-task sourced from the Cognitive & Brain
    Training Agent's exercise generator, per the cross-agent integration in the design spec."""

    def __init__(self, llm: BaseChatModel):
        self._chain = DUAL_TASK_MOTOR_PROMPT | llm.with_structured_output(DualTaskMotorInstruction)
        self._cognitive_exercise_generator = CognitiveExerciseGenerator(llm)

    def generate_instruction(
        self, patient: PatientProfile, motor_task: str, include_cognitive_task: bool, intensity: int
    ) -> tuple[str, Optional[str]]:
        cognitive_task: Optional[str] = None
        if include_cognitive_task:
            domain = random.choice(COGNITIVE_OVERLAY_DOMAINS)
            exercise = self._cognitive_exercise_generator.generate(
                domain=domain, difficulty=1, exercise_type=COGNITIVE_OVERLAY_EXERCISE_TYPES[domain]
            )
            cognitive_task = exercise.content

        result: DualTaskMotorInstruction = self._chain.invoke(
            {
                "stage": patient.stage.value,
                "motor_task": motor_task,
                "cognitive_task": cognitive_task or "none — motor task only",
                "intensity": intensity,
            }
        )
        return result.instruction, cognitive_task


class FeedbackEncouragementLayer:
    def __init__(self, llm: BaseChatModel):
        self._chain = MOTOR_ASSESSMENT_PROMPT | llm.with_structured_output(MotorResponseAssessment)

    def generate(self, prompt: str, has_cognitive_task: bool, response: str) -> MotorResponseAssessment:
        return self._chain.invoke(
            {"prompt": prompt, "has_cognitive_task": has_cognitive_task, "response": response}
        )


class CaregiverReporter:
    def __init__(self, llm: BaseChatModel):
        self._summary_chain = CAREGIVER_SUMMARY_PROMPT | llm
        self._alert_chain = SAFETY_ALERT_PROMPT | llm

    def summarize(self, session_json: str) -> str:
        result = self._summary_chain.invoke({"session_json": session_json})
        return result.content

    def safety_alert(self, session_json: str) -> str:
        result = self._alert_chain.invoke({"session_json": session_json})
        return result.content


def resolve_module(schedule_activity: str) -> ModuleId:
    lowered = schedule_activity.lower()
    for keyword, module in _SCHEDULE_KEYWORD_TO_MODULE.items():
        if keyword in lowered:
            return module
    return ModuleId.AEROBIC


def detect_safety_incident(response_text: Optional[str]) -> IncidentFlag:
    if not response_text:
        return IncidentFlag.NONE
    lowered = response_text.lower()
    for flag, keywords in SAFETY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return flag
    return IncidentFlag.NONE


def assess_exertion(response_text: Optional[str], response_latency_s: float) -> tuple[bool, float]:
    if not response_text:
        return True, 0.0
    lowered = response_text.lower()
    keyword_hit = any(keyword in lowered for keyword in FATIGUE_KEYWORDS)
    slow_response = response_latency_s > 45
    fatigue_detected = keyword_hit or slow_response
    engagement_score = 0.3 if fatigue_detected else min(1.0, 1.2 - response_latency_s / 60)
    return fatigue_detected, round(max(0.0, engagement_score), 2)


def adjust_intensity(
    intensity: int,
    ceiling: int,
    sustained: Optional[bool],
    fatigue_detected: bool,
    incident_flag: IncidentFlag,
) -> int:
    if incident_flag is not IncidentFlag.NONE:
        return 1
    if fatigue_detected:
        return max(1, intensity - 1)
    if sustained is True:
        return min(ceiling, intensity + 1)
    if sustained is False:
        return max(1, intensity - 1)
    return intensity


def is_safety_stop(incident_flag: IncidentFlag) -> bool:
    return incident_flag is not IncidentFlag.NONE
