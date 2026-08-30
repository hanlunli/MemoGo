from __future__ import annotations

import random
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from agents.cognitive_brain_training.models import DiseaseStage as CognitiveDiseaseStage
from agents.cognitive_brain_training.models import PatientProfile as CognitivePatientProfile
from agents.cognitive_brain_training.modules import RealityOrientationEngine

from .models import AssistanceLevel, DiseaseStage, IncidentFlag, ModuleId, PatientProfile
from .prompts import (
    CAREGIVER_SUMMARY_PROMPT,
    HAZARD_ALERT_PROMPT,
    STEP_ASSESSMENT_PROMPT,
    STEP_INSTRUCTION_PROMPT,
    StepAssessment,
)

FRUSTRATION_KEYWORDS = (
    "too hard",
    "can't do this",
    "cannot do this",
    "frustrated",
    "stop",
    "leave me alone",
    "don't want",
    "no more",
    "hate this",
)

HAZARD_KEYWORDS: dict[IncidentFlag, tuple[str, ...]] = {
    IncidentFlag.BURN: ("burned", "burn", "scalded", "hot water spilled"),
    IncidentFlag.CUT: ("cut myself", "cut my", "bleeding"),
    IncidentFlag.FALL: ("fell", "fall", "on the floor", "on the ground"),
    IncidentFlag.DISTRESS: ("dizzy", "chest pain", "chest hurts", "can't breathe", "cannot breathe", "pain"),
}

ORIENTATION_STEP = "__orientation_check__"

PERSONAL_CARE_TASKS: dict[str, list[tuple[str, bool]]] = {
    "brushing teeth": [
        ("Pick up your toothbrush.", False),
        ("Put a small amount of toothpaste on the brush.", False),
        ("Brush your teeth gently.", False),
        ("Rinse your mouth with water.", False),
    ],
    "getting dressed": [
        ("Pick out a top to wear.", False),
        ("Put on the top.", False),
        ("Pick out a pair of pants.", False),
        ("Put on the pants.", False),
    ],
    "making a warm drink": [
        ("Fill the kettle with water.", True),
        ("Turn on the kettle to heat the water.", True),
        ("Put a tea bag in the cup.", False),
        ("Pour the hot water into the cup.", True),
    ],
}

HOUSEHOLD_CHORES_TASKS: dict[str, list[tuple[str, bool]]] = {
    "folding clothes": [
        ("Pick up one piece of clothing from the basket.", False),
        ("Fold it in half.", False),
        ("Place it on the stack.", False),
    ],
    "wiping the table": [
        ("Pick up the cloth.", False),
        ("Wipe the table from one side to the other.", False),
        ("Rinse the cloth in the sink.", False),
    ],
    "tidying the kitchen counter": [
        ("Pick up the items on the counter one at a time.", False),
        ("Put the knife back in the drawer.", True),
        ("Wipe the counter clean.", False),
    ],
}

TASK_LIBRARY: dict[ModuleId, dict[str, list[tuple[str, bool]]]] = {
    ModuleId.PERSONAL_CARE: PERSONAL_CARE_TASKS,
    ModuleId.HOUSEHOLD_CHORES: HOUSEHOLD_CHORES_TASKS,
}

ASSISTANCE_ORDER = (
    AssistanceLevel.INDEPENDENT,
    AssistanceLevel.VERBAL_CUE,
    AssistanceLevel.DEMONSTRATION,
    AssistanceLevel.PHYSICAL_ASSIST,
)

STAGE_BASELINE_ASSISTANCE = {
    DiseaseStage.MILD: AssistanceLevel.VERBAL_CUE,
    DiseaseStage.MODERATE: AssistanceLevel.DEMONSTRATION,
    DiseaseStage.SEVERE: AssistanceLevel.PHYSICAL_ASSIST,
}

STAGE_FLOOR_ASSISTANCE = {
    DiseaseStage.MILD: AssistanceLevel.INDEPENDENT,
    DiseaseStage.MODERATE: AssistanceLevel.VERBAL_CUE,
    DiseaseStage.SEVERE: AssistanceLevel.DEMONSTRATION,
}

_SCHEDULE_KEYWORD_TO_MODULE = {
    "household chores": ModuleId.HOUSEHOLD_CHORES,
    "chores": ModuleId.HOUSEHOLD_CHORES,
    "tidying": ModuleId.HOUSEHOLD_CHORES,
    "morning routine": ModuleId.PERSONAL_CARE,
    "hygiene": ModuleId.PERSONAL_CARE,
}


class StepInstructionEngine:
    def __init__(self, llm: BaseChatModel):
        self._chain = STEP_INSTRUCTION_PROMPT | llm

    def generate(
        self,
        patient: PatientProfile,
        task_name: str,
        step_description: str,
        assistance_level: AssistanceLevel,
        is_hazard_step: bool,
    ) -> str:
        result = self._chain.invoke(
            {
                "stage": patient.stage.value,
                "name": patient.name,
                "task_name": task_name,
                "step_description": step_description,
                "assistance_level": assistance_level.value,
                "hazard_note": (
                    "This step involves a potential hazard — stay close and supervise."
                    if is_hazard_step
                    else "none"
                ),
                "limitations": ", ".join(patient.physical_limitations) or "none reported",
            }
        )
        return result.content


class MorningOrientationOpener:
    """Opens the personal-care morning routine with a brief orientation check, sourced from the
    Cognitive & Brain Training Agent's reality-orientation-engine, per the cross-agent integration
    in the design spec."""

    def __init__(self, llm: BaseChatModel):
        self._engine = RealityOrientationEngine(llm)

    def generate(self, patient: PatientProfile) -> str:
        cognitive_patient = CognitivePatientProfile(
            patient_id=patient.patient_id,
            name=patient.name,
            stage=CognitiveDiseaseStage(patient.stage.value),
            preferences=list(patient.preferred_adl_tasks),
        )
        return self._engine.generate_prompt(cognitive_patient)


class FeedbackEncouragementLayer:
    def __init__(self, llm: BaseChatModel):
        self._chain = STEP_ASSESSMENT_PROMPT | llm.with_structured_output(StepAssessment)

    def generate(self, prompt: str, assistance_level: AssistanceLevel, response: str) -> StepAssessment:
        return self._chain.invoke(
            {"prompt": prompt, "assistance_level": assistance_level.value, "response": response}
        )


class CaregiverReporter:
    def __init__(self, llm: BaseChatModel):
        self._summary_chain = CAREGIVER_SUMMARY_PROMPT | llm
        self._alert_chain = HAZARD_ALERT_PROMPT | llm

    def summarize(self, session_json: str) -> str:
        result = self._summary_chain.invoke({"session_json": session_json})
        return result.content

    def hazard_alert(self, session_json: str) -> str:
        result = self._alert_chain.invoke({"session_json": session_json})
        return result.content


def resolve_module(schedule_activity: str) -> ModuleId:
    lowered = schedule_activity.lower()
    for keyword, module in _SCHEDULE_KEYWORD_TO_MODULE.items():
        if keyword in lowered:
            return module
    return ModuleId.PERSONAL_CARE


def select_task(module: ModuleId, patient: PatientProfile) -> str:
    tasks = TASK_LIBRARY[module]
    preferred = [t for t in patient.preferred_adl_tasks if t in tasks]
    return random.choice(preferred) if preferred else random.choice(list(tasks))


def build_step_sequence(module: ModuleId, task_name: str, schedule_activity: str) -> list[tuple[str, bool]]:
    steps = list(TASK_LIBRARY[module][task_name])
    if module == ModuleId.PERSONAL_CARE and "morning" in schedule_activity.lower():
        steps = [(ORIENTATION_STEP, False)] + steps
    return steps


def detect_hazard_incident(response_text: Optional[str]) -> IncidentFlag:
    if not response_text:
        return IncidentFlag.NONE
    lowered = response_text.lower()
    for flag, keywords in HAZARD_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return flag
    return IncidentFlag.NONE


def assess_frustration(response_text: Optional[str], response_latency_s: float) -> tuple[bool, float]:
    if not response_text:
        return True, 0.0
    lowered = response_text.lower()
    keyword_hit = any(keyword in lowered for keyword in FRUSTRATION_KEYWORDS)
    slow_response = response_latency_s > 45
    frustration_detected = keyword_hit or slow_response
    engagement_score = 0.3 if frustration_detected else min(1.0, 1.2 - response_latency_s / 60)
    return frustration_detected, round(max(0.0, engagement_score), 2)


def enforce_hazard_floor(level: AssistanceLevel, is_hazard_step: bool) -> AssistanceLevel:
    if is_hazard_step and ASSISTANCE_ORDER.index(level) < ASSISTANCE_ORDER.index(AssistanceLevel.DEMONSTRATION):
        return AssistanceLevel.DEMONSTRATION
    return level


def _escalate(level: AssistanceLevel) -> AssistanceLevel:
    idx = ASSISTANCE_ORDER.index(level)
    return ASSISTANCE_ORDER[min(idx + 1, len(ASSISTANCE_ORDER) - 1)]


def _de_escalate(level: AssistanceLevel, floor: AssistanceLevel) -> AssistanceLevel:
    idx = ASSISTANCE_ORDER.index(level)
    floor_idx = ASSISTANCE_ORDER.index(floor)
    return ASSISTANCE_ORDER[max(idx - 1, floor_idx)]


def adjust_assistance_level(
    level: AssistanceLevel,
    floor: AssistanceLevel,
    completed: Optional[bool],
    frustration_detected: bool,
    incident_flag: IncidentFlag,
) -> AssistanceLevel:
    if incident_flag is not IncidentFlag.NONE:
        return level
    if frustration_detected:
        return _escalate(level)
    if completed is True:
        return _de_escalate(level, floor)
    if completed is False:
        return _escalate(level)
    return level


def is_safety_stop(incident_flag: IncidentFlag) -> bool:
    return incident_flag is not IncidentFlag.NONE
