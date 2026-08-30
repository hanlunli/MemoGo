from __future__ import annotations

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from agents.adl_training.modules import TASK_ROOM_LABELS as ADL_TASK_ROOM_LABELS
from agents.cognitive_brain_training.models import DiseaseStage as CognitiveDiseaseStage
from agents.cognitive_brain_training.models import PatientProfile as CognitivePatientProfile
from agents.cognitive_brain_training.modules import RealityOrientationEngine

from .models import (
    CheckpointStatus,
    CheckpointType,
    DeviationEvent,
    DeviationSeverity,
    DeviationType,
    PatientProfile,
    RoutineCheckpoint,
)
from .prompts import (
    CAREGIVER_SUMMARY_PROMPT,
    CHECKPOINT_ASSESSMENT_PROMPT,
    CHECKPOINT_DELIVERY_PROMPT,
    DEVIATION_ASSESSMENT_PROMPT,
    DEVIATION_RESPONSE_PROMPT,
    ROUTINE_ALERT_PROMPT,
    CheckpointAssessment,
    DeviationAssessment,
)

_SCHEDULE_KEYWORD_TO_CHECKPOINTS: dict[str, list[CheckpointType]] = {
    "morning routine": [CheckpointType.WAKE_UP, CheckpointType.DRESSING],
    "hygiene": [CheckpointType.DRESSING],
    "breakfast": [CheckpointType.MEAL],
    "lunch": [CheckpointType.MEAL],
    "midday rest": [CheckpointType.MEAL],
    "dinner": [CheckpointType.MEAL],
    "medication": [CheckpointType.MEDICATION],
    "walk": [CheckpointType.WALK],
    "bedtime": [CheckpointType.BEDTIME],
}

_DRINK_CHOICE_OPTIONS = ["water", "tea"]

# The room a checkpoint sends the patient to, keyed to the ADL agent's own task-room mapping so
# wayfinding reuses the existing door label instead of a separately tracked room list, per the
# cross-agent integration in the design spec.
_CHECKPOINT_ADL_TASK_KEY: dict[CheckpointType, str] = {
    CheckpointType.DRESSING: "getting dressed",
    CheckpointType.MEAL: "making a warm drink",
}

# A caregiver-confirmed dosing-supervision event is safety-critical and must never be inferred by
# the model from an ambiguous reply — mirrors the Home Safety agent's deterministic incident-
# acknowledgment routing.
SUPERVISION_KEYWORDS = (
    "supervised",
    "watched",
    "gave the pill",
    "gave the medication",
    "administered",
    "witnessed",
    "took it in front of me",
    "confirmed the dose",
)

DOUBLE_MEAL_KEYWORDS = (
    "already ate",
    "already had",
    "second helping",
    "asking for food again",
    "wants to eat again",
    "just ate",
)

DEVIATION_ACK_KEYWORDS = (
    "adjusted",
    "rescheduled",
    "back on schedule",
    "handled",
    "addressed",
    "resolved",
    "confirmed",
)

DEVIATION_NO_ACK_KEYWORDS = (
    "can't",
    "cannot",
    "don't know",
    "not sure",
    "unable",
    "no idea",
)

ACK_TIMEOUT_S = 90


def resolve_checkpoints(schedule_activity: str) -> list[CheckpointType]:
    lowered = schedule_activity.lower()
    matched: list[CheckpointType] = []
    for keyword, checkpoints in _SCHEDULE_KEYWORD_TO_CHECKPOINTS.items():
        if keyword in lowered:
            for checkpoint_type in checkpoints:
                if checkpoint_type not in matched:
                    matched.append(checkpoint_type)
    return matched or [CheckpointType.WAKE_UP]


def room_label_for_checkpoint(checkpoint_type: CheckpointType) -> Optional[str]:
    """Reuses the ADL agent's environmental cue map so a checkpoint that sends the patient to a
    room (dressing, meal) references the same door label instead of a separately tracked one."""
    task_key = _CHECKPOINT_ADL_TASK_KEY.get(checkpoint_type)
    if task_key is None:
        return None
    return ADL_TASK_ROOM_LABELS.get(task_key)


def build_checkpoint_sequence(
    patient: PatientProfile, checkpoint_types: list[CheckpointType]
) -> list[RoutineCheckpoint]:
    checkpoints: list[RoutineCheckpoint] = []
    for checkpoint_type in checkpoint_types:
        choice_options: list[str] = []
        if checkpoint_type == CheckpointType.DRESSING:
            # Never expose the full closet — at most two stage-appropriate outfits per session.
            choice_options = patient.seasonal_outfit_set[:2]
        elif checkpoint_type == CheckpointType.MEAL:
            choice_options = list(_DRINK_CHOICE_OPTIONS)
        elif checkpoint_type == CheckpointType.MEDICATION:
            choice_options = list(patient.medication_names)
        checkpoints.append(RoutineCheckpoint(checkpoint_type=checkpoint_type, choice_options=choice_options))
    return checkpoints


def classify_deviation_severity(deviation_type: DeviationType) -> DeviationSeverity:
    """An unplanned environment change is always treated as a heightened-attention window; a
    schedule slip on its own is moderate until the caregiver's reply says otherwise."""
    if deviation_type == DeviationType.ENVIRONMENT_CHANGE:
        return DeviationSeverity.MAJOR
    return DeviationSeverity.MODERATE


def resolve_checkpoint_status(resolved: Optional[bool]) -> CheckpointStatus:
    if resolved is True:
        return CheckpointStatus.COMPLETED
    if resolved is False:
        return CheckpointStatus.NEEDS_ATTENTION
    return CheckpointStatus.PENDING


def assess_medication_supervision(response_text: Optional[str]) -> bool:
    """Deterministic, not LLM-judged: a dose is only ever marked supervised on an explicit
    caregiver confirmation phrase — an ambiguous or absent reply must never be read as taken."""
    if not response_text:
        return False
    lowered = response_text.lower()
    return any(keyword in lowered for keyword in SUPERVISION_KEYWORDS)


def detect_double_meal_request(response_text: Optional[str]) -> bool:
    if not response_text:
        return False
    lowered = response_text.lower()
    return any(keyword in lowered for keyword in DOUBLE_MEAL_KEYWORDS)


def assess_deviation_acknowledgment(response_text: Optional[str], response_latency_s: float) -> tuple[bool, bool]:
    """Returns (acknowledged, needs_caregiver_followup).

    Deterministic, not LLM-judged, mirroring the Home Safety agent's incident-acknowledgment
    routing — a routine deviation must never be auto-resolved on a vague or absent reply.
    """
    if not response_text:
        return False, True
    if response_latency_s > ACK_TIMEOUT_S:
        return False, True
    lowered = response_text.lower()
    if any(keyword in lowered for keyword in DEVIATION_NO_ACK_KEYWORDS):
        return False, True
    acknowledged = any(keyword in lowered for keyword in DEVIATION_ACK_KEYWORDS)
    return acknowledged, not acknowledged


class WakeUpOrientationOpener:
    """Opens the wake-up routine checkpoint with a brief orientation check, sourced from the
    Cognitive & Brain Training Agent's reality-orientation-engine, per the cross-agent integration
    in the design spec."""

    def __init__(self, llm: BaseChatModel):
        self._engine = RealityOrientationEngine(llm)

    def generate(self, patient: PatientProfile) -> str:
        cognitive_patient = CognitivePatientProfile(
            patient_id=patient.patient_id,
            name=patient.name,
            stage=CognitiveDiseaseStage(patient.stage.value),
            preferences=[],
        )
        return self._engine.generate_prompt(cognitive_patient)


class CheckpointDeliveryEngine:
    def __init__(self, llm: BaseChatModel):
        self._chain = CHECKPOINT_DELIVERY_PROMPT | llm

    def generate(self, patient: PatientProfile, checkpoint: RoutineCheckpoint) -> str:
        room_label = room_label_for_checkpoint(checkpoint.checkpoint_type)
        room_note = f"Guide them to the door labeled '{room_label}'." if room_label else "none"
        choice_note = (
            f"Offer exactly these options as a choice: {', '.join(checkpoint.choice_options)}."
            if checkpoint.choice_options
            else "none - this checkpoint has no choice to offer."
        )
        result = self._chain.invoke(
            {
                "stage": patient.stage.value,
                "checkpoint_type": checkpoint.checkpoint_type.value,
                "room_note": room_note,
                "choice_note": choice_note,
                "dietary_restrictions": ", ".join(patient.dietary_restrictions) or "none reported",
            }
        )
        return result.content


class DeviationResponseEngine:
    def __init__(self, llm: BaseChatModel):
        self._chain = DEVIATION_RESPONSE_PROMPT | llm

    def generate(self, event: DeviationEvent, severity: DeviationSeverity) -> str:
        result = self._chain.invoke(
            {
                "deviation_type": event.deviation_type.value,
                "checkpoint_type": event.checkpoint_type.value,
                "description": event.description or "none provided",
                "severity": severity.value,
            }
        )
        return result.content


class CheckpointAssessmentLayer:
    def __init__(self, llm: BaseChatModel):
        self._chain = CHECKPOINT_ASSESSMENT_PROMPT | llm.with_structured_output(CheckpointAssessment)

    def generate(self, prompt: str, response: str) -> CheckpointAssessment:
        return self._chain.invoke({"prompt": prompt, "response": response})


class DeviationAssessmentLayer:
    def __init__(self, llm: BaseChatModel):
        self._chain = DEVIATION_ASSESSMENT_PROMPT | llm.with_structured_output(DeviationAssessment)

    def generate(self, prompt: str, response: str) -> DeviationAssessment:
        return self._chain.invoke({"prompt": prompt, "response": response})


class CaregiverReporter:
    def __init__(self, llm: BaseChatModel):
        self._summary_chain = CAREGIVER_SUMMARY_PROMPT | llm
        self._alert_chain = ROUTINE_ALERT_PROMPT | llm

    def summarize(self, session_json: str) -> str:
        result = self._summary_chain.invoke({"session_json": session_json})
        return result.content

    def routine_alert(self, session_json: str) -> str:
        result = self._alert_chain.invoke({"session_json": session_json})
        return result.content
