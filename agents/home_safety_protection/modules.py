from __future__ import annotations

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from agents.adl_training.modules import TASK_ROOM_LABELS as ADL_TASK_ROOM_LABELS

from .models import ChecklistItem, ChecklistStatus, HazardType, IncidentEvent, ModuleId, PatientProfile, Severity, WanderingRiskLevel
from .prompts import (
    CAREGIVER_SUMMARY_PROMPT,
    CHECKLIST_ITEM_PROMPT,
    EMERGENCY_ALERT_PROMPT,
    INCIDENT_ASSESSMENT_PROMPT,
    INCIDENT_TRIAGE_PROMPT,
    ITEM_ASSESSMENT_PROMPT,
    IncidentAssessment,
    ItemAssessment,
)

# Every checklist item below carries a hazard type so priority ordering and future risk
# calibration can key off the same taxonomy used for real-time incidents.
HOME_CHECKLIST_TEMPLATES: dict[str, list[tuple[HazardType, str, bool]]] = {
    "Kitchen": [
        (HazardType.FIRE_GAS, "Install an automatic gas shut-off valve on the stove.", True),
        (HazardType.FIRE_GAS, "Mount a smoke detector above the cooking area and test it.", True),
        (HazardType.MEDICATION_CHEMICAL_ACCESS, "Store chemical cleaners and sharp knives in a locked cabinet.", True),
        (HazardType.FALL, "Keep the kitchen floor dry, clutter-free, and well lit.", False),
    ],
    "Bathroom": [
        (HazardType.FALL, "Install grab bars near the toilet and shower.", True),
        (HazardType.FALL, "Place a non-slip mat in the shower or tub.", True),
        (HazardType.FALL, "Keep a nightlight on in the bathroom overnight.", False),
    ],
    "Bedroom": [
        (HazardType.MEDICATION_CHEMICAL_ACCESS, "Lock all medications in a caregiver-controlled cabinet.", True),
        (HazardType.FALL, "Keep a nightlight on and a clear path from the bed to the bathroom.", False),
    ],
    "Hallway": [
        (HazardType.WANDERING, "Install a door alarm or concealed lock on the exit door.", True),
        (HazardType.WANDERING, "Hang a wall-matching curtain over the exit door to reduce the urge to leave.", False),
        (HazardType.FALL, "Keep the hallway floor clutter-free and the nightlight on.", False),
    ],
}

_SCHEDULE_KEYWORD_TO_MODULE = {
    "home safety": ModuleId.HOME_SAFETY_AUDIT,
    "safety walkthrough": ModuleId.HOME_SAFETY_AUDIT,
    "safety audit": ModuleId.HOME_SAFETY_AUDIT,
    "safety check": ModuleId.HOME_SAFETY_AUDIT,
}

_RISK_ORDER = (WanderingRiskLevel.LOW, WanderingRiskLevel.MODERATE, WanderingRiskLevel.HIGH)

_RISK_DEVICE_RECOMMENDATION: dict[WanderingRiskLevel, str] = {
    WanderingRiskLevel.MODERATE: "Consider adding a GPS-enabled wearable or location-tracking insole.",
    WanderingRiskLevel.HIGH: (
        "A GPS-enabled wearable or tracking insole is strongly recommended; verify every "
        "exit-door alarm is active."
    ),
}

# Hazard types the caregiver's own reply cannot walk back from — an ambiguous or absent
# reply must still count as "unacknowledged", never as implicit resolution.
ACK_KEYWORDS = (
    "handled",
    "fixed",
    "resolved",
    "called",
    "turned off",
    "shut off",
    "locked",
    "secured",
    "on my way",
    "checking now",
    "ventilat",
    "okay now",
)

NO_ACK_KEYWORDS = (
    "can't",
    "cannot",
    "don't know",
    "not sure",
    "unable",
    "no idea",
    "not home",
    "too busy",
)

ACK_TIMEOUT_S = 90


def resolve_module(schedule_activity: str) -> ModuleId:
    lowered = schedule_activity.lower()
    for keyword, module in _SCHEDULE_KEYWORD_TO_MODULE.items():
        if keyword in lowered:
            return module
    return ModuleId.HOME_SAFETY_AUDIT


def existing_adl_room_labels() -> list[str]:
    """Rooms that already carry an ADL-agent environmental-cue door label, so wandering-prevention
    door measures are layered onto the existing label instead of tracked as a separate door list."""
    return list(ADL_TASK_ROOM_LABELS.values())


def build_checklist_sequence(rooms: list[str]) -> list[ChecklistItem]:
    items: list[ChecklistItem] = []
    for room in rooms:
        for hazard_type, mitigation_item, is_high_priority in HOME_CHECKLIST_TEMPLATES.get(room, []):
            items.append(
                ChecklistItem(
                    room=room,
                    hazard_type=hazard_type,
                    mitigation_item=mitigation_item,
                    is_high_priority=is_high_priority,
                )
            )
    # Stable sort: fire/gas and medication/chemical items (marked high priority) surface
    # ahead of lower-severity clutter/lighting items, without reshuffling same-tier items.
    items.sort(key=lambda item: not item.is_high_priority)
    return items


def classify_incident_severity(hazard_type: HazardType) -> Severity:
    """Every hazard type the real-time monitor currently models — fire/gas, falls, wandering
    geofence breaches, and unauthorized medication/chemical access — is emergency severity by
    design, regardless of the patient's disease stage."""
    del hazard_type
    return Severity.EMERGENCY


def escalate_risk_level(current: WanderingRiskLevel, hazard_type: HazardType) -> WanderingRiskLevel:
    if hazard_type not in (HazardType.WANDERING, HazardType.MEDICATION_CHEMICAL_ACCESS):
        return current
    idx = _RISK_ORDER.index(current)
    return _RISK_ORDER[min(idx + 1, len(_RISK_ORDER) - 1)]


def recommend_device_for_risk(risk: WanderingRiskLevel) -> Optional[str]:
    return _RISK_DEVICE_RECOMMENDATION.get(risk)


def resolve_checklist_status(resolved: Optional[bool]) -> ChecklistStatus:
    if resolved is True:
        return ChecklistStatus.DONE
    if resolved is False:
        return ChecklistStatus.NEEDS_ATTENTION
    return ChecklistStatus.PENDING


def assess_incident_acknowledgment(response_text: Optional[str], response_latency_s: float) -> tuple[bool, bool]:
    """Returns (acknowledged, should_escalate_to_family_contact).

    This routing decision is deterministic, not LLM-judged: an emergency escalation must never
    hinge on a model's read of a free-text reply.
    """
    if not response_text:
        return False, True
    lowered = response_text.lower()
    if any(keyword in lowered for keyword in NO_ACK_KEYWORDS) or response_latency_s > ACK_TIMEOUT_S:
        return False, True
    return True, False


class ChecklistItemEngine:
    def __init__(self, llm: BaseChatModel):
        self._chain = CHECKLIST_ITEM_PROMPT | llm

    def generate(self, patient: PatientProfile, item: ChecklistItem) -> str:
        existing_cue_note = "none"
        if item.hazard_type == HazardType.WANDERING:
            labeled_rooms = existing_adl_room_labels()
            if labeled_rooms:
                existing_cue_note = (
                    f"Doors already labeled for wayfinding: {', '.join(labeled_rooms)} — "
                    "place this measure on the actual exit/entry door, not those task-room doors."
                )
        result = self._chain.invoke(
            {
                "stage": patient.stage.value,
                "room": item.room,
                "hazard_type": item.hazard_type.value,
                "mitigation_item": item.mitigation_item,
                "priority_note": "high priority" if item.is_high_priority else "routine",
                "existing_cue_note": existing_cue_note,
            }
        )
        return result.content


class IncidentTriageEngine:
    def __init__(self, llm: BaseChatModel):
        self._chain = INCIDENT_TRIAGE_PROMPT | llm

    def generate(self, event: IncidentEvent, severity: Severity) -> str:
        result = self._chain.invoke(
            {
                "hazard_type": event.hazard_type.value,
                "location": event.location,
                "source_device": event.source_device,
                "description": event.description or "none provided",
                "severity": severity.value,
            }
        )
        return result.content


class ItemAssessmentLayer:
    def __init__(self, llm: BaseChatModel):
        self._chain = ITEM_ASSESSMENT_PROMPT | llm.with_structured_output(ItemAssessment)

    def generate(self, prompt: str, response: str) -> ItemAssessment:
        return self._chain.invoke({"prompt": prompt, "response": response})


class IncidentAssessmentLayer:
    def __init__(self, llm: BaseChatModel):
        self._chain = INCIDENT_ASSESSMENT_PROMPT | llm.with_structured_output(IncidentAssessment)

    def generate(self, prompt: str, response: str) -> IncidentAssessment:
        return self._chain.invoke({"prompt": prompt, "response": response})


class CaregiverReporter:
    def __init__(self, llm: BaseChatModel):
        self._summary_chain = CAREGIVER_SUMMARY_PROMPT | llm
        self._alert_chain = EMERGENCY_ALERT_PROMPT | llm

    def summarize(self, session_json: str) -> str:
        result = self._summary_chain.invoke({"session_json": session_json})
        return result.content

    def emergency_alert(self, session_json: str) -> str:
        result = self._alert_chain.invoke({"session_json": session_json})
        return result.content
