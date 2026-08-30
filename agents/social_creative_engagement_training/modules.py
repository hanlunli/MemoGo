from __future__ import annotations

import random
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from agents.cognitive_brain_training.models import MemoryItem

from .models import DiseaseStage, IncidentFlag, ModuleId, PatientProfile, SocialContact
from .prompts import (
    CAREGIVER_SUMMARY_PROMPT,
    CRAFT_STEP_INSTRUCTION_PROMPT,
    MUSIC_INSTRUCTION_PROMPT,
    PARTICIPATION_ASSESSMENT_PROMPT,
    SOCIAL_CONVERSATION_PROMPT,
    URGENT_ALERT_PROMPT,
    ParticipationAssessment,
)

AGITATION_KEYWORDS = (
    "agitated",
    "upset",
    "scared",
    "angry",
    "overwhelmed",
    "don't want",
    "no more",
    "stop",
    "leave me alone",
)

HAZARD_KEYWORDS: dict[IncidentFlag, tuple[str, ...]] = {
    IncidentFlag.CUT: ("cut myself", "cut my", "bleeding"),
    IncidentFlag.BURN: ("burned", "got burned", "scalded"),
    IncidentFlag.CHOKING_RISK: ("choking", "swallowed", "stuck in my throat", "choked on"),
}

DEFAULT_SONG_MEMORY = MemoryItem(media_type="song", theme="folk songs", decade="1960s")

PARTICIPATION_MODE_LABELS = {
    "clap_along": "clap along",
    "sing_along": "sing along",
    "listen_only": "just listen",
}

CRAFT_TASKS: dict[str, list[tuple[str, bool]]] = {
    "bead stringing": [
        ("Pick up one bead.", False),
        ("Thread the bead onto the string.", False),
        ("Push the bead down to the end.", False),
        ("Pick up another bead.", False),
    ],
    "origami": [
        ("Pick up the square paper.", False),
        ("Fold the paper in half to make a triangle.", False),
        ("Press along the fold to flatten it.", False),
    ],
    "watering plants": [
        ("Pick up the watering can.", False),
        ("Walk to the first plant.", False),
        ("Pour a little water into the soil.", False),
    ],
    "drawing": [
        ("Pick up a crayon.", False),
        ("Draw a simple shape on the paper.", False),
    ],
    "paper cutting": [
        ("Pick up the safety scissors.", True),
        ("Cut slowly along the line.", True),
        ("Set the scissors down on the table.", True),
    ],
}

CRAFT_ACTIVITY_TYPES = tuple(CRAFT_TASKS.keys())

DEFAULT_CONVERSATION_TOPICS = ("a favorite childhood memory", "the weather today", "a favorite meal")

STAGE_COMPLEXITY_CEILING = {
    DiseaseStage.MILD: 5,
    DiseaseStage.MODERATE: 3,
    DiseaseStage.SEVERE: 1,
}

ACTIVITY_KIND_LABELS = {
    ModuleId.MUSIC_THERAPY: "music therapy prompt",
    ModuleId.CRAFTS_HORTICULTURE: "craft/horticulture step",
    ModuleId.SOCIAL_INTERACTION: "social conversation turn",
}

_SCHEDULE_KEYWORD_TO_MODULE = {
    "music": ModuleId.MUSIC_THERAPY,
    "fine motor": ModuleId.CRAFTS_HORTICULTURE,
    "art therapy": ModuleId.CRAFTS_HORTICULTURE,
    "crafts": ModuleId.CRAFTS_HORTICULTURE,
    "horticulture": ModuleId.CRAFTS_HORTICULTURE,
    "social engagement": ModuleId.SOCIAL_INTERACTION,
    "reminiscence & social": ModuleId.SOCIAL_INTERACTION,
}


class MusicTherapySessionEngine:
    def __init__(self, llm: BaseChatModel):
        self._chain = MUSIC_INSTRUCTION_PROMPT | llm

    def generate_instruction(self, patient: PatientProfile, memory: MemoryItem, participation_mode: str) -> str:
        result = self._chain.invoke(
            {
                "stage": patient.stage.value,
                "name": patient.name,
                "theme": memory.theme,
                "decade": memory.decade,
                "participation_mode_label": PARTICIPATION_MODE_LABELS[participation_mode],
            }
        )
        return result.content


class CraftsHorticultureActivityEngine:
    def __init__(self, llm: BaseChatModel):
        self._chain = CRAFT_STEP_INSTRUCTION_PROMPT | llm

    def generate(
        self,
        patient: PatientProfile,
        activity_type: str,
        step_description: str,
        is_hazard_step: bool,
    ) -> str:
        result = self._chain.invoke(
            {
                "stage": patient.stage.value,
                "name": patient.name,
                "activity_type": activity_type,
                "step_description": step_description,
                "hazard_note": (
                    "This step needs a caregiver present — supervise closely."
                    if is_hazard_step
                    else "none"
                ),
                "sensitivities": ", ".join(patient.material_sensitivities) or "none reported",
            }
        )
        return result.content


class SocialInteractionFacilitator:
    def __init__(self, llm: BaseChatModel):
        self._chain = SOCIAL_CONVERSATION_PROMPT | llm

    def generate(self, patient: PatientProfile, contact: Optional[SocialContact], topic: str) -> str:
        contact_desc = f"{contact.name} ({contact.relationship})" if contact else "a caregiver or family member"
        result = self._chain.invoke(
            {
                "stage": patient.stage.value,
                "name": patient.name,
                "topic": topic,
                "contact_desc": contact_desc,
            }
        )
        return result.content


class FeedbackEncouragementLayer:
    def __init__(self, llm: BaseChatModel):
        self._chain = PARTICIPATION_ASSESSMENT_PROMPT | llm.with_structured_output(ParticipationAssessment)

    def generate(self, prompt: str, module: ModuleId, response: str) -> ParticipationAssessment:
        return self._chain.invoke(
            {"prompt": prompt, "activity_kind": ACTIVITY_KIND_LABELS[module], "response": response}
        )


class CaregiverReporter:
    def __init__(self, llm: BaseChatModel):
        self._summary_chain = CAREGIVER_SUMMARY_PROMPT | llm
        self._alert_chain = URGENT_ALERT_PROMPT | llm

    def summarize(self, session_json: str) -> str:
        result = self._summary_chain.invoke({"session_json": session_json})
        return result.content

    def urgent_alert(self, session_json: str) -> str:
        result = self._alert_chain.invoke({"session_json": session_json})
        return result.content


def resolve_module(schedule_activity: str) -> ModuleId:
    lowered = schedule_activity.lower()
    for keyword, module in _SCHEDULE_KEYWORD_TO_MODULE.items():
        if keyword in lowered:
            return module
    return ModuleId.CRAFTS_HORTICULTURE


def select_craft_activity(patient: PatientProfile) -> str:
    preferred = [c for c in patient.preferred_crafts if c in CRAFT_TASKS]
    return random.choice(preferred) if preferred else random.choice(CRAFT_ACTIVITY_TYPES)


def build_craft_step_sequence(activity_type: str, stage: DiseaseStage) -> list[tuple[str, bool]]:
    steps = list(CRAFT_TASKS[activity_type])
    if activity_type == "bead stringing" and stage in (DiseaseStage.MODERATE, DiseaseStage.SEVERE):
        steps = [(description, True) for description, _ in steps]
    return steps


def select_contact(patient: PatientProfile) -> Optional[SocialContact]:
    return random.choice(patient.social_contacts) if patient.social_contacts else None


def select_conversation_topic(patient: PatientProfile) -> str:
    topics = patient.conversation_topics or DEFAULT_CONVERSATION_TOPICS
    return random.choice(topics)


def choose_participation_mode(complexity: int, agitation_detected: bool) -> str:
    if agitation_detected:
        return "listen_only"
    if complexity >= 4:
        return "sing_along"
    if complexity >= 2:
        return "clap_along"
    return "listen_only"


def detect_material_hazard(response_text: Optional[str]) -> IncidentFlag:
    if not response_text:
        return IncidentFlag.NONE
    lowered = response_text.lower()
    for flag, keywords in HAZARD_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return flag
    return IncidentFlag.NONE


def assess_mood_and_engagement(response_text: Optional[str], response_latency_s: float) -> tuple[bool, float]:
    if not response_text:
        return True, 0.0
    lowered = response_text.lower()
    keyword_hit = any(keyword in lowered for keyword in AGITATION_KEYWORDS)
    slow_response = response_latency_s > 45
    agitation_detected = keyword_hit or slow_response
    engagement_score = 0.3 if agitation_detected else min(1.0, 1.2 - response_latency_s / 60)
    return agitation_detected, round(max(0.0, engagement_score), 2)


def adjust_complexity(
    complexity: int,
    ceiling: int,
    participated: Optional[bool],
    agitation_detected: bool,
    incident_flag: IncidentFlag,
) -> int:
    if incident_flag is not IncidentFlag.NONE:
        return 1
    if agitation_detected:
        return max(1, complexity - 1)
    if participated is True:
        return min(ceiling, complexity + 1)
    if participated is False:
        return max(1, complexity - 1)
    return complexity


def is_safety_stop(incident_flag: IncidentFlag) -> bool:
    return incident_flag is not IncidentFlag.NONE
