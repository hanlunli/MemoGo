from __future__ import annotations

from datetime import datetime
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from agents.cognitive_brain_training.models import MemoryItem
from agents.cognitive_brain_training.modules import ReminiscenceSessionManager
from agents.social_creative_engagement_training.modules import DEFAULT_SONG_MEMORY, MusicTherapySessionEngine

from .models import (
    HabitType,
    OutburstEvent,
    OutburstSeverity,
    PatientProfile,
    PreventionHabitItem,
    RedirectionType,
    SundowningJournalEntry,
)
from .prompts import (
    CAREGIVER_SUMMARY_PROMPT,
    DEESCALATION_ASSESSMENT_PROMPT,
    HABIT_ASSESSMENT_PROMPT,
    INDEPENDENCE_ASSESSMENT_PROMPT,
    INDEPENDENCE_TASK_PROMPT,
    OUTBURST_DEESCALATION_PROMPT,
    PREVENTION_HABIT_PROMPT,
    URGENT_ALERT_PROMPT,
    DeescalationAssessment,
    HabitAssessment,
    IndependenceAssessment,
)

# Every habit below carries a HabitType so priority ordering matches the "closer to the dusk
# window matters more" rule from the caregiving plan's environment/lighting adjustments.
PREVENTION_HABIT_TEMPLATE: list[tuple[HabitType, str, bool]] = [
    (
        HabitType.MORNING_SUNLIGHT,
        "Schedule 30-60 minutes of outdoor morning sunlight plus a light daytime activity "
        "(watering plants, drawing) to regulate the circadian rhythm.",
        False,
    ),
    (
        HabitType.NAP_CAFFEINE_SUGAR_LIMIT,
        "Keep daytime naps short and limit afternoon caffeine and high-sugar intake; keep dinner light.",
        False,
    ),
    (
        HabitType.ENVIRONMENT_LIGHTING,
        "Draw the curtains and turn on bright, warm indoor lights before the dusk window; turn off "
        "TVs/radios and reduce household traffic.",
        True,
    ),
    (
        HabitType.EVENING_ROUTINE_PROTECTION,
        "Confirm no room changes, baths, or guest visits are scheduled inside the dusk/evening window.",
        True,
    ),
    (
        HabitType.BEDTIME_WARM_DRINK,
        "Offer a small cup of warm milk or a gentle hot drink before bedtime.",
        False,
    ),
]

DEFAULT_PHOTO_MEMORY = MemoryItem(
    media_type="photo", theme="a family photo album", decade="various decades", description="familiar family photos"
)

_REDIRECTION_KEYWORDS: dict[RedirectionType, tuple[str, ...]] = {
    RedirectionType.MUSIC: ("music", "song", "sing"),
    RedirectionType.PHOTO_ALBUM: ("photo", "album", "picture"),
    RedirectionType.SNACK: ("snack", "food", "treat", "milk", "tea", "cookie"),
    RedirectionType.HANDS_ON_TASK: ("fold", "towel", "task", "clothes"),
    RedirectionType.FAMILIAR_ITEM: ("vintage", "item", "keepsake", "blanket"),
}

_DEFAULT_REDIRECTION_TEXT: dict[RedirectionType, str] = {
    RedirectionType.FAMILIAR_ITEM: "a familiar vintage item, like an old keepsake",
    RedirectionType.SNACK: "a light, favorite snack",
    RedirectionType.HANDS_ON_TASK: "a simple hands-on task, like folding a towel",
}

# A calmed/still-distressed reply must never be ambiguous — an absent or unrelated reply keeps
# the de-escalation loop going rather than being read as either outcome.
DEESCALATION_ACK_KEYWORDS = (
    "calm",
    "calmer",
    "settled",
    "settling",
    "smiling",
    "listening",
    "sleeping",
    "relaxed",
    "better now",
    "singing along",
    "holding my hand",
    "engaged",
)

DEESCALATION_NO_ACK_KEYWORDS = (
    "still upset",
    "won't stop",
    "getting worse",
    "worse",
    "hitting",
    "screaming",
    "won't calm",
    "still agitated",
    "no change",
    "still crying",
)

DEESCALATION_TIMEOUT_S = 90

HIGH_SEVERITY_SYMPTOM_KEYWORDS = (
    "hitting",
    "screaming",
    "trying to leave",
    "throwing",
    "aggressive",
    "hallucinat",
    "combative",
)

MILD_SYMPTOM_KEYWORDS = (
    "restless",
    "pacing",
    "muttering",
    "fidgeting",
    "anxious",
    "uneasy",
)


def build_prevention_sequence(patient: PatientProfile) -> list[PreventionHabitItem]:
    habit_items = [
        PreventionHabitItem(habit_type=habit_type, description=description, is_high_priority=is_high_priority)
        for habit_type, description, is_high_priority in PREVENTION_HABIT_TEMPLATE
    ]
    # Stable sort: environment/lighting and evening-routine-protection (high priority, closest to
    # the dusk window) surface first without reshuffling same-tier items.
    habit_items.sort(key=lambda item: not item.is_high_priority)
    independence_item = PreventionHabitItem(
        habit_type=HabitType.INDEPENDENCE_TASK_OFFER,
        description=select_independence_task(patient),
        is_high_priority=False,
    )
    return habit_items + [independence_item]


def select_independence_task(patient: PatientProfile) -> str:
    return patient.independence_tasks[0] if patient.independence_tasks else "watering the plants"


def classify_outburst_severity(event: OutburstEvent) -> OutburstSeverity:
    reported = [s.lower() for s in event.symptoms_observed] or ([event.description.lower()] if event.description else [])
    if not reported:
        return OutburstSeverity.RESTLESSNESS
    if any(any(keyword in item for keyword in HIGH_SEVERITY_SYMPTOM_KEYWORDS) for item in reported):
        return OutburstSeverity.OUTBURST
    if all(any(keyword in item for keyword in MILD_SYMPTOM_KEYWORDS) for item in reported):
        return OutburstSeverity.RESTLESSNESS
    return OutburstSeverity.OUTBURST


def assess_calming_signal(response_text: Optional[str], response_latency_s: float) -> Optional[bool]:
    """Returns True if calmed, False if clearly still distressed, None if ambiguous/no reply.

    This routing decision is deterministic, not LLM-judged: whether the de-escalation loop
    continues or stops must never hinge on a model's read of a free-text reply.
    """
    if not response_text:
        return None
    if response_latency_s > DEESCALATION_TIMEOUT_S:
        return None
    lowered = response_text.lower()
    if any(keyword in lowered for keyword in DEESCALATION_NO_ACK_KEYWORDS):
        return False
    if any(keyword in lowered for keyword in DEESCALATION_ACK_KEYWORDS):
        return True
    return None


def _map_preferences_to_redirection_types(preferences: list[str]) -> list[RedirectionType]:
    lowered_prefs = [p.lower() for p in preferences]
    matched: list[RedirectionType] = []
    for redirection_type, keywords in _REDIRECTION_KEYWORDS.items():
        if any(any(keyword in pref for keyword in keywords) for pref in lowered_prefs):
            matched.append(redirection_type)
    return matched


def select_redirection(patient: PatientProfile, used: list[RedirectionType]) -> RedirectionType:
    preferred = _map_preferences_to_redirection_types(patient.calming_preferences)
    for candidates in (preferred, list(RedirectionType)):
        for redirection_type in candidates:
            if redirection_type not in used:
                return redirection_type
    all_types = list(RedirectionType)
    return all_types[len(used) % len(all_types)]


def resolve_checklist_status(addressed: Optional[bool]) -> str:
    if addressed is True:
        return "done"
    if addressed is False:
        return "needs_attention"
    return "pending"


def _significant_words(text: str) -> set[str]:
    # Strip a trailing "s" as a light singular/plural normalization (e.g. "visitor" vs
    # "visitors") and drop short filler words so overlap comparisons key on meaningful terms.
    return {word.rstrip("s") for word in text.lower().split() if len(word) > 3}


def matches_known_trigger(event: OutburstEvent, patient: PatientProfile) -> Optional[str]:
    """Checks the outburst's reported context against the patient's known/confirmed triggers, so a
    recognized antecedent is surfaced to the caregiver in the journal instead of being logged as if
    every episode were novel — feeding the personalization profile's known_triggers back into the
    journal per the sundowning-journal-trigger-tracker's avoid-list/correlation behavior."""
    candidate_words: set[str] = set()
    for candidate in (event.preceding_event, event.environmental_factor, event.food_or_drink_intake):
        if candidate:
            candidate_words |= _significant_words(candidate)
    for trigger in patient.known_triggers:
        if _significant_words(trigger) & candidate_words:
            return trigger
    return None


def build_journal_entry(
    event: OutburstEvent,
    intervention_used: str,
    outcome: str,
    started_at: str,
    matched_known_trigger: Optional[str] = None,
) -> SundowningJournalEntry:
    started = datetime.fromisoformat(started_at)
    return SundowningJournalEntry(
        date=started.date().isoformat(),
        time_of_onset=started.time().isoformat(timespec="minutes"),
        preceding_event=event.preceding_event or "not reported",
        food_or_drink_intake=event.food_or_drink_intake or "not reported",
        environmental_factor=event.environmental_factor or "not reported",
        symptoms_observed=event.symptoms_observed,
        intervention_used=intervention_used,
        outcome=outcome,
        matched_known_trigger=matched_known_trigger,
    )


class PreventionHabitEngine:
    def __init__(self, llm: BaseChatModel):
        self._chain = PREVENTION_HABIT_PROMPT | llm

    def generate(self, patient: PatientProfile, item: PreventionHabitItem) -> str:
        result = self._chain.invoke(
            {
                "stage": patient.stage.value,
                "habit_type": item.habit_type.value,
                "description": item.description,
                "priority_note": "high priority — close to the dusk window" if item.is_high_priority else "routine",
            }
        )
        return result.content


class IndependenceTaskEngine:
    def __init__(self, llm: BaseChatModel):
        self._chain = INDEPENDENCE_TASK_PROMPT | llm

    def generate(self, patient: PatientProfile, task: str) -> str:
        result = self._chain.invoke({"name": patient.name, "task": task})
        return result.content


class DeescalationCoach:
    """Delivers the validate-and-redirect script, drawing redirection content directly from the
    Social & Creative Engagement agent's music-therapy engine and the Cognitive agent's
    reminiscence manager instead of maintaining a separate redirection-content library."""

    def __init__(self, llm: BaseChatModel):
        self._chain = OUTBURST_DEESCALATION_PROMPT | llm
        self._music_engine = MusicTherapySessionEngine(llm)
        self._reminiscence_engine = ReminiscenceSessionManager(llm)

    def render_redirection_content(self, patient: PatientProfile, redirection_type: RedirectionType) -> str:
        if redirection_type == RedirectionType.MUSIC:
            return self._music_engine.generate_instruction(patient, DEFAULT_SONG_MEMORY, "listen_only")
        if redirection_type == RedirectionType.PHOTO_ALBUM:
            return self._reminiscence_engine.generate_prompt(patient, DEFAULT_PHOTO_MEMORY)
        for preference in patient.calming_preferences:
            if any(keyword in preference.lower() for keyword in _REDIRECTION_KEYWORDS[redirection_type]):
                return preference
        return _DEFAULT_REDIRECTION_TEXT[redirection_type]

    def generate(
        self,
        patient: PatientProfile,
        severity: OutburstSeverity,
        attempt_number: int,
        redirection_type: RedirectionType,
        redirection_content: str,
        preceding_event: str,
    ) -> str:
        result = self._chain.invoke(
            {
                "severity": severity.value,
                "attempt_number": attempt_number,
                "redirection_type": redirection_type.value,
                "redirection_content": redirection_content,
                "contact_allowed": "yes" if patient.accepts_physical_contact else "no — avoid touch/hand-holding",
                "preceding_event": preceding_event or "not reported",
            }
        )
        return result.content


class HabitAssessmentLayer:
    def __init__(self, llm: BaseChatModel):
        self._chain = HABIT_ASSESSMENT_PROMPT | llm.with_structured_output(HabitAssessment)

    def generate(self, prompt: str, response: str) -> HabitAssessment:
        return self._chain.invoke({"prompt": prompt, "response": response})


class IndependenceAssessmentLayer:
    def __init__(self, llm: BaseChatModel):
        self._chain = INDEPENDENCE_ASSESSMENT_PROMPT | llm.with_structured_output(IndependenceAssessment)

    def generate(self, prompt: str, response: str) -> IndependenceAssessment:
        return self._chain.invoke({"prompt": prompt, "response": response})


class DeescalationAssessmentLayer:
    def __init__(self, llm: BaseChatModel):
        self._chain = DEESCALATION_ASSESSMENT_PROMPT | llm.with_structured_output(DeescalationAssessment)

    def generate(self, prompt: str, response: str) -> DeescalationAssessment:
        return self._chain.invoke({"prompt": prompt, "response": response})


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
