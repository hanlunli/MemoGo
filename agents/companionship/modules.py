from __future__ import annotations

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from agents.cognitive_brain_training.models import MemoryItem
from agents.cognitive_brain_training.modules import ReminiscenceSessionManager

from .models import (
    ActivityStepType,
    DistortionType,
    PatientProfile,
    ReminiscenceBoxItem,
    SelfCareNudgeType,
    StageActivityStep,
)
from .prompts import (
    ACTIVE_LISTENING_PROMPT,
    ATMOSPHERE_SETUP_PROMPT,
    CAREGIVER_SUMMARY_PROMPT,
    DISTORTION_RESPONSE_PROMPT,
    DISTORTION_VALIDATION_ASSESSMENT_PROMPT,
    HEURISTIC_PROMPTING_PROMPT,
    REMINISCENCE_ENGAGEMENT_ASSESSMENT_PROMPT,
    SELF_CARE_NUDGE_PROMPT,
    STAGE_ACTIVITY_ASSESSMENT_PROMPT,
    STAGE_ACTIVITY_PROMPT,
    TOPIC_MEDIA_PREP_PROMPT,
    UNRESOLVED_ALERT_PROMPT,
    DistortionValidationAssessment,
    ReminiscenceEngagementAssessment,
    StageActivityAssessment,
)

# Each stage's activities come straight from companionship_content.json's
# stageBasedCompanionshipStrategies; only the Early-stage "practice Reminiscence Therapy" bullet
# is tagged REMINISCENCE_THERAPY so the graph can expand it into the full 4-step protocol below.
EARLY_STAGE_ACTIVITIES: list[tuple[ActivityStepType, str, str]] = [
    (
        ActivityStepType.STAGE_ACTIVITY,
        "low_intensity_exercise",
        "Accompany them in low-intensity exercise (walking, jogging, Tai Chi).",
    ),
    (
        ActivityStepType.REMINISCENCE_THERAPY,
        "reminiscence_therapy",
        "Look through old photo albums, listen to classic songs, and practice Reminiscence Therapy.",
    ),
    (
        ActivityStepType.STAGE_ACTIVITY,
        "light_chores",
        "Encourage participation in light chores (folding clothes, sorting vegetables) to maintain a "
        "sense of self-worth.",
    ),
]

MIDDLE_STAGE_ACTIVITIES: list[tuple[ActivityStepType, str, str]] = [
    (
        ActivityStepType.STAGE_ACTIVITY,
        "simplified_language",
        "Use short, simple sentences with a slow pace, supplemented by gestures and eye contact.",
    ),
    (
        ActivityStepType.STAGE_ACTIVITY,
        "predictable_routine_labels",
        "Establish a predictable daily routine; label essential items with large-font text and icon cues.",
    ),
    (
        ActivityStepType.STAGE_ACTIVITY,
        "tactile_activity",
        "Offer simple tactile activities (hand squeezes, light massages, simple puzzles, or modeling clay).",
    ),
]

LATE_STAGE_ACTIVITIES: list[tuple[ActivityStepType, str, str]] = [
    (
        ActivityStepType.STAGE_ACTIVITY,
        "non_verbal_connection",
        "Focus on non-verbal communication: hand-holding, back-stroking, and playing familiar music or "
        "white noise.",
    ),
    (
        ActivityStepType.STAGE_ACTIVITY,
        "grooming_dignity_care",
        "Maintain proper grooming and hygiene; use warm physical contact to provide a reassuring sense "
        "of safety.",
    ),
]

STAGE_ACTIVITY_TEMPLATE: dict[str, list[tuple[ActivityStepType, str, str]]] = {
    "mild": EARLY_STAGE_ACTIVITIES,
    "moderate": MIDDLE_STAGE_ACTIVITIES,
    "severe": LATE_STAGE_ACTIVITIES,
}

SELF_CARE_NUDGE_GUIDANCE: dict[SelfCareNudgeType, str] = {
    SelfCareNudgeType.EMOTIONAL_BOUNDARY_REFRAME: (
        "Recognize that aggressive language or emotional detachment stems from the disease itself, not "
        "personal hostility."
    ),
    SelfCareNudgeType.RESPITE_REMINDER: (
        "Utilize community day centers, short-term care facilities, or family shift-rotations to ensure "
        "caregivers get dedicated weekly rest time."
    ),
}

RESPITE_OVERDUE_DAYS = 7

DEFAULT_SAFE_REMINISCENCE_THEME = "a favorite everyday memory, like a family meal or a garden"

REMINISCENCE_STEP_COUNT = 4

FATIGUE_KEYWORDS = (
    "tired",
    "yawning",
    "wants to stop",
    "losing interest",
    "getting bored",
    "sleepy",
    "had enough",
    "restless now",
)

VALIDATION_ACK_KEYWORDS = (
    "settled",
    "sat down",
    "agreed",
    "calmer",
    "smiling",
    "let it go",
    "moved on",
    "distracted",
    "relaxed",
    "nodding",
)

VALIDATION_NO_ACK_KEYWORDS = (
    "still insists",
    "still upset",
    "won't let it go",
    "still asking",
    "getting more agitated",
    "still accusing",
    "still trying to leave",
)

VALIDATION_TIMEOUT_S = 90

_DISTORTION_KEYWORDS: dict[DistortionType, tuple[str, ...]] = {
    DistortionType.WANTS_TO_GO_HOME: ("go home", "want to leave", "take me home"),
    DistortionType.MISIDENTIFICATION: ("who are you", "you're not my", "where is my", "mistook", "thinks i'm"),
    DistortionType.ACCUSATION_OR_SUSPICION: ("stole", "missing", "took my", "someone took"),
}

NUDGE_ACK_KEYWORDS = ("yes", "will do", "booked", "scheduled", "sounds good", "already planned", "ok i will")
NUDGE_NO_ACK_KEYWORDS = ("no time", "can't", "too busy", "not now", "won't")


def build_stage_activity_sequence(patient: PatientProfile) -> list[StageActivityStep]:
    template = STAGE_ACTIVITY_TEMPLATE[patient.stage.value]
    steps = [
        StageActivityStep(step_type=step_type, activity_key=activity_key, description=description)
        for step_type, activity_key, description in template
    ]
    steps.append(
        StageActivityStep(
            step_type=ActivityStepType.SELF_CARE_NUDGE,
            activity_key="self_care",
            description="Caregiver self-care nudge",
        )
    )
    return steps


def select_self_care_nudge(patient: PatientProfile) -> SelfCareNudgeType:
    if patient.days_since_last_respite >= RESPITE_OVERDUE_DAYS:
        return SelfCareNudgeType.RESPITE_REMINDER
    return SelfCareNudgeType.EMOTIONAL_BOUNDARY_REFRAME


def personalize_reminiscence_theme(patient: PatientProfile) -> str:
    if patient.reminiscence_background:
        return patient.reminiscence_background[0]
    return "their younger years"


def is_sensitive_topic(theme: str, patient: PatientProfile) -> bool:
    lowered_theme = theme.lower()
    return any(
        topic.lower() in lowered_theme or lowered_theme in topic.lower() for topic in patient.sensitive_topics_to_avoid
    )


def resolve_reminiscence_theme(patient: PatientProfile) -> str:
    """Never probes a traumatic/sensitive topic — swaps to a safe default instead of asking why."""
    theme = personalize_reminiscence_theme(patient)
    if is_sensitive_topic(theme, patient):
        return DEFAULT_SAFE_REMINISCENCE_THEME
    return theme


def build_reminiscence_box(theme: str) -> ReminiscenceBoxItem:
    return ReminiscenceBoxItem(
        visual=f"Old photos, keepsakes, or mementos related to {theme}.",
        auditory=f"Familiar songs or sounds from the era of {theme}.",
        tactile_olfactory="A familiar fabric, scent, or spice tied to that time.",
    )


def detects_fatigue_signal(response_text: Optional[str]) -> bool:
    if not response_text:
        return False
    lowered = response_text.lower()
    return any(keyword in lowered for keyword in FATIGUE_KEYWORDS)


def assess_validation_signal(response_text: Optional[str], response_latency_s: float) -> Optional[bool]:
    """Returns True if settled, False if clearly still distressed, None if ambiguous/no reply.

    This routing decision is deterministic, not LLM-judged: whether the validate-and-redirect loop
    continues or stops must never hinge on a model's read of a free-text reply.
    """
    if not response_text:
        return None
    if response_latency_s > VALIDATION_TIMEOUT_S:
        return None
    lowered = response_text.lower()
    if any(keyword in lowered for keyword in VALIDATION_NO_ACK_KEYWORDS):
        return False
    if any(keyword in lowered for keyword in VALIDATION_ACK_KEYWORDS):
        return True
    return None


def classify_distortion_type(statement: str) -> DistortionType:
    lowered = (statement or "").lower()
    for distortion_type, keywords in _DISTORTION_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return distortion_type
    return DistortionType.OTHER


def detect_nudge_acceptance(response_text: Optional[str]) -> Optional[bool]:
    if not response_text:
        return None
    lowered = response_text.lower()
    if any(keyword in lowered for keyword in NUDGE_NO_ACK_KEYWORDS):
        return False
    if any(keyword in lowered for keyword in NUDGE_ACK_KEYWORDS):
        return True
    return None


def resolve_engagement_status(addressed: Optional[bool]) -> str:
    if addressed is True:
        return "done"
    if addressed is False:
        return "needs_attention"
    return "pending"


class StageActivityEngine:
    def __init__(self, llm: BaseChatModel):
        self._chain = STAGE_ACTIVITY_PROMPT | llm

    def generate(self, patient: PatientProfile, step: StageActivityStep) -> str:
        result = self._chain.invoke(
            {"stage": patient.stage.value, "activity_key": step.activity_key, "description": step.description}
        )
        return result.content


class SelfCareNudgeEngine:
    def __init__(self, llm: BaseChatModel):
        self._chain = SELF_CARE_NUDGE_PROMPT | llm

    def generate(self, patient: PatientProfile, nudge_type: SelfCareNudgeType) -> str:
        result = self._chain.invoke(
            {
                "caregiver_name": patient.caregiver_name or "there",
                "nudge_type": nudge_type.value,
                "guidance": SELF_CARE_NUDGE_GUIDANCE[nudge_type],
            }
        )
        return result.content


class ReminiscenceSessionEngine:
    """Runs the 4-step Reminiscence Therapy protocol from companionship_content.json.

    Step 3's open-ended question is generated by the Cognitive & Brain Training agent's
    ReminiscenceSessionManager (shared memory-bank engine) instead of a separate question
    generator, matching this repo's convention of reusing another agent's capability rather
    than duplicating it.
    """

    def __init__(self, llm: BaseChatModel):
        self._topic_media_chain = TOPIC_MEDIA_PREP_PROMPT | llm
        self._atmosphere_chain = ATMOSPHERE_SETUP_PROMPT | llm
        self._heuristic_chain = HEURISTIC_PROMPTING_PROMPT | llm
        self._listening_chain = ACTIVE_LISTENING_PROMPT | llm
        self._reminiscence_manager = ReminiscenceSessionManager(llm)

    def generate_step(
        self, patient: PatientProfile, step_index: int, theme: str, box: ReminiscenceBoxItem
    ) -> str:
        if step_index == 0:
            result = self._topic_media_chain.invoke(
                {
                    "name": patient.name,
                    "theme": theme,
                    "visual": box.visual,
                    "auditory": box.auditory,
                    "tactile_olfactory": box.tactile_olfactory,
                }
            )
            return result.content
        if step_index == 1:
            result = self._atmosphere_chain.invoke({"name": patient.name})
            return result.content
        if step_index == 2:
            memory = MemoryItem(media_type="photo_or_song", theme=theme, decade="their youth", description=box.visual)
            generated_question = self._reminiscence_manager.generate_prompt(patient, memory)
            result = self._heuristic_chain.invoke(
                {"name": patient.name, "theme": theme, "generated_question": generated_question}
            )
            return result.content
        result = self._listening_chain.invoke({"name": patient.name})
        return result.content


class DistortionResponseCoach:
    def __init__(self, llm: BaseChatModel):
        self._chain = DISTORTION_RESPONSE_PROMPT | llm

    def generate(self, distortion_type: DistortionType, attempt_number: int, patient_statement: str) -> str:
        result = self._chain.invoke(
            {
                "distortion_type": distortion_type.value,
                "attempt_number": attempt_number,
                "patient_statement": patient_statement or "not reported",
            }
        )
        return result.content


class StageActivityAssessmentLayer:
    def __init__(self, llm: BaseChatModel):
        self._chain = STAGE_ACTIVITY_ASSESSMENT_PROMPT | llm.with_structured_output(StageActivityAssessment)

    def generate(self, prompt: str, response: str) -> StageActivityAssessment:
        return self._chain.invoke({"prompt": prompt, "response": response})


class ReminiscenceEngagementAssessmentLayer:
    def __init__(self, llm: BaseChatModel):
        self._chain = REMINISCENCE_ENGAGEMENT_ASSESSMENT_PROMPT | llm.with_structured_output(
            ReminiscenceEngagementAssessment
        )

    def generate(self, prompt: str, response: str) -> ReminiscenceEngagementAssessment:
        return self._chain.invoke({"prompt": prompt, "response": response})


class DistortionValidationAssessmentLayer:
    def __init__(self, llm: BaseChatModel):
        self._chain = DISTORTION_VALIDATION_ASSESSMENT_PROMPT | llm.with_structured_output(
            DistortionValidationAssessment
        )

    def generate(self, prompt: str, response: str) -> DistortionValidationAssessment:
        return self._chain.invoke({"prompt": prompt, "response": response})


class CaregiverReporter:
    def __init__(self, llm: BaseChatModel):
        self._summary_chain = CAREGIVER_SUMMARY_PROMPT | llm
        self._alert_chain = UNRESOLVED_ALERT_PROMPT | llm

    def summarize(self, session_json: str) -> str:
        result = self._summary_chain.invoke({"session_json": session_json})
        return result.content

    def unresolved_alert(self, session_json: str) -> str:
        result = self._alert_chain.invoke({"session_json": session_json})
        return result.content
