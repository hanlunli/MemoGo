import os
import time

import streamlit as st
from dotenv import load_dotenv

from agents.adl_training import ADLTrainingAgent
from agents.adl_training import DiseaseStage as ADLDiseaseStage
from agents.adl_training import MobilityLevel as ADLMobilityLevel
from agents.adl_training import PatientProfile as ADLPatientProfile
from agents.adl_training.llm_logging import configure_logging as configure_adl_logging
from agents.cognitive_brain_training import CognitiveBrainTrainingAgent
from agents.cognitive_brain_training import DiseaseStage as CognitiveDiseaseStage
from agents.cognitive_brain_training import PatientProfile as CognitivePatientProfile
from agents.cognitive_brain_training.llm_logging import configure_logging as configure_cognitive_logging
from agents.daily_life_routine_management import DailyLifeRoutineManagementAgent, DeviationEvent, DeviationType
from agents.daily_life_routine_management import DiseaseStage as DailyLifeDiseaseStage
from agents.daily_life_routine_management import MobilityLevel as DailyLifeMobilityLevel
from agents.daily_life_routine_management import PatientProfile as DailyLifePatientProfile
from agents.daily_life_routine_management.llm_logging import configure_logging as configure_daily_life_logging
from agents.daily_life_routine_management.models import CheckpointType
from agents.emotional_support_comfort import EmotionalSupportComfortAgent, OutburstEvent, SundowningRiskLevel
from agents.emotional_support_comfort import DiseaseStage as EmotionalDiseaseStage
from agents.emotional_support_comfort import PatientProfile as EmotionalPatientProfile
from agents.emotional_support_comfort.llm_logging import configure_logging as configure_emotional_logging
from agents.exercise_motor_coordination_training import ExerciseMotorCoordinationTrainingAgent, MobilityLevel
from agents.exercise_motor_coordination_training import DiseaseStage as ExerciseDiseaseStage
from agents.exercise_motor_coordination_training import PatientProfile as ExercisePatientProfile
from agents.exercise_motor_coordination_training.llm_logging import configure_logging as configure_exercise_logging
from agents.home_safety_protection import HazardType, HomeSafetyProtectionAgent, IncidentEvent
from agents.home_safety_protection import DiseaseStage as HomeSafetyDiseaseStage
from agents.home_safety_protection import MobilityLevel as HomeSafetyMobilityLevel
from agents.home_safety_protection import PatientProfile as HomeSafetyPatientProfile
from agents.home_safety_protection import WanderingRiskLevel
from agents.home_safety_protection.llm_logging import configure_logging as configure_home_safety_logging
from agents.social_creative_engagement_training import FineMotorLevel, SocialContact, SocialCreativeEngagementTrainingAgent
from agents.social_creative_engagement_training import DiseaseStage as SocialDiseaseStage
from agents.social_creative_engagement_training import PatientProfile as SocialPatientProfile
from agents.social_creative_engagement_training.llm_logging import configure_logging as configure_social_logging

load_dotenv()
configure_cognitive_logging()
configure_exercise_logging()
configure_adl_logging()
configure_social_logging()
configure_home_safety_logging()
configure_daily_life_logging()
configure_emotional_logging()

AGENT_LABELS = {
    "cognitive": "🧠 Cognitive & Brain Training",
    "exercise": "🏃 Exercise & Motor Coordination Training",
    "adl": "🧺 Activities of Daily Living (ADL) Training",
    "social": "🎨 Social & Creative Engagement Training",
    "home_safety": "🏠 Home Safety & Protection",
    "daily_life": "📅 Daily Life & Routine Management",
    "emotional_support": "💛 Emotional Support & Comfort",
}

COGNITIVE_SCHEDULE_OPTIONS = [
    "07:30-08:30 Reality orientation morning check-in",
    "09:30-10:30 Targeted cognitive training",
    "13:00-14:30 Reminiscence & social engagement",
]

EXERCISE_SCHEDULE_OPTIONS = [
    "08:30-09:30 Outdoor aerobic exercise",
    "15:30-16:30 Light indoor exercise: Dual-Task Training",
]

ADL_SCHEDULE_OPTIONS = [
    "07:30-08:30 Morning routine, hygiene & breakfast",
    "10:30-11:30 Household chores",
]

SOCIAL_SCHEDULE_OPTIONS = [
    "13:00-14:30 Reminiscence & social engagement",
    "14:30-15:30 Fine motor skills & art therapy",
]

HOME_SAFETY_SCHEDULE_OPTIONS = [
    "09:00-09:30 Weekly home safety walkthrough",
]

DAILY_LIFE_SCHEDULE_OPTIONS = [
    "07:30-08:30 Morning routine, hygiene & breakfast",
    "11:30-13:00 Lunch & midday rest",
    "14:00 Midday medication reminder",
    "16:30-18:00 Relaxation & dinner",
    "19:30-21:30 Bedtime prep & relaxation",
]

STAGE_LABELS = {"mild": "Mild", "moderate": "Moderate", "severe": "Severe"}

MOBILITY_LABELS = {
    MobilityLevel.INDEPENDENT: "Independent",
    MobilityLevel.NEEDS_SUPERVISION: "Needs supervision",
    MobilityLevel.USES_ASSISTIVE_DEVICE: "Uses assistive device",
}

ADL_MOBILITY_LABELS = {
    ADLMobilityLevel.INDEPENDENT: "Independent",
    ADLMobilityLevel.NEEDS_SUPERVISION: "Needs supervision",
    ADLMobilityLevel.USES_ASSISTIVE_DEVICE: "Uses assistive device",
}

EXERCISE_TYPE_OPTIONS = ["walking", "tai chi", "baduanjin", "square dancing"]

ADL_TASK_OPTIONS = [
    "brushing teeth",
    "getting dressed",
    "making a warm drink",
    "folding clothes",
    "wiping the table",
    "tidying the kitchen counter",
]

SOCIAL_FINE_MOTOR_LABELS = {
    FineMotorLevel.INDEPENDENT: "Independent",
    FineMotorLevel.MILD_TREMOR: "Mild tremor",
    FineMotorLevel.NEEDS_ASSIST: "Needs assist",
}

CRAFT_OPTIONS = ["bead stringing", "origami", "watering plants", "drawing", "paper cutting"]

HOME_SAFETY_MOBILITY_LABELS = {
    HomeSafetyMobilityLevel.INDEPENDENT: "Independent",
    HomeSafetyMobilityLevel.NEEDS_SUPERVISION: "Needs supervision",
    HomeSafetyMobilityLevel.USES_ASSISTIVE_DEVICE: "Uses assistive device",
}

HOME_SAFETY_ROOM_OPTIONS = ["Kitchen", "Bathroom", "Bedroom", "Hallway"]

HOME_SAFETY_RISK_LABELS = {
    WanderingRiskLevel.LOW: "Low",
    WanderingRiskLevel.MODERATE: "Moderate",
    WanderingRiskLevel.HIGH: "High",
}

HOME_SAFETY_HAZARD_LABELS = {
    HazardType.FIRE_GAS: "Fire / gas alarm",
    HazardType.FALL: "Fall detected",
    HazardType.WANDERING: "Wandering / exit-door breach",
    HazardType.MEDICATION_CHEMICAL_ACCESS: "Unauthorized medication/chemical access",
}

DAILY_LIFE_MOBILITY_LABELS = {
    DailyLifeMobilityLevel.INDEPENDENT: "Independent",
    DailyLifeMobilityLevel.NEEDS_SUPERVISION: "Needs supervision",
    DailyLifeMobilityLevel.USES_ASSISTIVE_DEVICE: "Uses assistive device",
}

DAILY_LIFE_CHECKPOINT_LABELS = {
    CheckpointType.WAKE_UP: "Wake-up",
    CheckpointType.DRESSING: "Dressing",
    CheckpointType.MEAL: "Meal",
    CheckpointType.MEDICATION: "Medication",
    CheckpointType.WALK: "Walk",
    CheckpointType.BEDTIME: "Bedtime",
}

DAILY_LIFE_DEVIATION_LABELS = {
    DeviationType.SCHEDULE_SLIP: "Schedule slip",
    DeviationType.ENVIRONMENT_CHANGE: "Environment change",
}

EMOTIONAL_SCHEDULE_OPTIONS = [
    "16:00-16:30 Sundowning prevention window",
]

EMOTIONAL_RISK_LABELS = {
    SundowningRiskLevel.LOW: "Low",
    SundowningRiskLevel.MODERATE: "Moderate",
    SundowningRiskLevel.HIGH: "High",
}

PROVIDER_LABELS = {
    "ollama": "Ollama (local, llama3.3)",
    "gemini": "Gemini (cloud, needs GOOGLE_API_KEY)",
}

st.set_page_config(page_title="MemoGo Training Assistant", page_icon="🧩", layout="centered")

st.markdown(
    """
    <style>
    .stApp { font-size: 20px; }
    .big-prompt {
        font-size: 30px;
        font-weight: 600;
        line-height: 1.6;
        padding: 1.5rem;
        background-color: #f0f4ff;
        color: #1a1a2e;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    div.stButton > button, div.stFormSubmitButton > button {
        font-size: 20px;
        padding: 0.6rem 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_cognitive_agent(provider: str) -> CognitiveBrainTrainingAgent:
    return CognitiveBrainTrainingAgent(provider=provider)


@st.cache_resource
def get_exercise_agent(provider: str) -> ExerciseMotorCoordinationTrainingAgent:
    return ExerciseMotorCoordinationTrainingAgent(provider=provider)


@st.cache_resource
def get_adl_agent(provider: str) -> ADLTrainingAgent:
    return ADLTrainingAgent(provider=provider)


@st.cache_resource
def get_social_agent(provider: str) -> SocialCreativeEngagementTrainingAgent:
    return SocialCreativeEngagementTrainingAgent(provider=provider)


@st.cache_resource
def get_home_safety_agent(provider: str) -> HomeSafetyProtectionAgent:
    return HomeSafetyProtectionAgent(provider=provider)


@st.cache_resource
def get_daily_life_agent(provider: str) -> DailyLifeRoutineManagementAgent:
    return DailyLifeRoutineManagementAgent(provider=provider)


@st.cache_resource
def get_emotional_agent(provider: str) -> EmotionalSupportComfortAgent:
    return EmotionalSupportComfortAgent(provider=provider)


def get_agent(agent_kind: str, provider: str):
    if agent_kind == "cognitive":
        return get_cognitive_agent(provider)
    if agent_kind == "exercise":
        return get_exercise_agent(provider)
    if agent_kind == "adl":
        return get_adl_agent(provider)
    if agent_kind == "home_safety":
        return get_home_safety_agent(provider)
    if agent_kind == "daily_life":
        return get_daily_life_agent(provider)
    if agent_kind == "emotional_support":
        return get_emotional_agent(provider)
    return get_social_agent(provider)


def init_state() -> None:
    st.session_state.setdefault("agent_kind", None)
    st.session_state.setdefault("thread_id", None)
    st.session_state.setdefault("step", None)
    st.session_state.setdefault("prompt_started_at", None)
    st.session_state.setdefault("provider", None)


def start_new_session(agent_kind: str, provider: str, patient, schedule_activity: str, max_turns: int, **kwargs) -> None:
    agent = get_agent(agent_kind, provider)
    try:
        thread_id, step = agent.start_session(
            patient=patient, schedule_activity=schedule_activity, max_turns=max_turns, **kwargs
        )
    except Exception as exc:
        st.error(f"Could not start the session: {exc}")
        return
    st.session_state["agent_kind"] = agent_kind
    st.session_state["provider"] = provider
    st.session_state["thread_id"] = thread_id
    st.session_state["step"] = step
    st.session_state["prompt_started_at"] = time.monotonic()


def start_incident_session(agent_kind: str, provider: str, patient, incident_event, max_turns: int) -> None:
    agent = get_agent(agent_kind, provider)
    try:
        thread_id, step = agent.start_incident_session(
            patient=patient, incident_event=incident_event, max_turns=max_turns
        )
    except Exception as exc:
        st.error(f"Could not start the session: {exc}")
        return
    st.session_state["agent_kind"] = agent_kind
    st.session_state["provider"] = provider
    st.session_state["thread_id"] = thread_id
    st.session_state["step"] = step
    st.session_state["prompt_started_at"] = time.monotonic()


def start_deviation_session(agent_kind: str, provider: str, patient, deviation_event, max_turns: int) -> None:
    agent = get_agent(agent_kind, provider)
    try:
        thread_id, step = agent.start_deviation_session(
            patient=patient, deviation_event=deviation_event, max_turns=max_turns
        )
    except Exception as exc:
        st.error(f"Could not start the session: {exc}")
        return
    st.session_state["agent_kind"] = agent_kind
    st.session_state["provider"] = provider
    st.session_state["thread_id"] = thread_id
    st.session_state["step"] = step
    st.session_state["prompt_started_at"] = time.monotonic()


def start_outburst_session(agent_kind: str, provider: str, patient, outburst_event, max_turns: int) -> None:
    agent = get_agent(agent_kind, provider)
    try:
        thread_id, step = agent.start_outburst_session(
            patient=patient, outburst_event=outburst_event, max_turns=max_turns
        )
    except Exception as exc:
        st.error(f"Could not start the session: {exc}")
        return
    st.session_state["agent_kind"] = agent_kind
    st.session_state["provider"] = provider
    st.session_state["thread_id"] = thread_id
    st.session_state["step"] = step
    st.session_state["prompt_started_at"] = time.monotonic()


def submit_response(response_text: str) -> bool:
    agent = get_agent(st.session_state["agent_kind"], st.session_state["provider"])
    latency = time.monotonic() - st.session_state["prompt_started_at"]
    try:
        step = agent.submit_response(st.session_state["thread_id"], response_text, latency)
    except Exception as exc:
        st.error(f"Could not process the response, please try submitting again: {exc}")
        return False
    st.session_state["step"] = step
    st.session_state["prompt_started_at"] = time.monotonic()
    return True


def reset_session() -> None:
    st.session_state["agent_kind"] = None
    st.session_state["thread_id"] = None
    st.session_state["step"] = None
    st.session_state["prompt_started_at"] = None


def _gemini_ready(provider: str) -> bool:
    if provider == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
        st.error("Set GOOGLE_API_KEY in the .env file to use Gemini.")
        return False
    return True


def render_cognitive_fields(name: str, stage_value: str, provider: str) -> None:
    biography = st.text_area(
        "Biography",
        value="Retired schoolteacher, raised three children, loves gardening and opera.",
    )
    preferences = st.text_input("Preferences (comma-separated)", value="gardening, classic songs")
    schedule_activity = st.selectbox("Current schedule activity", options=COGNITIVE_SCHEDULE_OPTIONS, index=1)
    max_turns = st.slider("Number of turns", min_value=1, max_value=8, value=3)

    if st.button("Start new session", type="primary", use_container_width=True):
        if _gemini_ready(provider):
            patient = CognitivePatientProfile(
                patient_id="gui-patient",
                name=name,
                stage=CognitiveDiseaseStage(stage_value),
                biography=biography,
                preferences=[p.strip() for p in preferences.split(",") if p.strip()],
            )
            start_new_session("cognitive", provider, patient, schedule_activity, max_turns)


def render_exercise_fields(name: str, stage_value: str, provider: str) -> None:
    mobility_label = st.selectbox("Mobility level", options=list(MOBILITY_LABELS.values()), index=0)
    mobility_level = next(m for m, label in MOBILITY_LABELS.items() if label == mobility_label)
    limitations = st.text_input("Physical limitations (comma-separated)", value="")
    preferred_types = st.multiselect(
        "Preferred exercise types", options=EXERCISE_TYPE_OPTIONS, default=["walking", "tai chi"]
    )
    indoor_outdoor = st.selectbox("Indoor/outdoor preference", options=["either", "indoor", "outdoor"], index=0)
    schedule_activity = st.selectbox("Current schedule activity", options=EXERCISE_SCHEDULE_OPTIONS, index=0)
    max_turns = st.slider("Number of turns", min_value=1, max_value=6, value=3)
    target_duration_min = st.slider("Target duration (minutes)", min_value=10, max_value=45, value=30, step=5)

    if st.button("Start new session", type="primary", use_container_width=True):
        if _gemini_ready(provider):
            patient = ExercisePatientProfile(
                patient_id="gui-patient",
                name=name,
                stage=ExerciseDiseaseStage(stage_value),
                mobility_level=mobility_level,
                physical_limitations=[l.strip() for l in limitations.split(",") if l.strip()],
                preferred_exercise_types=preferred_types,
                indoor_outdoor_preference=indoor_outdoor,
            )
            start_new_session(
                "exercise", provider, patient, schedule_activity, max_turns, target_duration_min=target_duration_min
            )


def render_adl_fields(name: str, stage_value: str, provider: str) -> None:
    mobility_label = st.selectbox("Mobility level", options=list(ADL_MOBILITY_LABELS.values()), index=0)
    mobility_level = next(m for m, label in ADL_MOBILITY_LABELS.items() if label == mobility_label)
    limitations = st.text_input("Physical limitations (comma-separated)", value="")
    preferred_tasks = st.multiselect("Preferred ADL tasks", options=ADL_TASK_OPTIONS, default=["folding clothes"])
    schedule_activity = st.selectbox("Current schedule activity", options=ADL_SCHEDULE_OPTIONS, index=0)
    max_turns = st.slider("Number of turns", min_value=1, max_value=10, value=6)

    if st.button("Start new session", type="primary", use_container_width=True):
        if _gemini_ready(provider):
            patient = ADLPatientProfile(
                patient_id="gui-patient",
                name=name,
                stage=ADLDiseaseStage(stage_value),
                mobility_level=mobility_level,
                physical_limitations=[l.strip() for l in limitations.split(",") if l.strip()],
                preferred_adl_tasks=preferred_tasks,
            )
            start_new_session("adl", provider, patient, schedule_activity, max_turns)


def render_social_fields(name: str, stage_value: str, provider: str) -> None:
    fine_motor_label = st.selectbox("Fine motor level", options=list(SOCIAL_FINE_MOTOR_LABELS.values()), index=0)
    fine_motor_level = next(m for m, label in SOCIAL_FINE_MOTOR_LABELS.items() if label == fine_motor_label)
    sensitivities = st.text_input("Material sensitivities (comma-separated)", value="")
    preferred_crafts = st.multiselect("Preferred crafts/horticulture", options=CRAFT_OPTIONS, default=["bead stringing"])
    preferred_songs = st.text_input("Preferred song themes (comma-separated)", value="folk songs")
    contact_name = st.text_input("Social contact name", value="Mrs. Lee")
    contact_relationship = st.text_input("Social contact relationship", value="neighbor")
    schedule_activity = st.selectbox("Current schedule activity", options=SOCIAL_SCHEDULE_OPTIONS, index=1)
    max_turns = st.slider("Number of turns", min_value=1, max_value=8, value=3)

    if st.button("Start new session", type="primary", use_container_width=True):
        if _gemini_ready(provider):
            social_contacts = (
                [SocialContact(name=contact_name, relationship=contact_relationship)] if contact_name else []
            )
            patient = SocialPatientProfile(
                patient_id="gui-patient",
                name=name,
                stage=SocialDiseaseStage(stage_value),
                fine_motor_level=fine_motor_level,
                material_sensitivities=[s.strip() for s in sensitivities.split(",") if s.strip()],
                preferred_crafts=preferred_crafts,
                preferred_songs=[s.strip() for s in preferred_songs.split(",") if s.strip()],
                social_contacts=social_contacts,
            )
            start_new_session("social", provider, patient, schedule_activity, max_turns)


def render_home_safety_fields(name: str, stage_value: str, provider: str) -> None:
    mobility_label = st.selectbox("Mobility level", options=list(HOME_SAFETY_MOBILITY_LABELS.values()), index=0)
    mobility_level = next(m for m, label in HOME_SAFETY_MOBILITY_LABELS.items() if label == mobility_label)
    limitations = st.text_input("Physical limitations (comma-separated)", value="")
    home_rooms = st.multiselect("Rooms in the home", options=HOME_SAFETY_ROOM_OPTIONS, default=HOME_SAFETY_ROOM_OPTIONS)
    risk_label = st.selectbox("Current wandering-risk level", options=list(HOME_SAFETY_RISK_LABELS.values()), index=0)
    wandering_risk_level = next(r for r, label in HOME_SAFETY_RISK_LABELS.items() if label == risk_label)
    emergency_contacts = st.text_input("Emergency contacts (comma-separated)", value="Daughter Amy")

    session_type = st.radio(
        "Session type", options=["Scheduled home safety audit", "Simulate a hazard/wandering event"], index=0
    )

    if session_type == "Scheduled home safety audit":
        schedule_activity = st.selectbox("Current schedule activity", options=HOME_SAFETY_SCHEDULE_OPTIONS, index=0)
        max_turns = st.slider("Number of checklist items", min_value=1, max_value=12, value=6)

        if st.button("Start new session", type="primary", use_container_width=True):
            if _gemini_ready(provider):
                patient = HomeSafetyPatientProfile(
                    patient_id="gui-patient",
                    name=name,
                    stage=HomeSafetyDiseaseStage(stage_value),
                    mobility_level=mobility_level,
                    physical_limitations=[l.strip() for l in limitations.split(",") if l.strip()],
                    home_rooms=home_rooms,
                    wandering_risk_level=wandering_risk_level,
                    emergency_contacts=[c.strip() for c in emergency_contacts.split(",") if c.strip()],
                )
                start_new_session("home_safety", provider, patient, schedule_activity, max_turns)
    else:
        hazard_label = st.selectbox("Hazard type", options=list(HOME_SAFETY_HAZARD_LABELS.values()), index=0)
        hazard_type = next(h for h, label in HOME_SAFETY_HAZARD_LABELS.items() if label == hazard_label)
        location = st.text_input("Location", value="Kitchen")
        source_device = st.text_input("Source device", value="Smoke detector")
        description = st.text_input("Device description (optional)", value="")

        if st.button("Simulate event", type="primary", use_container_width=True):
            if _gemini_ready(provider):
                patient = HomeSafetyPatientProfile(
                    patient_id="gui-patient",
                    name=name,
                    stage=HomeSafetyDiseaseStage(stage_value),
                    mobility_level=mobility_level,
                    physical_limitations=[l.strip() for l in limitations.split(",") if l.strip()],
                    home_rooms=home_rooms,
                    wandering_risk_level=wandering_risk_level,
                    emergency_contacts=[c.strip() for c in emergency_contacts.split(",") if c.strip()],
                )
                incident_event = IncidentEvent(
                    hazard_type=hazard_type, location=location, source_device=source_device, description=description
                )
                start_incident_session("home_safety", provider, patient, incident_event, max_turns=2)


def render_daily_life_fields(name: str, stage_value: str, provider: str) -> None:
    mobility_label = st.selectbox("Mobility level", options=list(DAILY_LIFE_MOBILITY_LABELS.values()), index=0)
    mobility_level = next(m for m, label in DAILY_LIFE_MOBILITY_LABELS.items() if label == mobility_label)
    limitations = st.text_input("Physical limitations (comma-separated)", value="")
    dietary_restrictions = st.text_input("Dietary restrictions (comma-separated)", value="easy to chew")
    seasonal_outfits = st.text_input(
        "Seasonal outfit set (comma-separated)",
        value="blue zip-up cardigan and pants, grey velcro-strap tracksuit",
    )
    medication_names = st.text_input("Daily medications (comma-separated)", value="Donepezil")

    session_type = st.radio(
        "Session type", options=["Scheduled routine checkpoint", "Report a routine deviation"], index=0
    )

    if session_type == "Scheduled routine checkpoint":
        schedule_activity = st.selectbox("Current schedule activity", options=DAILY_LIFE_SCHEDULE_OPTIONS, index=0)
        max_turns = st.slider("Number of checkpoints", min_value=1, max_value=6, value=2)

        if st.button("Start new session", type="primary", use_container_width=True):
            if _gemini_ready(provider):
                patient = DailyLifePatientProfile(
                    patient_id="gui-patient",
                    name=name,
                    stage=DailyLifeDiseaseStage(stage_value),
                    mobility_level=mobility_level,
                    physical_limitations=[l.strip() for l in limitations.split(",") if l.strip()],
                    dietary_restrictions=[d.strip() for d in dietary_restrictions.split(",") if d.strip()],
                    seasonal_outfit_set=[o.strip() for o in seasonal_outfits.split(",") if o.strip()],
                    medication_names=[m.strip() for m in medication_names.split(",") if m.strip()],
                )
                start_new_session("daily_life", provider, patient, schedule_activity, max_turns)
    else:
        deviation_label = st.selectbox("Deviation type", options=list(DAILY_LIFE_DEVIATION_LABELS.values()), index=0)
        deviation_type = next(d for d, label in DAILY_LIFE_DEVIATION_LABELS.items() if label == deviation_label)
        checkpoint_label = st.selectbox(
            "Affected checkpoint", options=list(DAILY_LIFE_CHECKPOINT_LABELS.values()), index=0
        )
        checkpoint_type = next(c for c, label in DAILY_LIFE_CHECKPOINT_LABELS.items() if label == checkpoint_label)
        description = st.text_input("Description (optional)", value="")

        if st.button("Report deviation", type="primary", use_container_width=True):
            if _gemini_ready(provider):
                patient = DailyLifePatientProfile(
                    patient_id="gui-patient",
                    name=name,
                    stage=DailyLifeDiseaseStage(stage_value),
                    mobility_level=mobility_level,
                    physical_limitations=[l.strip() for l in limitations.split(",") if l.strip()],
                    dietary_restrictions=[d.strip() for d in dietary_restrictions.split(",") if d.strip()],
                    seasonal_outfit_set=[o.strip() for o in seasonal_outfits.split(",") if o.strip()],
                    medication_names=[m.strip() for m in medication_names.split(",") if m.strip()],
                )
                deviation_event = DeviationEvent(
                    deviation_type=deviation_type, checkpoint_type=checkpoint_type, description=description
                )
                start_deviation_session("daily_life", provider, patient, deviation_event, max_turns=2)


def render_emotional_fields(name: str, stage_value: str, provider: str) -> None:
    risk_label = st.selectbox("Sundowning risk level", options=list(EMOTIONAL_RISK_LABELS.values()), index=0)
    sundowning_risk_level = next(r for r, label in EMOTIONAL_RISK_LABELS.items() if label == risk_label)
    known_triggers = st.text_input("Known triggers (comma-separated)", value="unfamiliar visitors")
    calming_preferences = st.text_input(
        "Calming preferences (comma-separated)", value="soft music from the 1960s, family photo album"
    )
    accepts_physical_contact = st.checkbox("Accepts light touch/hand-holding", value=True)
    independence_tasks = st.text_input("Independence tasks (comma-separated)", value="folding clothes, watering plants")
    emergency_contacts = st.text_input("Emergency contacts (comma-separated)", value="Daughter Amy")

    session_type = st.radio(
        "Session type", options=["Scheduled sundowning prevention", "Simulate an emotional outburst"], index=0
    )

    if session_type == "Scheduled sundowning prevention":
        schedule_activity = st.selectbox("Current schedule activity", options=EMOTIONAL_SCHEDULE_OPTIONS, index=0)
        max_turns = st.slider("Number of prevention steps", min_value=1, max_value=6, value=6)

        if st.button("Start new session", type="primary", use_container_width=True):
            if _gemini_ready(provider):
                patient = EmotionalPatientProfile(
                    patient_id="gui-patient",
                    name=name,
                    stage=EmotionalDiseaseStage(stage_value),
                    sundowning_risk_level=sundowning_risk_level,
                    known_triggers=[t.strip() for t in known_triggers.split(",") if t.strip()],
                    calming_preferences=[c.strip() for c in calming_preferences.split(",") if c.strip()],
                    accepts_physical_contact=accepts_physical_contact,
                    independence_tasks=[t.strip() for t in independence_tasks.split(",") if t.strip()],
                    emergency_contacts=[c.strip() for c in emergency_contacts.split(",") if c.strip()],
                )
                start_new_session("emotional_support", provider, patient, schedule_activity, max_turns)
    else:
        preceding_event = st.text_input("Preceding event", value="Unfamiliar visitor arrived at dusk")
        symptoms = st.text_input("Symptoms observed (comma-separated)", value="pacing, raised voice")
        environmental_factor = st.text_input("Environmental factor", value="Room was dim and noisy")
        food_or_drink_intake = st.text_input("Recent food/drink intake", value="Coffee after 4pm")
        max_turns = st.slider("Max de-escalation attempts", min_value=1, max_value=5, value=3)

        if st.button("Simulate outburst", type="primary", use_container_width=True):
            if _gemini_ready(provider):
                patient = EmotionalPatientProfile(
                    patient_id="gui-patient",
                    name=name,
                    stage=EmotionalDiseaseStage(stage_value),
                    sundowning_risk_level=sundowning_risk_level,
                    known_triggers=[t.strip() for t in known_triggers.split(",") if t.strip()],
                    calming_preferences=[c.strip() for c in calming_preferences.split(",") if c.strip()],
                    accepts_physical_contact=accepts_physical_contact,
                    independence_tasks=[t.strip() for t in independence_tasks.split(",") if t.strip()],
                    emergency_contacts=[c.strip() for c in emergency_contacts.split(",") if c.strip()],
                )
                outburst_event = OutburstEvent(
                    preceding_event=preceding_event,
                    symptoms_observed=[s.strip() for s in symptoms.split(",") if s.strip()],
                    environmental_factor=environmental_factor,
                    food_or_drink_intake=food_or_drink_intake,
                )
                start_outburst_session("emotional_support", provider, patient, outburst_event, max_turns=max_turns)


def render_sidebar() -> str:
    with st.sidebar:
        st.header("Training Category")
        agent_label = st.selectbox("Agent", options=list(AGENT_LABELS.values()), index=0)
        agent_kind = next(k for k, label in AGENT_LABELS.items() if label == agent_label)

        st.header("LLM Provider")
        provider_label = st.selectbox("Model", options=list(PROVIDER_LABELS.values()), index=0)
        provider = next(p for p, label in PROVIDER_LABELS.items() if label == provider_label)

        st.header("Patient Profile")
        name = st.text_input("Name", value="Grandma Chen")
        stage_label = st.selectbox("Disease stage", options=list(STAGE_LABELS.values()), index=0)
        stage_value = next(v for v, label in STAGE_LABELS.items() if label == stage_label)

        if agent_kind == "cognitive":
            render_cognitive_fields(name, stage_value, provider)
        elif agent_kind == "exercise":
            render_exercise_fields(name, stage_value, provider)
        elif agent_kind == "adl":
            render_adl_fields(name, stage_value, provider)
        elif agent_kind == "home_safety":
            render_home_safety_fields(name, stage_value, provider)
        elif agent_kind == "daily_life":
            render_daily_life_fields(name, stage_value, provider)
        elif agent_kind == "emotional_support":
            render_emotional_fields(name, stage_value, provider)
        else:
            render_social_fields(name, stage_value, provider)

    return agent_kind


def render_active_turn(prompt: str) -> None:
    st.markdown(f'<div class="big-prompt">{prompt}</div>', unsafe_allow_html=True)
    with st.form("response_form", clear_on_submit=True):
        response_text = st.text_input("Patient's / caregiver's response", key="response_input")
        submitted = st.form_submit_button("Submit response", type="primary", use_container_width=True)
    if submitted and response_text:
        if submit_response(response_text):
            st.rerun()


def render_completed_session(step) -> None:
    alert = (
        getattr(step, "safety_alert", None)
        or getattr(step, "hazard_alert", None)
        or getattr(step, "urgent_alert", None)
        or getattr(step, "routine_alert", None)
    )
    if alert:
        st.error(f"⚠️ Safety alert: {alert}")
    else:
        st.success("Session complete!")

    for i, turn in enumerate(step.session_log.turns, start=1):
        response = getattr(turn, "patient_response", None) or getattr(turn, "caregiver_response", None)
        with st.container(border=True):
            st.markdown(f"**Turn {i}**")
            st.write(f"Prompt: {turn.prompt}")
            st.write(f"Response: {response}")
            st.write(f"Feedback: {turn.feedback}")

    if not alert and step.caregiver_summary:
        st.subheader("Caregiver summary")
        st.write(step.caregiver_summary)

    if st.button("Start another session"):
        reset_session()
        st.rerun()


def main() -> None:
    init_state()

    selected_kind = render_sidebar()
    active_kind = st.session_state["agent_kind"] or selected_kind
    st.title(AGENT_LABELS[active_kind])

    step = st.session_state["step"]
    if step is None:
        st.info("Fill in the patient profile on the left, then click “Start new session”.")
        return

    if not step.done:
        render_active_turn(step.prompt)
    else:
        render_completed_session(step)


if __name__ == "__main__":
    main()
