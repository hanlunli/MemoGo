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
from agents.exercise_motor_coordination_training import ExerciseMotorCoordinationTrainingAgent, MobilityLevel
from agents.exercise_motor_coordination_training import DiseaseStage as ExerciseDiseaseStage
from agents.exercise_motor_coordination_training import PatientProfile as ExercisePatientProfile
from agents.exercise_motor_coordination_training.llm_logging import configure_logging as configure_exercise_logging
from agents.social_creative_engagement_training import FineMotorLevel, SocialContact, SocialCreativeEngagementTrainingAgent
from agents.social_creative_engagement_training import DiseaseStage as SocialDiseaseStage
from agents.social_creative_engagement_training import PatientProfile as SocialPatientProfile
from agents.social_creative_engagement_training.llm_logging import configure_logging as configure_social_logging

load_dotenv()
configure_cognitive_logging()
configure_exercise_logging()
configure_adl_logging()
configure_social_logging()

AGENT_LABELS = {
    "cognitive": "🧠 Cognitive & Brain Training",
    "exercise": "🏃 Exercise & Motor Coordination Training",
    "adl": "🧺 Activities of Daily Living (ADL) Training",
    "social": "🎨 Social & Creative Engagement Training",
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


def get_agent(agent_kind: str, provider: str):
    if agent_kind == "cognitive":
        return get_cognitive_agent(provider)
    if agent_kind == "exercise":
        return get_exercise_agent(provider)
    if agent_kind == "adl":
        return get_adl_agent(provider)
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
    )
    if alert:
        st.error(f"⚠️ Safety alert: {alert}")
    else:
        st.success("Session complete!")

    for i, turn in enumerate(step.session_log.turns, start=1):
        with st.container(border=True):
            st.markdown(f"**Turn {i}**")
            st.write(f"Prompt: {turn.prompt}")
            st.write(f"Response: {turn.patient_response}")
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
