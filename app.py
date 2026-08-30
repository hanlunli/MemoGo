import os
import time

import streamlit as st
from dotenv import load_dotenv

from agents.cognitive_brain_training import CognitiveBrainTrainingAgent, DiseaseStage, PatientProfile
from agents.cognitive_brain_training.llm_logging import configure_logging

load_dotenv()
configure_logging()

SCHEDULE_OPTIONS = [
    "07:30-08:30 Reality orientation morning check-in",
    "09:30-10:30 Targeted cognitive training",
    "13:00-14:30 Reminiscence & social engagement",
]

STAGE_LABELS = {
    DiseaseStage.MILD: "Mild",
    DiseaseStage.MODERATE: "Moderate",
    DiseaseStage.SEVERE: "Severe",
}

PROVIDER_LABELS = {
    "ollama": "Ollama (local, llama3.3)",
    "gemini": "Gemini (cloud, needs GOOGLE_API_KEY)",
}

st.set_page_config(page_title="Cognitive & Brain Training Assistant", page_icon="🧠", layout="centered")

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
def get_agent(provider: str) -> CognitiveBrainTrainingAgent:
    return CognitiveBrainTrainingAgent(provider=provider)


def init_state() -> None:
    st.session_state.setdefault("thread_id", None)
    st.session_state.setdefault("step", None)
    st.session_state.setdefault("prompt_started_at", None)
    st.session_state.setdefault("provider", None)


def start_new_session(
    provider: str, patient: PatientProfile, schedule_activity: str, max_turns: int
) -> None:
    agent = get_agent(provider)
    try:
        thread_id, step = agent.start_session(
            patient=patient, schedule_activity=schedule_activity, max_turns=max_turns
        )
    except Exception as exc:
        st.error(f"Could not start the session: {exc}")
        return
    st.session_state["provider"] = provider
    st.session_state["thread_id"] = thread_id
    st.session_state["step"] = step
    st.session_state["prompt_started_at"] = time.monotonic()


def submit_response(response_text: str) -> bool:
    agent = get_agent(st.session_state["provider"])
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
    st.session_state["thread_id"] = None
    st.session_state["step"] = None
    st.session_state["prompt_started_at"] = None


def render_sidebar() -> None:
    with st.sidebar:
        st.header("LLM Provider")
        provider_label = st.selectbox("Model", options=list(PROVIDER_LABELS.values()), index=0)
        provider = next(p for p, label in PROVIDER_LABELS.items() if label == provider_label)

        st.header("Patient Profile")
        name = st.text_input("Name", value="Grandma Chen")
        stage_label = st.selectbox("Disease stage", options=list(STAGE_LABELS.values()), index=0)
        stage = next(s for s, label in STAGE_LABELS.items() if label == stage_label)
        biography = st.text_area(
            "Biography",
            value="Retired schoolteacher, raised three children, loves gardening and opera.",
        )
        preferences = st.text_input("Preferences (comma-separated)", value="gardening, classic songs")
        schedule_activity = st.selectbox("Current schedule activity", options=SCHEDULE_OPTIONS, index=1)
        max_turns = st.slider("Number of turns", min_value=1, max_value=8, value=3)

        if st.button("Start new session", type="primary", use_container_width=True):
            if provider == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
                st.error("Set GOOGLE_API_KEY in the .env file to use Gemini.")
            else:
                patient = PatientProfile(
                    patient_id="gui-patient",
                    name=name,
                    stage=stage,
                    biography=biography,
                    preferences=[p.strip() for p in preferences.split(",") if p.strip()],
                )
                start_new_session(provider, patient, schedule_activity, max_turns)


def render_active_turn(prompt: str) -> None:
    st.markdown(f'<div class="big-prompt">{prompt}</div>', unsafe_allow_html=True)
    with st.form("response_form", clear_on_submit=True):
        response_text = st.text_input("Patient's answer", key="response_input")
        submitted = st.form_submit_button("Submit answer", type="primary", use_container_width=True)
    if submitted and response_text:
        if submit_response(response_text):
            st.rerun()


def render_completed_session(step) -> None:
    st.success("Session complete!")
    for i, turn in enumerate(step.session_log.turns, start=1):
        with st.container(border=True):
            st.markdown(f"**Turn {i}**")
            st.write(f"Question: {turn.prompt}")
            st.write(f"Answer: {turn.patient_response}")
            st.write(f"Feedback: {turn.feedback}")

    if step.caregiver_summary:
        st.subheader("Caregiver summary")
        st.write(step.caregiver_summary)

    if st.button("Start another session"):
        reset_session()
        st.rerun()


def main() -> None:
    init_state()

    st.title("🧠 Cognitive & Brain Training Assistant")

    render_sidebar()

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
