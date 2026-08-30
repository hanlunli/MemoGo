from langchain_ollama import ChatOllama
from langgraph.types import Command

from agents.social_creative_engagement_training import modules
from agents.social_creative_engagement_training.graph import build_session_graph
from agents.social_creative_engagement_training.models import DiseaseStage, PatientProfile
from agents.social_creative_engagement_training.prompts import ParticipationAssessment


def _make_patient() -> PatientProfile:
    return PatientProfile(patient_id="t1", name="Test", stage=DiseaseStage.MILD)


def _start_state() -> dict:
    return {
        "patient": _make_patient(),
        "schedule_activity": "14:30-15:30 Fine motor skills & art therapy",
        "activity_type": "drawing",
        "max_turns": 1,
        "turn_count": 0,
        "complexity": 2,
        "consecutive_agitation_count": 0,
        "agitation_detected": False,
    }


def _stub_instruction(self, patient, activity_type, step_description, is_hazard_step):
    return "Please pick up the crayon."


def _stub_music_instruction(self, patient, memory, participation_mode):
    return "Let's listen to a familiar song together."


def _raise(*args, **kwargs):
    raise RuntimeError("simulated LLM failure")


def test_session_survives_caregiver_summary_failure(monkeypatch):
    monkeypatch.setattr(modules.CraftsHorticultureActivityEngine, "generate", _stub_instruction)
    monkeypatch.setattr(
        modules.FeedbackEncouragementLayer,
        "generate",
        lambda self, prompt, module, response: ParticipationAssessment(participated=True, feedback="Nice!"),
    )
    monkeypatch.setattr(modules.CaregiverReporter, "summarize", _raise)

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-summary"}}

    result = graph.invoke(_start_state(), config=config)
    assert result.get("__interrupt__")

    result = graph.invoke(Command(resume={"response": "Drew a sun", "latency": 2.0}), config=config)

    assert not result.get("__interrupt__")
    assert result["session_log"].turns[0].feedback == "Nice!"
    assert result["caregiver_summary"] is None
    assert result["urgent_alert"] is None


def test_music_therapy_turn_persists_the_session_item(monkeypatch):
    monkeypatch.setattr(modules.MusicTherapySessionEngine, "generate_instruction", _stub_music_instruction)
    monkeypatch.setattr(
        modules.FeedbackEncouragementLayer,
        "generate",
        lambda self, prompt, module, response: ParticipationAssessment(participated=True, feedback="Lovely!"),
    )
    monkeypatch.setattr(modules.CaregiverReporter, "summarize", lambda self, session_json: "All good.")

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-music-session-item"}}

    start_state = dict(_start_state())
    start_state["schedule_activity"] = "Afternoon music therapy sing-along"

    graph.invoke(start_state, config=config)
    result = graph.invoke(Command(resume={"response": "Singing along happily", "latency": 2.0}), config=config)

    session_item = result["session_log"].turns[0].session_item
    assert session_item is not None
    assert session_item.theme == "folk songs"
    assert session_item.decade == "1960s"


def test_session_survives_feedback_failure(monkeypatch):
    monkeypatch.setattr(modules.CraftsHorticultureActivityEngine, "generate", _stub_instruction)
    monkeypatch.setattr(modules.FeedbackEncouragementLayer, "generate", _raise)
    monkeypatch.setattr(modules.CaregiverReporter, "summarize", lambda self, session_json: "All good.")

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-feedback"}}

    graph.invoke(_start_state(), config=config)
    result = graph.invoke(Command(resume={"response": "Drew a sun", "latency": 2.0}), config=config)

    assert result["session_log"].turns[0].feedback == "Thank you for joining in!"
    assert result["session_log"].turns[0].participation_confirmed is None
    assert result["caregiver_summary"] == "All good."


def test_session_triggers_urgent_alert_on_material_hazard(monkeypatch):
    monkeypatch.setattr(modules.CraftsHorticultureActivityEngine, "generate", _stub_instruction)
    monkeypatch.setattr(
        modules.FeedbackEncouragementLayer,
        "generate",
        lambda self, prompt, module, response: ParticipationAssessment(
            participated=False, feedback="That's okay, let's take it easy."
        ),
    )
    monkeypatch.setattr(
        modules.CaregiverReporter, "urgent_alert", lambda self, session_json: "Please check on them now."
    )

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-hazard"}}

    graph.invoke(_start_state(), config=config)
    result = graph.invoke(Command(resume={"response": "I cut my finger", "latency": 2.0}), config=config)

    assert result["session_log"].hazard_incident is True
    assert result["session_log"].ended_reason == "hazard_incident"
    assert result["urgent_alert"] == "Please check on them now."
    assert result["caregiver_summary"] is None
