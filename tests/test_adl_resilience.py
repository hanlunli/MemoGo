from langchain_ollama import ChatOllama
from langgraph.types import Command

from agents.adl_training import modules
from agents.adl_training.graph import build_session_graph
from agents.adl_training.models import DiseaseStage, PatientProfile
from agents.adl_training.prompts import StepAssessment


def _make_patient() -> PatientProfile:
    return PatientProfile(patient_id="t1", name="Test", stage=DiseaseStage.MILD)


def _start_state() -> dict:
    return {
        "patient": _make_patient(),
        "schedule_activity": "10:30-11:30 Household chores",
        "max_turns": 1,
        "turn_count": 0,
        "consecutive_frustration_count": 0,
    }


def _stub_instruction(self, patient, task_name, step_description, assistance_level, is_hazard_step):
    return "Please pick up the cloth."


def _raise(*args, **kwargs):
    raise RuntimeError("simulated LLM failure")


def test_session_survives_caregiver_summary_failure(monkeypatch):
    monkeypatch.setattr(modules.StepInstructionEngine, "generate", _stub_instruction)
    monkeypatch.setattr(
        modules.FeedbackEncouragementLayer,
        "generate",
        lambda self, prompt, assistance_level, response: StepAssessment(completed=True, feedback="Nice!"),
    )
    monkeypatch.setattr(modules.CaregiverReporter, "summarize", _raise)

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-summary"}}

    result = graph.invoke(_start_state(), config=config)
    assert result.get("__interrupt__")

    result = graph.invoke(Command(resume={"response": "Folded one shirt", "latency": 2.0}), config=config)

    assert not result.get("__interrupt__")
    assert result["session_log"].turns[0].feedback == "Nice!"
    assert result["caregiver_summary"] is None
    assert result["hazard_alert"] is None


def test_session_survives_feedback_failure(monkeypatch):
    monkeypatch.setattr(modules.StepInstructionEngine, "generate", _stub_instruction)
    monkeypatch.setattr(modules.FeedbackEncouragementLayer, "generate", _raise)
    monkeypatch.setattr(modules.CaregiverReporter, "summarize", lambda self, session_json: "All good.")

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-feedback"}}

    graph.invoke(_start_state(), config=config)
    result = graph.invoke(Command(resume={"response": "Folded one shirt", "latency": 2.0}), config=config)

    assert result["session_log"].turns[0].feedback == "Great effort, thank you!"
    assert result["session_log"].turns[0].step_completed is None
    assert result["caregiver_summary"] == "All good."


def test_session_triggers_hazard_alert_on_burn(monkeypatch):
    monkeypatch.setattr(modules.StepInstructionEngine, "generate", _stub_instruction)
    monkeypatch.setattr(
        modules.FeedbackEncouragementLayer,
        "generate",
        lambda self, prompt, assistance_level, response: StepAssessment(
            completed=False, feedback="That's okay, let's take it easy."
        ),
    )
    monkeypatch.setattr(
        modules.CaregiverReporter, "hazard_alert", lambda self, session_json: "Please check on them now."
    )

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-hazard"}}

    graph.invoke(_start_state(), config=config)
    result = graph.invoke(Command(resume={"response": "I burned my hand", "latency": 2.0}), config=config)

    assert result["session_log"].hazard_incident is True
    assert result["session_log"].ended_reason == "hazard_incident"
    assert result["hazard_alert"] == "Please check on them now."
    assert result["caregiver_summary"] is None
