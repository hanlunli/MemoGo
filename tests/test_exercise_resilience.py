from langchain_ollama import ChatOllama
from langgraph.types import Command

from agents.exercise_motor_coordination_training import modules
from agents.exercise_motor_coordination_training.graph import build_session_graph
from agents.exercise_motor_coordination_training.models import DiseaseStage, PatientProfile
from agents.exercise_motor_coordination_training.prompts import MotorResponseAssessment


def _make_patient() -> PatientProfile:
    return PatientProfile(patient_id="t1", name="Test", stage=DiseaseStage.MILD)


def _start_state() -> dict:
    return {
        "patient": _make_patient(),
        "schedule_activity": "08:30-09:30 Outdoor aerobic exercise",
        "max_turns": 1,
        "turn_count": 0,
        "target_duration_min": 30,
        "intensity": 2,
        "fatigue_detected": False,
    }


def _stub_instruction(self, patient, exercise_type, phase, intensity):
    return "Please take a few steps forward."


def _raise(*args, **kwargs):
    raise RuntimeError("simulated LLM failure")


def test_session_survives_caregiver_summary_failure(monkeypatch):
    monkeypatch.setattr(modules.AerobicExerciseSessionEngine, "generate_instruction", _stub_instruction)
    monkeypatch.setattr(
        modules.FeedbackEncouragementLayer,
        "generate",
        lambda self, prompt, has_cognitive_task, response: MotorResponseAssessment(
            sustained=True, feedback="Nice!"
        ),
    )
    monkeypatch.setattr(modules.CaregiverReporter, "summarize", _raise)

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-summary"}}

    result = graph.invoke(_start_state(), config=config)
    assert result.get("__interrupt__")

    result = graph.invoke(Command(resume={"response": "Walked fine", "latency": 2.0}), config=config)

    assert not result.get("__interrupt__")
    assert result["session_log"].turns[0].feedback == "Nice!"
    assert result["caregiver_summary"] is None
    assert result["safety_alert"] is None


def test_session_survives_feedback_failure(monkeypatch):
    monkeypatch.setattr(modules.AerobicExerciseSessionEngine, "generate_instruction", _stub_instruction)
    monkeypatch.setattr(modules.FeedbackEncouragementLayer, "generate", _raise)
    monkeypatch.setattr(modules.CaregiverReporter, "summarize", lambda self, session_json: "All good.")

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-feedback"}}

    graph.invoke(_start_state(), config=config)
    result = graph.invoke(Command(resume={"response": "Walked fine", "latency": 2.0}), config=config)

    assert result["session_log"].turns[0].feedback == "Great effort, thank you!"
    assert result["session_log"].turns[0].sustained is None
    assert result["caregiver_summary"] == "All good."


def test_session_triggers_safety_alert_on_fall(monkeypatch):
    monkeypatch.setattr(modules.AerobicExerciseSessionEngine, "generate_instruction", _stub_instruction)
    monkeypatch.setattr(
        modules.FeedbackEncouragementLayer,
        "generate",
        lambda self, prompt, has_cognitive_task, response: MotorResponseAssessment(
            sustained=False, feedback="That's okay, let's take it easy."
        ),
    )
    monkeypatch.setattr(
        modules.CaregiverReporter, "safety_alert", lambda self, session_json: "Please check on them now."
    )

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-safety"}}

    graph.invoke(_start_state(), config=config)
    result = graph.invoke(Command(resume={"response": "I fell down", "latency": 2.0}), config=config)

    assert result["session_log"].safety_incident is True
    assert result["session_log"].ended_reason == "safety_incident"
    assert result["safety_alert"] == "Please check on them now."
    assert result["caregiver_summary"] is None
