from langchain_ollama import ChatOllama
from langgraph.types import Command

from agents.cognitive_brain_training import modules
from agents.cognitive_brain_training.graph import build_session_graph
from agents.cognitive_brain_training.models import DiseaseStage, ExerciseItem, PatientProfile
from agents.cognitive_brain_training.prompts import ResponseAssessment


def _make_patient() -> PatientProfile:
    return PatientProfile(patient_id="t1", name="Test", stage=DiseaseStage.MILD)


def _start_state() -> dict:
    return {
        "patient": _make_patient(),
        "schedule_activity": "09:30-10:30 Targeted cognitive training",
        "memory_items": [],
        "max_turns": 1,
        "turn_count": 0,
        "difficulty": 2,
        "fatigue_detected": False,
    }


def _raise(*args, **kwargs):
    raise RuntimeError("simulated LLM failure")


def test_session_survives_caregiver_summary_failure(monkeypatch):
    monkeypatch.setattr(
        modules.CognitiveExerciseGenerator,
        "generate",
        lambda self, domain, difficulty, exercise_type: ExerciseItem(
            type=exercise_type, domain=domain, difficulty=difficulty, content="2+2?", answer="4"
        ),
    )
    monkeypatch.setattr(
        modules.FeedbackEncouragementLayer,
        "generate",
        lambda self, prompt, expected_answer, response: ResponseAssessment(
            correct=True, feedback="Nice!"
        ),
    )
    monkeypatch.setattr(modules.CaregiverReporter, "summarize", _raise)

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-summary"}}

    result = graph.invoke(_start_state(), config=config)
    assert result.get("__interrupt__")

    result = graph.invoke(Command(resume={"response": "4", "latency": 2.0}), config=config)

    assert not result.get("__interrupt__")
    assert result["session_log"].turns[0].feedback == "Nice!"
    assert result["caregiver_summary"] is None


def test_session_survives_feedback_failure(monkeypatch):
    monkeypatch.setattr(
        modules.CognitiveExerciseGenerator,
        "generate",
        lambda self, domain, difficulty, exercise_type: ExerciseItem(
            type=exercise_type, domain=domain, difficulty=difficulty, content="2+2?", answer="4"
        ),
    )
    monkeypatch.setattr(modules.FeedbackEncouragementLayer, "generate", _raise)
    monkeypatch.setattr(modules.CaregiverReporter, "summarize", lambda self, session_json: "All good.")

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-feedback"}}

    graph.invoke(_start_state(), config=config)
    result = graph.invoke(Command(resume={"response": "4", "latency": 2.0}), config=config)

    assert result["session_log"].turns[0].feedback == "Thank you for your answer!"
    assert result["session_log"].turns[0].correct is None
    assert result["caregiver_summary"] == "All good."
