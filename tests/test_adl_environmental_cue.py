from langchain_ollama import ChatOllama
from langgraph.types import Command

from agents.adl_training import modules
from agents.adl_training.graph import build_session_graph
from agents.adl_training.models import DiseaseStage, PatientProfile


def _make_patient() -> PatientProfile:
    return PatientProfile(patient_id="t1", name="Test", stage=DiseaseStage.MILD)


def _fail_if_called(*args, **kwargs):
    raise AssertionError("feedback layer should not be called for the environmental cue turn")


def test_environmental_cue_turn_skips_assessment_and_advances(monkeypatch):
    monkeypatch.setattr(
        modules.EnvironmentalCueEngine,
        "generate",
        lambda self, patient, room_label: f"Let's find the door labeled {room_label}.",
    )
    monkeypatch.setattr(
        modules.StepInstructionEngine, "generate", lambda self, *args, **kwargs: "Pick up your toothbrush."
    )
    monkeypatch.setattr(modules.FeedbackEncouragementLayer, "generate", _fail_if_called)
    monkeypatch.setattr(modules.CaregiverReporter, "summarize", lambda self, session_json: "All good.")

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "env-cue-test"}}

    # "hygiene" (not "morning") routes to personal care without the orientation step, so the
    # environmental cue step (every personal-care task has a room label) is the very first turn.
    result = graph.invoke(
        {
            "patient": _make_patient(),
            "schedule_activity": "20:00-20:30 Evening hygiene",
            "max_turns": 5,
            "turn_count": 0,
            "consecutive_frustration_count": 0,
        },
        config=config,
    )
    interrupt = result["__interrupt__"][0]
    assert "door labeled" in interrupt.value["prompt"]

    result = graph.invoke(Command(resume={"response": "Found it!", "latency": 2.0}), config=config)

    first_turn = result["session_log"].turns[0]
    assert first_turn.feedback == "Great, you found it!"
    assert first_turn.step_completed is True
    assert result["session_log"].steps_completed == 0

    next_interrupt = result["__interrupt__"][0]
    assert next_interrupt.value["prompt"] == "Pick up your toothbrush."
