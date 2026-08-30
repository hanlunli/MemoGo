from langchain_ollama import ChatOllama
from langgraph.types import Command

from agents.daily_life_routine_management import modules
from agents.daily_life_routine_management.graph import build_session_graph
from agents.daily_life_routine_management.models import (
    CheckpointType,
    DeviationEvent,
    DeviationType,
    DiseaseStage,
    PatientProfile,
)
from agents.daily_life_routine_management.prompts import CheckpointAssessment, DeviationAssessment


def _make_patient() -> PatientProfile:
    return PatientProfile(patient_id="t1", name="Test", stage=DiseaseStage.MILD, medication_names=["Donepezil"])


def _checkpoint_start_state(schedule_activity: str = "19:30-21:30 Bedtime prep & relaxation") -> dict:
    return {
        "patient": _make_patient(),
        "trigger_type": "scheduled_checkpoint",
        "schedule_activity": schedule_activity,
        "max_turns": 1,
        "turn_count": 0,
        "consecutive_no_response_count": 0,
    }


def _deviation_start_state(deviation_type: DeviationType = DeviationType.SCHEDULE_SLIP) -> dict:
    return {
        "patient": _make_patient(),
        "trigger_type": "deviation_report",
        "deviation_event": DeviationEvent(deviation_type=deviation_type, checkpoint_type=CheckpointType.MEAL),
        "max_turns": 2,
        "turn_count": 0,
    }


def _stub_checkpoint_prompt(self, patient, checkpoint):
    return "Please confirm this checkpoint."


def _stub_deviation_prompt(self, event, severity):
    return "A routine deviation was reported. Please confirm."


def _raise(*args, **kwargs):
    raise RuntimeError("simulated LLM failure")


def test_checkpoint_session_survives_caregiver_summary_failure(monkeypatch):
    monkeypatch.setattr(modules.CheckpointDeliveryEngine, "generate", _stub_checkpoint_prompt)
    monkeypatch.setattr(
        modules.CheckpointAssessmentLayer,
        "generate",
        lambda self, prompt, response: CheckpointAssessment(resolved=True, feedback="Nice, that's done!"),
    )
    monkeypatch.setattr(modules.CaregiverReporter, "summarize", _raise)

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-checkpoint-summary"}}

    result = graph.invoke(_checkpoint_start_state(), config=config)
    assert result.get("__interrupt__")

    result = graph.invoke(Command(resume={"response": "All done", "latency": 2.0}), config=config)

    assert not result.get("__interrupt__")
    assert result["session_log"].turns[0].feedback == "Nice, that's done!"
    assert result["caregiver_summary"] is None
    assert result["routine_alert"] is None


def test_checkpoint_session_survives_assessment_failure(monkeypatch):
    monkeypatch.setattr(modules.CheckpointDeliveryEngine, "generate", _stub_checkpoint_prompt)
    monkeypatch.setattr(modules.CheckpointAssessmentLayer, "generate", _raise)
    monkeypatch.setattr(modules.CaregiverReporter, "summarize", lambda self, session_json: "All good.")

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-checkpoint-assessment"}}

    graph.invoke(_checkpoint_start_state(), config=config)
    result = graph.invoke(Command(resume={"response": "Not yet", "latency": 2.0}), config=config)

    assert result["session_log"].turns[0].feedback == "Thank you for confirming."
    assert result["session_log"].turns[0].checkpoint_resolved is None
    assert result["session_log"].checkpoint_status_counts.get("pending") == 1
    assert result["caregiver_summary"] == "All good."


def test_medication_checkpoint_requires_explicit_supervision_confirmation(monkeypatch):
    monkeypatch.setattr(modules.CheckpointDeliveryEngine, "generate", _stub_checkpoint_prompt)
    monkeypatch.setattr(
        modules.CheckpointAssessmentLayer,
        "generate",
        lambda self, prompt, response: CheckpointAssessment(resolved=True, feedback="Great, all set."),
    )
    monkeypatch.setattr(
        modules.CaregiverReporter, "routine_alert", lambda self, session_json: "Please supervise the next dose."
    )

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-medication-supervision"}}

    graph.invoke(_checkpoint_start_state("14:00 Medication reminder"), config=config)
    result = graph.invoke(Command(resume={"response": "Took the pill", "latency": 2.0}), config=config)

    assert result["session_log"].missed_medication_supervision is True
    assert result["session_log"].turns[0].checkpoint_resolved is False
    assert result["routine_alert"] == "Please supervise the next dose."


def test_deviation_session_flags_major_environment_change(monkeypatch):
    monkeypatch.setattr(modules.DeviationResponseEngine, "generate", _stub_deviation_prompt)
    monkeypatch.setattr(
        modules.DeviationAssessmentLayer,
        "generate",
        lambda self, prompt, response: DeviationAssessment(resolved=True, feedback="Understood, thank you."),
    )
    monkeypatch.setattr(modules.CaregiverReporter, "routine_alert", lambda self, session_json: "Please monitor closely.")

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-deviation-major"}}

    graph.invoke(_deviation_start_state(DeviationType.ENVIRONMENT_CHANGE), config=config)
    result = graph.invoke(Command(resume={"response": "Adjusted the routine", "latency": 3.0}), config=config)

    assert result["session_log"].major_deviation is True
    assert result["session_log"].ended_reason == "major_deviation"
    assert result["routine_alert"] == "Please monitor closely."
    assert result["caregiver_summary"] is None


def test_deviation_session_survives_assessment_failure(monkeypatch):
    monkeypatch.setattr(modules.DeviationResponseEngine, "generate", _stub_deviation_prompt)
    monkeypatch.setattr(modules.DeviationAssessmentLayer, "generate", _raise)
    monkeypatch.setattr(modules.CaregiverReporter, "summarize", lambda self, session_json: "Logged.")

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-deviation-assessment-failure"}}

    graph.invoke(_deviation_start_state(), config=config)
    result = graph.invoke(Command(resume={"response": "Handled it", "latency": 2.0}), config=config)

    assert result["session_log"].turns[0].feedback == "Thank you — this has been logged and flagged for follow-up."
