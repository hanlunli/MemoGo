from langchain_ollama import ChatOllama
from langgraph.types import Command

from agents.home_safety_protection import modules
from agents.home_safety_protection.graph import build_session_graph
from agents.home_safety_protection.models import DiseaseStage, HazardType, IncidentEvent, PatientProfile
from agents.home_safety_protection.prompts import IncidentAssessment, ItemAssessment


def _make_patient() -> PatientProfile:
    return PatientProfile(patient_id="t1", name="Test", stage=DiseaseStage.MILD)


def _audit_start_state() -> dict:
    return {
        "patient": _make_patient(),
        "trigger_type": "scheduled_audit",
        "schedule_activity": "09:00-09:30 Weekly home safety walkthrough",
        "max_turns": 1,
        "turn_count": 0,
    }


def _incident_start_state(hazard_type: HazardType = HazardType.FIRE_GAS) -> dict:
    return {
        "patient": _make_patient(),
        "trigger_type": "event_driven",
        "incident_event": IncidentEvent(hazard_type=hazard_type, location="Kitchen", source_device="smoke detector"),
        "max_turns": 2,
        "turn_count": 0,
    }


def _stub_checklist_prompt(self, patient, item):
    return "Please confirm this checklist item."


def _stub_incident_prompt(self, event, severity):
    return "Emergency detected in the kitchen. Please confirm."


def _raise(*args, **kwargs):
    raise RuntimeError("simulated LLM failure")


def test_audit_session_survives_caregiver_summary_failure(monkeypatch):
    monkeypatch.setattr(modules.ChecklistItemEngine, "generate", _stub_checklist_prompt)
    monkeypatch.setattr(
        modules.ItemAssessmentLayer,
        "generate",
        lambda self, prompt, response: ItemAssessment(resolved=True, feedback="Nice, that's installed!"),
    )
    monkeypatch.setattr(modules.CaregiverReporter, "summarize", _raise)

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-audit-summary"}}

    result = graph.invoke(_audit_start_state(), config=config)
    assert result.get("__interrupt__")

    result = graph.invoke(Command(resume={"response": "Installed the grab bars", "latency": 2.0}), config=config)

    assert not result.get("__interrupt__")
    assert result["session_log"].turns[0].feedback == "Nice, that's installed!"
    assert result["caregiver_summary"] is None
    assert result["safety_alert"] is None


def test_audit_session_survives_item_assessment_failure(monkeypatch):
    monkeypatch.setattr(modules.ChecklistItemEngine, "generate", _stub_checklist_prompt)
    monkeypatch.setattr(modules.ItemAssessmentLayer, "generate", _raise)
    monkeypatch.setattr(modules.CaregiverReporter, "summarize", lambda self, session_json: "All good.")

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-audit-item"}}

    graph.invoke(_audit_start_state(), config=config)
    result = graph.invoke(Command(resume={"response": "Not yet installed", "latency": 2.0}), config=config)

    assert result["session_log"].turns[0].feedback == "Thank you for checking on this."
    assert result["session_log"].turns[0].item_resolved is None
    assert result["session_log"].checklist_status_counts.get("pending") == 1
    assert result["caregiver_summary"] == "All good."


def test_incident_session_triggers_emergency_alert(monkeypatch):
    monkeypatch.setattr(modules.IncidentTriageEngine, "generate", _stub_incident_prompt)
    monkeypatch.setattr(
        modules.IncidentAssessmentLayer,
        "generate",
        lambda self, prompt, response: IncidentAssessment(resolved=True, feedback="Understood, thank you."),
    )
    monkeypatch.setattr(
        modules.CaregiverReporter, "emergency_alert", lambda self, session_json: "Please check the kitchen now."
    )

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-incident-alert"}}

    graph.invoke(_incident_start_state(), config=config)
    result = graph.invoke(Command(resume={"response": "Turned off the stove", "latency": 3.0}), config=config)

    assert result["session_log"].emergency_incident is True
    assert result["session_log"].ended_reason == "emergency_incident"
    assert result["safety_alert"] == "Please check the kitchen now."
    assert result["caregiver_summary"] is None


def test_incident_session_escalates_to_contact_on_no_ack(monkeypatch):
    monkeypatch.setattr(modules.IncidentTriageEngine, "generate", _stub_incident_prompt)
    monkeypatch.setattr(
        modules.IncidentAssessmentLayer,
        "generate",
        lambda self, prompt, response: IncidentAssessment(resolved=False, feedback="Noted."),
    )
    monkeypatch.setattr(modules.CaregiverReporter, "emergency_alert", lambda self, session_json: "Escalating now.")

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-incident-escalate"}}

    graph.invoke(_incident_start_state(hazard_type=HazardType.WANDERING), config=config)
    result = graph.invoke(Command(resume={"response": "I can't get there right now", "latency": 5.0}), config=config)

    assert result["session_log"].escalated_to_contact is True
    assert result["session_log"].wandering_risk_level_after == "moderate"
    assert result["session_log"].device_recommendation is not None


def test_incident_session_survives_incident_assessment_failure(monkeypatch):
    monkeypatch.setattr(modules.IncidentTriageEngine, "generate", _stub_incident_prompt)
    monkeypatch.setattr(modules.IncidentAssessmentLayer, "generate", _raise)
    monkeypatch.setattr(modules.CaregiverReporter, "emergency_alert", lambda self, session_json: "Escalating now.")

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-incident-assessment-failure"}}

    graph.invoke(_incident_start_state(), config=config)
    result = graph.invoke(Command(resume={"response": "Handled it", "latency": 2.0}), config=config)

    assert result["session_log"].turns[0].feedback == "Thank you — this has been logged and flagged for follow-up."
    assert result["safety_alert"] == "Escalating now."
