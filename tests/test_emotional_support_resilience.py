from langchain_ollama import ChatOllama
from langgraph.types import Command

from agents.emotional_support_comfort import modules
from agents.emotional_support_comfort.graph import build_session_graph
from agents.emotional_support_comfort.models import DiseaseStage, OutburstEvent, PatientProfile
from agents.emotional_support_comfort.prompts import DeescalationAssessment, HabitAssessment


def _make_patient() -> PatientProfile:
    return PatientProfile(patient_id="t1", name="Test", stage=DiseaseStage.MILD)


def _prevention_start_state(max_turns: int = 1) -> dict:
    return {
        "patient": _make_patient(),
        "trigger_type": "scheduled_prevention",
        "schedule_activity": "16:00-16:30 Sundowning prevention window",
        "max_turns": max_turns,
        "turn_count": 0,
    }


def _outburst_start_state(max_turns: int = 2) -> dict:
    return {
        "patient": _make_patient(),
        "trigger_type": "event_driven",
        "outburst_event": OutburstEvent(preceding_event="unfamiliar visitor arrived", symptoms_observed=["crying"]),
        "max_turns": max_turns,
        "turn_count": 0,
    }


def _stub_habit_prompt(self, patient, item):
    return "Please confirm this prevention habit."


def _stub_redirection_content(self, patient, redirection_type):
    return "a familiar song from the 1960s"


def _stub_deescalation_prompt(self, **kwargs):
    return "1. Validate her feelings. 2. Play the song softly. 3. Speak in a calm, low voice."


def _raise(*args, **kwargs):
    raise RuntimeError("simulated LLM failure")


def test_prevention_session_survives_caregiver_summary_failure(monkeypatch):
    monkeypatch.setattr(modules.PreventionHabitEngine, "generate", _stub_habit_prompt)
    monkeypatch.setattr(
        modules.HabitAssessmentLayer,
        "generate",
        lambda self, prompt, response: HabitAssessment(addressed=True, feedback="Nice, that's done!"),
    )
    monkeypatch.setattr(modules.CaregiverReporter, "summarize", _raise)

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-prevention-summary"}}

    result = graph.invoke(_prevention_start_state(max_turns=1), config=config)
    assert result.get("__interrupt__")

    result = graph.invoke(Command(resume={"response": "Done, curtains are drawn", "latency": 2.0}), config=config)

    assert not result.get("__interrupt__")
    assert result["session_log"].turns[0].feedback == "Nice, that's done!"
    assert result["caregiver_summary"] is None
    assert result["safety_alert"] is None


def test_prevention_session_survives_habit_assessment_failure(monkeypatch):
    monkeypatch.setattr(modules.PreventionHabitEngine, "generate", _stub_habit_prompt)
    monkeypatch.setattr(modules.HabitAssessmentLayer, "generate", _raise)
    monkeypatch.setattr(modules.CaregiverReporter, "summarize", lambda self, session_json: "All good.")

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-prevention-habit"}}

    graph.invoke(_prevention_start_state(max_turns=1), config=config)
    result = graph.invoke(Command(resume={"response": "Not yet", "latency": 2.0}), config=config)

    assert result["session_log"].turns[0].feedback == "Thank you for checking on this."
    assert result["session_log"].turns[0].item_addressed is None
    assert result["session_log"].prevention_status_counts.get("pending") == 1
    assert result["caregiver_summary"] == "All good."


def test_outburst_session_deescalates_and_summarizes(monkeypatch):
    monkeypatch.setattr(modules.DeescalationCoach, "render_redirection_content", _stub_redirection_content)
    monkeypatch.setattr(modules.DeescalationCoach, "generate", _stub_deescalation_prompt)
    monkeypatch.setattr(
        modules.DeescalationAssessmentLayer,
        "generate",
        lambda self, prompt, response: DeescalationAssessment(calmed=True, feedback="Great, she's settled."),
    )
    monkeypatch.setattr(modules.CaregiverReporter, "summarize", lambda self, session_json: "All calm now.")

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-outburst-deescalated"}}

    graph.invoke(_outburst_start_state(max_turns=2), config=config)
    result = graph.invoke(
        Command(resume={"response": "She is settled and listening to the music now", "latency": 8.0}), config=config
    )

    assert result["session_log"].ended_reason == "deescalated"
    assert result["session_log"].outburst_active is False
    assert result["session_log"].escalated_to_contact is False
    assert result["caregiver_summary"] == "All calm now."
    assert result["safety_alert"] is None


def test_outburst_session_escalates_after_attempts_exhausted(monkeypatch):
    monkeypatch.setattr(modules.DeescalationCoach, "render_redirection_content", _stub_redirection_content)
    monkeypatch.setattr(modules.DeescalationCoach, "generate", _stub_deescalation_prompt)
    monkeypatch.setattr(
        modules.DeescalationAssessmentLayer,
        "generate",
        lambda self, prompt, response: DeescalationAssessment(calmed=False, feedback="Let's try the next step."),
    )
    monkeypatch.setattr(modules.CaregiverReporter, "urgent_alert", lambda self, session_json: "Escalating now.")

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-outburst-escalate"}}

    graph.invoke(_outburst_start_state(max_turns=1), config=config)
    result = graph.invoke(Command(resume={"response": "Still upset, won't stop crying", "latency": 5.0}), config=config)

    assert result["session_log"].ended_reason == "escalated_unresolved"
    assert result["session_log"].escalated_to_contact is True
    assert result["safety_alert"] == "Escalating now."
    assert result["caregiver_summary"] is None
    assert result["session_log"].journal_entry is not None
    assert result["session_log"].journal_entry.preceding_event == "unfamiliar visitor arrived"


def test_outburst_session_survives_deescalation_assessment_failure(monkeypatch):
    monkeypatch.setattr(modules.DeescalationCoach, "render_redirection_content", _stub_redirection_content)
    monkeypatch.setattr(modules.DeescalationCoach, "generate", _stub_deescalation_prompt)
    monkeypatch.setattr(modules.DeescalationAssessmentLayer, "generate", _raise)
    monkeypatch.setattr(modules.CaregiverReporter, "urgent_alert", lambda self, session_json: "Escalating now.")

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-outburst-assessment-failure"}}

    graph.invoke(_outburst_start_state(max_turns=1), config=config)
    result = graph.invoke(Command(resume={"response": "Still upset, won't stop crying", "latency": 5.0}), config=config)

    assert result["session_log"].turns[0].feedback == "Understood — keep going gently, I'm logging this."
    assert result["safety_alert"] == "Escalating now."
