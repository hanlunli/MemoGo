from langchain_ollama import ChatOllama
from langgraph.types import Command

from agents.companionship import modules
from agents.companionship.graph import build_session_graph
from agents.companionship.models import DiseaseStage, PatientProfile, RealityDistortionEvent
from agents.companionship.prompts import (
    DistortionValidationAssessment,
    ReminiscenceEngagementAssessment,
    StageActivityAssessment,
)


def _make_patient() -> PatientProfile:
    return PatientProfile(patient_id="t1", name="Test", stage=DiseaseStage.MILD)


def _scheduled_start_state(max_turns: int = 8) -> dict:
    return {
        "patient": _make_patient(),
        "trigger_type": "scheduled_companionship",
        "schedule_activity": "13:00-14:30 Reminiscence & social engagement",
        "max_turns": max_turns,
        "turn_count": 0,
    }


def _distortion_start_state(max_turns: int = 2) -> dict:
    return {
        "patient": _make_patient(),
        "trigger_type": "event_driven",
        "distortion_event": RealityDistortionEvent(patient_statement="I want to go home now"),
        "max_turns": max_turns,
        "turn_count": 0,
    }


def _stub_activity_prompt(self, patient, step):
    return "Please try this activity together."


def _stub_nudge_prompt(self, patient, nudge_type):
    return "Please remember to rest this week too."


def _stub_reminiscence_step(self, patient, step_index, theme, box):
    return f"Reminiscence step {step_index} instruction."


def _stub_distortion_prompt(self, distortion_type, attempt_number, patient_statement):
    return "1. Validate her feelings. 2. Sit with her for a while."


def _raise(*args, **kwargs):
    raise RuntimeError("simulated LLM failure")


def test_scheduled_session_survives_caregiver_summary_failure(monkeypatch):
    monkeypatch.setattr(modules.StageActivityEngine, "generate", _stub_activity_prompt)
    monkeypatch.setattr(
        modules.StageActivityAssessmentLayer,
        "generate",
        lambda self, prompt, response: StageActivityAssessment(engaged=True, feedback="Lovely, she joined in!"),
    )
    monkeypatch.setattr(modules.CaregiverReporter, "summarize", _raise)

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-scheduled-summary"}}

    result = graph.invoke(_scheduled_start_state(max_turns=1), config=config)
    assert result.get("__interrupt__")

    result = graph.invoke(Command(resume={"response": "She enjoyed the walk", "latency": 2.0}), config=config)

    assert not result.get("__interrupt__")
    assert result["session_log"].turns[0].feedback == "Lovely, she joined in!"
    assert result["caregiver_summary"] is None
    assert result["caregiver_alert"] is None


def test_scheduled_session_survives_stage_activity_assessment_failure(monkeypatch):
    monkeypatch.setattr(modules.StageActivityEngine, "generate", _stub_activity_prompt)
    monkeypatch.setattr(modules.StageActivityAssessmentLayer, "generate", _raise)
    monkeypatch.setattr(modules.CaregiverReporter, "summarize", lambda self, session_json: "All good.")

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-scheduled-activity"}}

    graph.invoke(_scheduled_start_state(max_turns=1), config=config)
    result = graph.invoke(Command(resume={"response": "Not really engaged", "latency": 2.0}), config=config)

    assert result["session_log"].turns[0].feedback == "Thank you for spending this time together."
    assert result["session_log"].turns[0].item_addressed is None
    assert result["session_log"].activity_status_counts.get("pending") == 1
    assert result["caregiver_summary"] == "All good."


def test_scheduled_session_runs_reminiscence_subflow_and_self_care_nudge(monkeypatch):
    monkeypatch.setattr(modules.StageActivityEngine, "generate", _stub_activity_prompt)
    monkeypatch.setattr(
        modules.StageActivityAssessmentLayer,
        "generate",
        lambda self, prompt, response: StageActivityAssessment(engaged=True, feedback="Nice moment together."),
    )
    monkeypatch.setattr(modules.ReminiscenceSessionEngine, "generate_step", _stub_reminiscence_step)
    monkeypatch.setattr(
        modules.ReminiscenceEngagementAssessmentLayer,
        "generate",
        lambda self, prompt, response: ReminiscenceEngagementAssessment(
            engaged=True, feedback="She lit up remembering that."
        ),
    )
    monkeypatch.setattr(modules.SelfCareNudgeEngine, "generate", _stub_nudge_prompt)
    monkeypatch.setattr(modules.CaregiverReporter, "summarize", lambda self, session_json: "A full session done.")

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-scheduled-full"}}

    result = graph.invoke(_scheduled_start_state(max_turns=8), config=config)
    # Turn 1: low_intensity_exercise -> submit; turns 2-5: the 4 reminiscence steps -> submit each;
    # turn 6: light_chores -> submit; turn 7: self-care nudge -> submit; then session closes.
    responses = [
        "Enjoyed the walk",
        "Looked at the photos with interest",
        "Sitting close together now",
        "She shared a lovely story",
        "Smiling and nodding along",
        "Folded a few towels together",
        "Yes, I booked respite care for this week",
    ]
    for response_text in responses:
        assert result.get("__interrupt__")
        result = graph.invoke(Command(resume={"response": response_text, "latency": 3.0}), config=config)

    assert not result.get("__interrupt__")
    session_log = result["session_log"]
    assert session_log.reminiscence_session_conducted is True
    assert session_log.self_care_nudge_delivered is not None
    assert session_log.self_care_nudge_accepted is True
    assert session_log.ended_reason == "companionship_completed"
    assert len(session_log.turns) == 7
    assert result["caregiver_summary"] == "A full session done."


def test_max_turns_cap_is_honored_even_mid_reminiscence_subflow(monkeypatch):
    monkeypatch.setattr(modules.StageActivityEngine, "generate", _stub_activity_prompt)
    monkeypatch.setattr(
        modules.StageActivityAssessmentLayer,
        "generate",
        lambda self, prompt, response: StageActivityAssessment(engaged=True, feedback="Nice moment together."),
    )
    monkeypatch.setattr(modules.ReminiscenceSessionEngine, "generate_step", _stub_reminiscence_step)
    monkeypatch.setattr(
        modules.ReminiscenceEngagementAssessmentLayer,
        "generate",
        lambda self, prompt, response: ReminiscenceEngagementAssessment(engaged=True, feedback="Lovely."),
    )
    monkeypatch.setattr(modules.CaregiverReporter, "summarize", lambda self, session_json: "Session summary.")

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-max-turns-mid-reminiscence"}}

    # Step 0 is low_intensity_exercise, step 1 is the 4-step reminiscence sub-flow for a MILD
    # patient — with max_turns=2, the cap should bite after the exercise turn plus one
    # reminiscence sub-step, well before all 4 reminiscence steps run.
    result = graph.invoke(_scheduled_start_state(max_turns=2), config=config)
    assert result.get("__interrupt__")
    result = graph.invoke(Command(resume={"response": "Enjoyed the walk", "latency": 3.0}), config=config)
    assert result.get("__interrupt__")
    result = graph.invoke(
        Command(resume={"response": "Looked at the photos with interest", "latency": 3.0}), config=config
    )

    assert not result.get("__interrupt__")
    session_log = result["session_log"]
    assert len(session_log.turns) == 2
    assert session_log.ended_reason == "max_turns_reached"
    assert session_log.reminiscence_session_conducted is False


def test_distortion_session_settles_and_summarizes(monkeypatch):
    monkeypatch.setattr(modules.DistortionResponseCoach, "generate", _stub_distortion_prompt)
    monkeypatch.setattr(
        modules.DistortionValidationAssessmentLayer,
        "generate",
        lambda self, prompt, response: DistortionValidationAssessment(settled=True, feedback="Great, she's settled."),
    )
    monkeypatch.setattr(modules.CaregiverReporter, "summarize", lambda self, session_json: "All calm now.")

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-distortion-settled"}}

    graph.invoke(_distortion_start_state(max_turns=2), config=config)
    result = graph.invoke(
        Command(resume={"response": "She agreed and sat down, much calmer now", "latency": 8.0}), config=config
    )

    assert result["session_log"].ended_reason == "validated_and_settled"
    assert result["caregiver_summary"] == "All calm now."
    assert result["caregiver_alert"] is None


def test_distortion_session_escalates_after_attempts_exhausted(monkeypatch):
    monkeypatch.setattr(modules.DistortionResponseCoach, "generate", _stub_distortion_prompt)
    monkeypatch.setattr(
        modules.DistortionValidationAssessmentLayer,
        "generate",
        lambda self, prompt, response: DistortionValidationAssessment(
            settled=False, feedback="Let's try the next step."
        ),
    )
    monkeypatch.setattr(modules.CaregiverReporter, "unresolved_alert", lambda self, session_json: "Consider more support.")

    graph = build_session_graph(ChatOllama(model="llama3.3"))
    config = {"configurable": {"thread_id": "resilience-test-distortion-escalate"}}

    graph.invoke(_distortion_start_state(max_turns=1), config=config)
    result = graph.invoke(Command(resume={"response": "Still insists on leaving", "latency": 5.0}), config=config)

    assert result["session_log"].ended_reason == "unresolved_after_attempts"
    assert result["caregiver_alert"] == "Consider more support."
    assert result["caregiver_summary"] is None
