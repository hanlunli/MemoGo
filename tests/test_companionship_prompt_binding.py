"""Invokes every LLM-backed engine through a fake chat model so a prompt template's declared
variables are checked against what modules.py actually passes in -- a mismatch (like a template
referencing {name} that the caller never supplies) raises immediately instead of only surfacing
during a real end-to-end run."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agents.companionship.modules import (
    DistortionResponseCoach,
    ReminiscenceSessionEngine,
    SelfCareNudgeEngine,
    StageActivityEngine,
    build_reminiscence_box,
)
from agents.companionship.models import (
    ActivityStepType,
    DiseaseStage,
    DistortionType,
    PatientProfile,
    SelfCareNudgeType,
    StageActivityStep,
)


def _fake_llm() -> FakeListChatModel:
    return FakeListChatModel(responses=["stub response"] * 10)


def _make_patient(**overrides) -> PatientProfile:
    defaults = dict(patient_id="t1", name="Test", stage=DiseaseStage.MILD)
    defaults.update(overrides)
    return PatientProfile(**defaults)


def test_stage_activity_engine_binds_all_template_variables():
    step = StageActivityStep(step_type=ActivityStepType.STAGE_ACTIVITY, activity_key="light_chores", description="Fold clothes.")
    result = StageActivityEngine(_fake_llm()).generate(_make_patient(), step)
    assert result == "stub response"


def test_self_care_nudge_engine_binds_all_template_variables():
    result = SelfCareNudgeEngine(_fake_llm()).generate(_make_patient(), SelfCareNudgeType.RESPITE_REMINDER)
    assert result == "stub response"


def test_reminiscence_session_engine_binds_all_template_variables_for_every_step():
    patient = _make_patient()
    box = build_reminiscence_box("running the family shop")
    engine = ReminiscenceSessionEngine(_fake_llm())
    for step_index in range(4):
        result = engine.generate_step(patient, step_index, "running the family shop", box)
        assert result == "stub response"


def test_distortion_response_coach_binds_all_template_variables():
    result = DistortionResponseCoach(_fake_llm()).generate(
        DistortionType.WANTS_TO_GO_HOME, attempt_number=1, patient_statement="I need to go home"
    )
    assert result == "stub response"
