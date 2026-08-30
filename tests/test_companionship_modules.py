from agents.companionship.models import (
    ActivityStepType,
    DiseaseStage,
    DistortionType,
    PatientProfile,
    SelfCareNudgeType,
)
from agents.companionship.modules import (
    assess_validation_signal,
    build_reminiscence_box,
    build_stage_activity_sequence,
    classify_distortion_type,
    detect_nudge_acceptance,
    detects_fatigue_signal,
    is_sensitive_topic,
    resolve_engagement_status,
    resolve_reminiscence_theme,
    select_self_care_nudge,
)


def _make_patient(**overrides) -> PatientProfile:
    defaults = dict(patient_id="t1", name="Test", stage=DiseaseStage.MILD)
    defaults.update(overrides)
    return PatientProfile(**defaults)


def test_build_stage_activity_sequence_uses_early_stage_template():
    steps = build_stage_activity_sequence(_make_patient(stage=DiseaseStage.MILD))
    assert steps[0].activity_key == "low_intensity_exercise"
    assert steps[1].step_type == ActivityStepType.REMINISCENCE_THERAPY


def test_build_stage_activity_sequence_appends_self_care_nudge_last():
    steps = build_stage_activity_sequence(_make_patient())
    assert steps[-1].step_type == ActivityStepType.SELF_CARE_NUDGE


def test_build_stage_activity_sequence_late_stage_has_no_reminiscence_step():
    steps = build_stage_activity_sequence(_make_patient(stage=DiseaseStage.SEVERE))
    assert all(step.step_type != ActivityStepType.REMINISCENCE_THERAPY for step in steps)


def test_select_self_care_nudge_defaults_to_boundary_reframe():
    assert select_self_care_nudge(_make_patient(days_since_last_respite=1)) == SelfCareNudgeType.EMOTIONAL_BOUNDARY_REFRAME


def test_select_self_care_nudge_switches_to_respite_reminder_when_overdue():
    assert select_self_care_nudge(_make_patient(days_since_last_respite=7)) == SelfCareNudgeType.RESPITE_REMINDER


def test_resolve_reminiscence_theme_uses_patient_background():
    patient = _make_patient(reminiscence_background=["her wedding in the countryside"])
    assert resolve_reminiscence_theme(patient) == "her wedding in the countryside"


def test_resolve_reminiscence_theme_avoids_sensitive_topics():
    patient = _make_patient(
        reminiscence_background=["losing her husband"], sensitive_topics_to_avoid=["losing her husband"]
    )
    theme = resolve_reminiscence_theme(patient)
    assert theme != "losing her husband"


def test_is_sensitive_topic_matches_case_insensitively():
    patient = _make_patient(sensitive_topics_to_avoid=["the war years"])
    assert is_sensitive_topic("The War Years", patient) is True


def test_build_reminiscence_box_mentions_theme_in_all_three_senses():
    box = build_reminiscence_box("her hometown market")
    assert "her hometown market" in box.visual
    assert "her hometown market" in box.auditory


def test_detects_fatigue_signal_true_on_keyword():
    assert detects_fatigue_signal("She looks tired and wants to stop") is True


def test_detects_fatigue_signal_false_without_keyword():
    assert detects_fatigue_signal("She is smiling and engaged") is False


def test_assess_validation_signal_no_response_is_ambiguous():
    assert assess_validation_signal(None, 0.0) is None


def test_assess_validation_signal_slow_response_is_ambiguous():
    assert assess_validation_signal("she settled down", 120.0) is None


def test_assess_validation_signal_ack_keyword_is_settled():
    assert assess_validation_signal("She agreed and is much calmer now", 10.0) is True


def test_assess_validation_signal_no_ack_keyword_is_not_settled():
    assert assess_validation_signal("Still insists on leaving", 10.0) is False


def test_classify_distortion_type_wants_to_go_home():
    assert classify_distortion_type("I need to go home now") == DistortionType.WANTS_TO_GO_HOME


def test_classify_distortion_type_misidentification():
    assert classify_distortion_type("Who are you? Where is my mother?") == DistortionType.MISIDENTIFICATION


def test_classify_distortion_type_accusation():
    assert classify_distortion_type("Someone stole my ring") == DistortionType.ACCUSATION_OR_SUSPICION


def test_classify_distortion_type_unrecognized_defaults_to_other():
    assert classify_distortion_type("I don't understand what's happening") == DistortionType.OTHER


def test_detect_nudge_acceptance_yes():
    assert detect_nudge_acceptance("Yes, I already booked respite care") is True


def test_detect_nudge_acceptance_no():
    assert detect_nudge_acceptance("No time for that this week") is False


def test_detect_nudge_acceptance_ambiguous():
    assert detect_nudge_acceptance("ok") is None


def test_resolve_engagement_status():
    assert resolve_engagement_status(True) == "done"
    assert resolve_engagement_status(False) == "needs_attention"
    assert resolve_engagement_status(None) == "pending"
