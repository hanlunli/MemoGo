from agents.daily_life_routine_management.models import (
    CheckpointStatus,
    CheckpointType,
    DeviationSeverity,
    DeviationType,
    DiseaseStage,
    PatientProfile,
)
from agents.daily_life_routine_management.modules import (
    assess_deviation_acknowledgment,
    assess_medication_supervision,
    build_checkpoint_sequence,
    classify_deviation_severity,
    detect_double_meal_request,
    resolve_checkpoint_status,
    resolve_checkpoints,
    room_label_for_checkpoint,
)


def _make_patient(**overrides) -> PatientProfile:
    defaults = dict(patient_id="t1", name="Test", stage=DiseaseStage.MILD)
    defaults.update(overrides)
    return PatientProfile(**defaults)


def test_resolve_checkpoints_matches_keywords():
    assert resolve_checkpoints("07:30-08:30 Morning routine, hygiene & breakfast") == [
        CheckpointType.WAKE_UP,
        CheckpointType.DRESSING,
        CheckpointType.MEAL,
    ]


def test_resolve_checkpoints_defaults_to_wake_up():
    assert resolve_checkpoints("08:30-09:30 Outdoor aerobic exercise") == [CheckpointType.WAKE_UP]


def test_build_checkpoint_sequence_covers_every_requested_checkpoint():
    patient = _make_patient()
    checkpoints = build_checkpoint_sequence(patient, [CheckpointType.DRESSING, CheckpointType.MEAL])
    types = [c.checkpoint_type for c in checkpoints]
    assert types == [CheckpointType.DRESSING, CheckpointType.MEAL]


def test_build_checkpoint_sequence_offers_at_most_two_outfit_options():
    patient = _make_patient(seasonal_outfit_set=["outfit a", "outfit b", "outfit c"])
    checkpoints = build_checkpoint_sequence(patient, [CheckpointType.DRESSING])
    assert len(checkpoints[0].choice_options) == 2


def test_build_checkpoint_sequence_meal_offers_drink_choice():
    patient = _make_patient()
    checkpoints = build_checkpoint_sequence(patient, [CheckpointType.MEAL])
    assert checkpoints[0].choice_options == ["water", "tea"]


def test_build_checkpoint_sequence_medication_uses_patient_medication_names():
    patient = _make_patient(medication_names=["Donepezil"])
    checkpoints = build_checkpoint_sequence(patient, [CheckpointType.MEDICATION])
    assert checkpoints[0].choice_options == ["Donepezil"]


def test_room_label_for_checkpoint_reuses_adl_labels():
    assert room_label_for_checkpoint(CheckpointType.DRESSING) == "Bedroom"
    assert room_label_for_checkpoint(CheckpointType.MEAL) == "Kitchen"
    assert room_label_for_checkpoint(CheckpointType.WALK) is None


def test_classify_deviation_severity():
    assert classify_deviation_severity(DeviationType.ENVIRONMENT_CHANGE) == DeviationSeverity.MAJOR
    assert classify_deviation_severity(DeviationType.SCHEDULE_SLIP) == DeviationSeverity.MODERATE


def test_resolve_checkpoint_status():
    assert resolve_checkpoint_status(True) == CheckpointStatus.COMPLETED
    assert resolve_checkpoint_status(False) == CheckpointStatus.NEEDS_ATTENTION
    assert resolve_checkpoint_status(None) == CheckpointStatus.PENDING


def test_assess_medication_supervision_requires_explicit_confirmation():
    assert assess_medication_supervision("Gave the medication just now") is True
    assert assess_medication_supervision("ok") is False
    assert assess_medication_supervision(None) is False


def test_detect_double_meal_request():
    assert detect_double_meal_request("She already ate but wants more food") is True
    assert detect_double_meal_request("Just served lunch") is False


def test_assess_deviation_acknowledgment_no_response_needs_followup():
    acknowledged, needs_followup = assess_deviation_acknowledgment(None, 0.0)
    assert acknowledged is False
    assert needs_followup is True


def test_assess_deviation_acknowledgment_refusal_needs_followup():
    acknowledged, needs_followup = assess_deviation_acknowledgment("I don't know what happened", 5.0)
    assert acknowledged is False
    assert needs_followup is True


def test_assess_deviation_acknowledgment_slow_response_needs_followup():
    acknowledged, needs_followup = assess_deviation_acknowledgment("Adjusted the routine", 120.0)
    assert acknowledged is False
    assert needs_followup is True


def test_assess_deviation_acknowledgment_prompt_reply_is_acknowledged():
    acknowledged, needs_followup = assess_deviation_acknowledgment("Handled, rescheduled the walk", 5.0)
    assert acknowledged is True
    assert needs_followup is False


def test_assess_deviation_acknowledgment_vague_reply_still_needs_followup():
    acknowledged, needs_followup = assess_deviation_acknowledgment("ok", 5.0)
    assert acknowledged is False
    assert needs_followup is True
