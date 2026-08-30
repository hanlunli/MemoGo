from agents.emotional_support_comfort.models import (
    DiseaseStage,
    HabitType,
    OutburstEvent,
    OutburstSeverity,
    PatientProfile,
    RedirectionType,
)
from agents.emotional_support_comfort.modules import (
    assess_calming_signal,
    build_prevention_sequence,
    classify_outburst_severity,
    matches_known_trigger,
    resolve_checklist_status,
    select_independence_task,
    select_redirection,
)


def _make_patient(**overrides) -> PatientProfile:
    defaults = dict(patient_id="t1", name="Test", stage=DiseaseStage.MILD)
    defaults.update(overrides)
    return PatientProfile(**defaults)


def test_build_prevention_sequence_surfaces_high_priority_habits_first():
    items = build_prevention_sequence(_make_patient())
    habit_items = [item for item in items if item.habit_type != HabitType.INDEPENDENCE_TASK_OFFER]
    priorities = [item.is_high_priority for item in habit_items]
    first_low_priority_index = priorities.index(False)
    assert all(priorities[:first_low_priority_index])


def test_build_prevention_sequence_appends_independence_task_last():
    items = build_prevention_sequence(_make_patient())
    assert items[-1].habit_type == HabitType.INDEPENDENCE_TASK_OFFER


def test_select_independence_task_uses_patients_preferred_list():
    patient = _make_patient(independence_tasks=["folding towels", "watering plants"])
    assert select_independence_task(patient) == "folding towels"


def test_select_independence_task_falls_back_when_empty():
    patient = _make_patient(independence_tasks=[])
    assert select_independence_task(patient) == "watering the plants"


def test_classify_outburst_severity_no_symptoms_is_restlessness():
    assert classify_outburst_severity(OutburstEvent()) == OutburstSeverity.RESTLESSNESS


def test_classify_outburst_severity_mild_symptoms_are_restlessness():
    event = OutburstEvent(symptoms_observed=["pacing", "muttering"])
    assert classify_outburst_severity(event) == OutburstSeverity.RESTLESSNESS


def test_classify_outburst_severity_high_severity_keyword_is_outburst():
    event = OutburstEvent(symptoms_observed=["hitting the caregiver"])
    assert classify_outburst_severity(event) == OutburstSeverity.OUTBURST


def test_classify_outburst_severity_unrecognized_symptom_defaults_to_outburst():
    event = OutburstEvent(symptoms_observed=["crying loudly"])
    assert classify_outburst_severity(event) == OutburstSeverity.OUTBURST


def test_assess_calming_signal_no_response_is_ambiguous():
    assert assess_calming_signal(None, 0.0) is None


def test_assess_calming_signal_slow_response_is_ambiguous():
    assert assess_calming_signal("she is calm now", 120.0) is None


def test_assess_calming_signal_ack_keyword_is_calmed():
    assert assess_calming_signal("She's settled and listening to the music", 10.0) is True


def test_assess_calming_signal_no_ack_keyword_is_not_calmed():
    assert assess_calming_signal("Still upset, won't stop crying", 10.0) is False


def test_assess_calming_signal_unrelated_reply_is_ambiguous():
    assert assess_calming_signal("ok", 5.0) is None


def test_select_redirection_prefers_patient_preferences():
    patient = _make_patient(calming_preferences=["soft music from the 1960s"])
    assert select_redirection(patient, used=[]) == RedirectionType.MUSIC


def test_select_redirection_skips_already_used_types():
    patient = _make_patient(calming_preferences=["soft music", "family photos"])
    first = select_redirection(patient, used=[])
    second = select_redirection(patient, used=[first])
    assert second != first


def test_matches_known_trigger_flags_a_matching_preceding_event():
    patient = _make_patient(known_triggers=["unfamiliar visitors"])
    event = OutburstEvent(preceding_event="An unfamiliar visitor arrived at the door")
    assert matches_known_trigger(event, patient) == "unfamiliar visitors"


def test_matches_known_trigger_none_when_no_overlap():
    patient = _make_patient(known_triggers=["unfamiliar visitors"])
    event = OutburstEvent(preceding_event="Woke up from a nap")
    assert matches_known_trigger(event, patient) is None


def test_resolve_checklist_status():
    assert resolve_checklist_status(True) == "done"
    assert resolve_checklist_status(False) == "needs_attention"
    assert resolve_checklist_status(None) == "pending"
