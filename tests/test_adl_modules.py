from agents.adl_training.models import AssistanceLevel, IncidentFlag, ModuleId
from agents.adl_training.modules import (
    ENVIRONMENTAL_CUE_STEP,
    ORIENTATION_STEP,
    adjust_assistance_level,
    assess_frustration,
    build_step_sequence,
    detect_hazard_incident,
    enforce_hazard_floor,
    is_safety_stop,
    resolve_module,
)


def test_resolve_module_matches_keywords():
    assert resolve_module("10:30-11:30 Household chores") == ModuleId.HOUSEHOLD_CHORES
    assert resolve_module("07:30-08:30 Morning routine, hygiene & breakfast") == ModuleId.PERSONAL_CARE


def test_resolve_module_defaults_to_personal_care():
    assert resolve_module("16:30-18:00 Relaxation & dinner") == ModuleId.PERSONAL_CARE


def test_build_step_sequence_prepends_environmental_cue_for_tasks_with_a_room():
    steps = build_step_sequence(ModuleId.PERSONAL_CARE, "brushing teeth", "10:30-11:30 Any time")
    assert steps[0] == (ENVIRONMENTAL_CUE_STEP, False)
    assert ORIENTATION_STEP not in [description for description, _ in steps]


def test_build_step_sequence_orders_orientation_before_environmental_cue_in_morning_routine():
    steps = build_step_sequence(ModuleId.PERSONAL_CARE, "brushing teeth", "07:30-08:30 Morning routine")
    assert steps[0] == (ORIENTATION_STEP, False)
    assert steps[1] == (ENVIRONMENTAL_CUE_STEP, False)


def test_build_step_sequence_no_environmental_cue_for_tasks_without_a_room():
    steps = build_step_sequence(ModuleId.HOUSEHOLD_CHORES, "folding clothes", "10:30-11:30 Household chores")
    assert ENVIRONMENTAL_CUE_STEP not in [description for description, _ in steps]


def test_adjust_assistance_level_escalates_on_frustration():
    assert (
        adjust_assistance_level(
            AssistanceLevel.VERBAL_CUE,
            floor=AssistanceLevel.INDEPENDENT,
            completed=True,
            frustration_detected=True,
            incident_flag=IncidentFlag.NONE,
        )
        == AssistanceLevel.DEMONSTRATION
    )


def test_adjust_assistance_level_escalates_on_incomplete_step():
    assert (
        adjust_assistance_level(
            AssistanceLevel.VERBAL_CUE,
            floor=AssistanceLevel.INDEPENDENT,
            completed=False,
            frustration_detected=False,
            incident_flag=IncidentFlag.NONE,
        )
        == AssistanceLevel.DEMONSTRATION
    )


def test_adjust_assistance_level_de_escalates_on_success():
    assert (
        adjust_assistance_level(
            AssistanceLevel.DEMONSTRATION,
            floor=AssistanceLevel.INDEPENDENT,
            completed=True,
            frustration_detected=False,
            incident_flag=IncidentFlag.NONE,
        )
        == AssistanceLevel.VERBAL_CUE
    )


def test_adjust_assistance_level_respects_floor():
    assert (
        adjust_assistance_level(
            AssistanceLevel.VERBAL_CUE,
            floor=AssistanceLevel.VERBAL_CUE,
            completed=True,
            frustration_detected=False,
            incident_flag=IncidentFlag.NONE,
        )
        == AssistanceLevel.VERBAL_CUE
    )


def test_adjust_assistance_level_unchanged_when_not_assessable():
    assert (
        adjust_assistance_level(
            AssistanceLevel.VERBAL_CUE,
            floor=AssistanceLevel.INDEPENDENT,
            completed=None,
            frustration_detected=False,
            incident_flag=IncidentFlag.NONE,
        )
        == AssistanceLevel.VERBAL_CUE
    )


def test_adjust_assistance_level_unchanged_on_hazard_incident():
    assert (
        adjust_assistance_level(
            AssistanceLevel.INDEPENDENT,
            floor=AssistanceLevel.INDEPENDENT,
            completed=True,
            frustration_detected=False,
            incident_flag=IncidentFlag.BURN,
        )
        == AssistanceLevel.INDEPENDENT
    )


def test_detect_hazard_incident_flags_burn():
    assert detect_hazard_incident("I burned my hand on the kettle") == IncidentFlag.BURN


def test_detect_hazard_incident_flags_fall():
    assert detect_hazard_incident("She fell near the counter") == IncidentFlag.FALL


def test_detect_hazard_incident_flags_distress():
    assert detect_hazard_incident("I feel dizzy and my chest hurts") == IncidentFlag.DISTRESS


def test_detect_hazard_incident_none_when_clean():
    assert detect_hazard_incident("Finished folding the clothes") == IncidentFlag.NONE


def test_detect_hazard_incident_ignores_bare_pain_mention():
    assert detect_hazard_incident("No pain at all, that was easy") == IncidentFlag.NONE


def test_detect_hazard_incident_none_when_no_response():
    assert detect_hazard_incident(None) == IncidentFlag.NONE


def test_is_safety_stop():
    assert is_safety_stop(IncidentFlag.FALL) is True
    assert is_safety_stop(IncidentFlag.NONE) is False


def test_assess_frustration_flags_keyword():
    frustrated, score = assess_frustration("this is too hard, I hate this", 5.0)
    assert frustrated is True
    assert score == 0.3


def test_assess_frustration_flags_slow_response():
    frustrated, score = assess_frustration("done", 60.0)
    assert frustrated is True


def test_assess_frustration_no_response_is_frustration():
    frustrated, score = assess_frustration(None, 0.0)
    assert frustrated is True
    assert score == 0.0


def test_assess_frustration_quick_reply_is_engaged():
    frustrated, score = assess_frustration("done", 5.0)
    assert frustrated is False
    assert score > 0.9


def test_assess_frustration_ignores_unrelated_no_more_and_don_t_want():
    frustrated, _ = assess_frustration("There's no more toothpaste, I don't want to run out", 5.0)
    assert frustrated is False


def test_enforce_hazard_floor_raises_low_tier():
    assert enforce_hazard_floor(AssistanceLevel.INDEPENDENT, is_hazard_step=True) == AssistanceLevel.DEMONSTRATION


def test_enforce_hazard_floor_leaves_high_tier():
    assert (
        enforce_hazard_floor(AssistanceLevel.PHYSICAL_ASSIST, is_hazard_step=True) == AssistanceLevel.PHYSICAL_ASSIST
    )


def test_enforce_hazard_floor_no_op_when_not_hazard():
    assert enforce_hazard_floor(AssistanceLevel.INDEPENDENT, is_hazard_step=False) == AssistanceLevel.INDEPENDENT
