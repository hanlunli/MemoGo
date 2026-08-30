from agents.exercise_motor_coordination_training.models import IncidentFlag, ModuleId
from agents.exercise_motor_coordination_training.modules import (
    adjust_intensity,
    assess_exertion,
    detect_safety_incident,
    is_safety_stop,
    resolve_module,
)


def test_resolve_module_matches_keywords():
    assert resolve_module("15:30-16:30 Dual-Task Training: walk and count") == ModuleId.DUAL_TASK
    assert resolve_module("08:30-09:30 Outdoor aerobic exercise") == ModuleId.AEROBIC


def test_resolve_module_defaults_to_aerobic():
    assert resolve_module("16:30-18:00 Relaxation & dinner") == ModuleId.AEROBIC


def test_adjust_intensity_steps_down_on_fatigue():
    assert adjust_intensity(3, ceiling=5, sustained=True, fatigue_detected=True, incident_flag=IncidentFlag.NONE) == 2


def test_adjust_intensity_steps_up_on_sustained():
    assert adjust_intensity(2, ceiling=5, sustained=True, fatigue_detected=False, incident_flag=IncidentFlag.NONE) == 3


def test_adjust_intensity_steps_down_on_not_sustained():
    assert adjust_intensity(2, ceiling=5, sustained=False, fatigue_detected=False, incident_flag=IncidentFlag.NONE) == 1


def test_adjust_intensity_respects_ceiling():
    assert adjust_intensity(5, ceiling=5, sustained=True, fatigue_detected=False, incident_flag=IncidentFlag.NONE) == 5


def test_adjust_intensity_unchanged_when_not_assessable():
    assert adjust_intensity(3, ceiling=5, sustained=None, fatigue_detected=False, incident_flag=IncidentFlag.NONE) == 3


def test_adjust_intensity_forces_minimum_on_safety_incident():
    assert adjust_intensity(5, ceiling=5, sustained=True, fatigue_detected=False, incident_flag=IncidentFlag.FALL) == 1


def test_detect_safety_incident_flags_fall():
    assert detect_safety_incident("I fell down near the chair") == IncidentFlag.FALL


def test_detect_safety_incident_flags_near_fall():
    assert detect_safety_incident("I stumbled but caught myself") == IncidentFlag.NEAR_FALL


def test_detect_safety_incident_flags_distress():
    assert detect_safety_incident("I feel dizzy and my chest hurts") == IncidentFlag.DISTRESS


def test_detect_safety_incident_none_when_clean():
    assert detect_safety_incident("Finished the walk, feeling good") == IncidentFlag.NONE


def test_detect_safety_incident_none_when_no_response():
    assert detect_safety_incident(None) == IncidentFlag.NONE


def test_is_safety_stop():
    assert is_safety_stop(IncidentFlag.FALL) is True
    assert is_safety_stop(IncidentFlag.NONE) is False


def test_assess_exertion_flags_fatigue_keyword():
    fatigue, score = assess_exertion("I'm tired, can we stop", 5.0)
    assert fatigue is True
    assert score == 0.3


def test_assess_exertion_flags_slow_response():
    fatigue, score = assess_exertion("done", 60.0)
    assert fatigue is True


def test_assess_exertion_no_response_is_fatigue():
    fatigue, score = assess_exertion(None, 0.0)
    assert fatigue is True
    assert score == 0.0


def test_assess_exertion_quick_reply_is_engaged():
    fatigue, score = assess_exertion("done", 5.0)
    assert fatigue is False
    assert score > 0.9
