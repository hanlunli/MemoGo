from agents.cognitive_brain_training.models import ModuleId
from agents.cognitive_brain_training.modules import (
    adjust_difficulty,
    assess_engagement,
    resolve_module,
)


def test_resolve_module_matches_keywords():
    assert resolve_module("09:30-10:30 Targeted cognitive training") == ModuleId.EXERCISE
    assert resolve_module("13:00-14:30 Reminiscence & social engagement") == ModuleId.REMINISCENCE
    assert resolve_module("07:30-08:30 Reality Orientation check-in") == ModuleId.REALITY_ORIENTATION


def test_resolve_module_defaults_to_exercise():
    assert resolve_module("16:30-18:00 Relaxation & dinner") == ModuleId.EXERCISE


def test_adjust_difficulty_steps_down_on_fatigue():
    assert adjust_difficulty(3, ceiling=5, correct=True, fatigue_detected=True) == 2


def test_adjust_difficulty_steps_up_on_correct():
    assert adjust_difficulty(2, ceiling=5, correct=True, fatigue_detected=False) == 3


def test_adjust_difficulty_steps_down_on_incorrect():
    assert adjust_difficulty(2, ceiling=5, correct=False, fatigue_detected=False) == 1


def test_adjust_difficulty_respects_ceiling():
    assert adjust_difficulty(5, ceiling=5, correct=True, fatigue_detected=False) == 5


def test_adjust_difficulty_unchanged_when_open_ended():
    assert adjust_difficulty(3, ceiling=5, correct=None, fatigue_detected=False) == 3


def test_assess_engagement_flags_fatigue_keyword():
    fatigue, score = assess_engagement("I'm tired, can we stop", 5.0)
    assert fatigue is True
    assert score == 0.3


def test_assess_engagement_flags_slow_response():
    fatigue, score = assess_engagement("42", 60.0)
    assert fatigue is True


def test_assess_engagement_no_response_is_fatigue():
    fatigue, score = assess_engagement(None, 0.0)
    assert fatigue is True
    assert score == 0.0


def test_assess_engagement_quick_reply_is_engaged():
    fatigue, score = assess_engagement("42", 5.0)
    assert fatigue is False
    assert score > 0.9
