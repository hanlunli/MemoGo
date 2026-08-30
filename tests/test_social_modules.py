from agents.social_creative_engagement_training.models import (
    DiseaseStage,
    FineMotorLevel,
    IncidentFlag,
    ModuleId,
    PatientProfile,
)
from agents.social_creative_engagement_training.modules import (
    adjust_complexity,
    assess_mood_and_engagement,
    build_craft_step_sequence,
    choose_participation_mode,
    detect_material_hazard,
    is_safety_stop,
    resolve_module,
    select_craft_activity,
)


def _make_patient(**overrides) -> PatientProfile:
    defaults = dict(patient_id="t1", name="Test", stage=DiseaseStage.MILD)
    defaults.update(overrides)
    return PatientProfile(**defaults)


def test_resolve_module_matches_keywords():
    assert resolve_module("13:00-14:30 Reminiscence & social engagement") == ModuleId.SOCIAL_INTERACTION
    assert resolve_module("14:30-15:30 Fine motor skills & art therapy") == ModuleId.CRAFTS_HORTICULTURE
    assert resolve_module("Afternoon music therapy sing-along") == ModuleId.MUSIC_THERAPY


def test_resolve_module_defaults_to_crafts():
    assert resolve_module("16:30-18:00 Relaxation & dinner") == ModuleId.CRAFTS_HORTICULTURE


def test_build_craft_step_sequence_applies_choking_risk_floor_for_beads_at_moderate_stage():
    steps = build_craft_step_sequence("bead stringing", DiseaseStage.MODERATE)
    assert all(is_hazard for _, is_hazard in steps)


def test_build_craft_step_sequence_leaves_beads_unflagged_for_mild_stage():
    steps = build_craft_step_sequence("bead stringing", DiseaseStage.MILD)
    assert all(not is_hazard for _, is_hazard in steps)


def test_build_craft_step_sequence_always_flags_paper_cutting():
    steps = build_craft_step_sequence("paper cutting", DiseaseStage.MILD)
    assert all(is_hazard for _, is_hazard in steps)


def test_select_craft_activity_avoids_high_dexterity_crafts_when_needs_assist():
    patient = _make_patient(fine_motor_level=FineMotorLevel.NEEDS_ASSIST)
    for _ in range(50):
        assert select_craft_activity(patient) in ("watering plants", "drawing")


def test_select_craft_activity_avoids_sensitive_material_even_when_preferred():
    patient = _make_patient(preferred_crafts=["origami"], material_sensitivities=["paper"])
    for _ in range(50):
        assert select_craft_activity(patient) != "origami"


def test_choose_participation_mode_listens_only_on_agitation():
    assert choose_participation_mode(complexity=5, agitation_detected=True) == "listen_only"


def test_choose_participation_mode_sings_at_high_complexity():
    assert choose_participation_mode(complexity=4, agitation_detected=False) == "sing_along"


def test_choose_participation_mode_claps_at_mid_complexity():
    assert choose_participation_mode(complexity=2, agitation_detected=False) == "clap_along"


def test_choose_participation_mode_listens_at_low_complexity():
    assert choose_participation_mode(complexity=1, agitation_detected=False) == "listen_only"


def test_adjust_complexity_steps_down_on_agitation():
    assert adjust_complexity(3, ceiling=5, participated=True, agitation_detected=True, incident_flag=IncidentFlag.NONE) == 2


def test_adjust_complexity_steps_up_on_participation():
    assert adjust_complexity(2, ceiling=5, participated=True, agitation_detected=False, incident_flag=IncidentFlag.NONE) == 3


def test_adjust_complexity_steps_down_on_non_participation():
    assert adjust_complexity(2, ceiling=5, participated=False, agitation_detected=False, incident_flag=IncidentFlag.NONE) == 1


def test_adjust_complexity_respects_ceiling():
    assert adjust_complexity(5, ceiling=5, participated=True, agitation_detected=False, incident_flag=IncidentFlag.NONE) == 5


def test_adjust_complexity_unchanged_when_not_assessable():
    assert adjust_complexity(3, ceiling=5, participated=None, agitation_detected=False, incident_flag=IncidentFlag.NONE) == 3


def test_adjust_complexity_forces_minimum_on_hazard_incident():
    assert adjust_complexity(5, ceiling=5, participated=True, agitation_detected=False, incident_flag=IncidentFlag.CUT) == 1


def test_detect_material_hazard_flags_cut():
    assert detect_material_hazard("I cut my finger on the scissors") == IncidentFlag.CUT


def test_detect_material_hazard_flags_burn():
    assert detect_material_hazard("I got burned by the glue gun") == IncidentFlag.BURN


def test_detect_material_hazard_flags_choking_risk():
    assert detect_material_hazard("She swallowed a small bead") == IncidentFlag.CHOKING_RISK


def test_detect_material_hazard_none_when_clean():
    assert detect_material_hazard("Finished stringing the beads, feeling proud") == IncidentFlag.NONE


def test_detect_material_hazard_none_when_no_response():
    assert detect_material_hazard(None) == IncidentFlag.NONE


def test_is_safety_stop():
    assert is_safety_stop(IncidentFlag.CUT) is True
    assert is_safety_stop(IncidentFlag.NONE) is False


def test_assess_mood_and_engagement_flags_agitation_keyword():
    agitated, score = assess_mood_and_engagement("I'm upset, no more of this", 5.0)
    assert agitated is True
    assert score == 0.3


def test_assess_mood_and_engagement_flags_slow_response():
    agitated, score = assess_mood_and_engagement("done", 60.0)
    assert agitated is True


def test_assess_mood_and_engagement_no_response_is_agitation():
    agitated, score = assess_mood_and_engagement(None, 0.0)
    assert agitated is True
    assert score == 0.0


def test_assess_mood_and_engagement_quick_reply_is_engaged():
    agitated, score = assess_mood_and_engagement("done", 5.0)
    assert agitated is False
    assert score > 0.9
