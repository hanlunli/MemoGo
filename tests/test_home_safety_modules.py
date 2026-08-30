from agents.home_safety_protection.models import ChecklistStatus, HazardType, ModuleId, Severity, WanderingRiskLevel
from agents.home_safety_protection.modules import (
    assess_incident_acknowledgment,
    build_checklist_sequence,
    classify_incident_severity,
    escalate_risk_level,
    existing_adl_room_labels,
    recommend_device_for_risk,
    resolve_checklist_status,
    resolve_module,
)


def test_resolve_module_matches_keywords():
    assert resolve_module("09:00-09:30 Weekly home safety walkthrough") == ModuleId.HOME_SAFETY_AUDIT
    assert resolve_module("Monthly home safety audit") == ModuleId.HOME_SAFETY_AUDIT


def test_resolve_module_defaults_to_home_safety_audit():
    assert resolve_module("16:30-18:00 Relaxation & dinner") == ModuleId.HOME_SAFETY_AUDIT


def test_build_checklist_sequence_covers_every_requested_room():
    items = build_checklist_sequence(["Kitchen", "Bathroom"])
    rooms = {item.room for item in items}
    assert rooms == {"Kitchen", "Bathroom"}


def test_build_checklist_sequence_surfaces_high_priority_items_first():
    items = build_checklist_sequence(["Kitchen", "Bathroom", "Bedroom", "Hallway"])
    priorities = [item.is_high_priority for item in items]
    first_low_priority_index = priorities.index(False)
    assert all(priorities[:first_low_priority_index])


def test_build_checklist_sequence_unknown_room_yields_no_items():
    assert build_checklist_sequence(["Garage"]) == []


def test_classify_incident_severity_is_always_emergency():
    for hazard_type in HazardType:
        assert classify_incident_severity(hazard_type) == Severity.EMERGENCY


def test_escalate_risk_level_raises_on_wandering():
    assert escalate_risk_level(WanderingRiskLevel.LOW, HazardType.WANDERING) == WanderingRiskLevel.MODERATE
    assert escalate_risk_level(WanderingRiskLevel.MODERATE, HazardType.WANDERING) == WanderingRiskLevel.HIGH


def test_escalate_risk_level_caps_at_high():
    assert escalate_risk_level(WanderingRiskLevel.HIGH, HazardType.WANDERING) == WanderingRiskLevel.HIGH


def test_escalate_risk_level_raises_on_medication_chemical_access():
    assert (
        escalate_risk_level(WanderingRiskLevel.LOW, HazardType.MEDICATION_CHEMICAL_ACCESS)
        == WanderingRiskLevel.MODERATE
    )


def test_escalate_risk_level_unaffected_by_unrelated_hazard():
    assert escalate_risk_level(WanderingRiskLevel.LOW, HazardType.FIRE_GAS) == WanderingRiskLevel.LOW
    assert escalate_risk_level(WanderingRiskLevel.LOW, HazardType.FALL) == WanderingRiskLevel.LOW


def test_recommend_device_for_risk():
    assert recommend_device_for_risk(WanderingRiskLevel.LOW) is None
    assert "GPS" in recommend_device_for_risk(WanderingRiskLevel.MODERATE)
    assert "GPS" in recommend_device_for_risk(WanderingRiskLevel.HIGH)


def test_resolve_checklist_status():
    assert resolve_checklist_status(True) == ChecklistStatus.DONE
    assert resolve_checklist_status(False) == ChecklistStatus.NEEDS_ATTENTION
    assert resolve_checklist_status(None) == ChecklistStatus.PENDING


def test_assess_incident_acknowledgment_no_response_escalates():
    acknowledged, escalate = assess_incident_acknowledgment(None, 0.0)
    assert acknowledged is False
    assert escalate is True


def test_assess_incident_acknowledgment_refusal_escalates():
    acknowledged, escalate = assess_incident_acknowledgment("I can't get there right now", 5.0)
    assert acknowledged is False
    assert escalate is True


def test_assess_incident_acknowledgment_slow_response_escalates():
    acknowledged, escalate = assess_incident_acknowledgment("Checking now", 120.0)
    assert acknowledged is False
    assert escalate is True


def test_assess_incident_acknowledgment_prompt_reply_is_acknowledged():
    acknowledged, escalate = assess_incident_acknowledgment("Handled, I turned off the stove", 5.0)
    assert acknowledged is True
    assert escalate is False


def test_assess_incident_acknowledgment_vague_reply_still_escalates():
    acknowledged, escalate = assess_incident_acknowledgment("ok", 5.0)
    assert acknowledged is False
    assert escalate is True


def test_existing_adl_room_labels_returns_task_room_labels():
    labels = existing_adl_room_labels()
    assert "Bathroom" in labels
    assert "Kitchen" in labels
