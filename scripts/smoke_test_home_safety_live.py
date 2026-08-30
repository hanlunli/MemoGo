"""Non-interactive end-to-end smoke test against the real Gemini API. Not part of the pytest suite."""

import itertools

from dotenv import load_dotenv

from agents.home_safety_protection import DiseaseStage, HazardType, HomeSafetyProtectionAgent, IncidentEvent, PatientProfile
from agents.home_safety_protection.llm_logging import configure_logging

load_dotenv()
configure_logging()

CANNED_RESPONSES = itertools.cycle(
    [
        ("Installed a non-slip mat", 3.0),
        ("Grab bars are already up", 4.5),
        ("Will add a nightlight tonight", 2.0),
    ]
)


def run_audit_smoke_test() -> None:
    patient = PatientProfile(
        patient_id="smoke-test-home-safety-001",
        name="Grandma Chen",
        stage=DiseaseStage.MILD,
        home_rooms=["Bathroom", "Bedroom"],
    )

    agent = HomeSafetyProtectionAgent(provider="gemini")
    thread_id, step = agent.start_session(
        patient=patient,
        schedule_activity="09:00-09:30 Weekly home safety walkthrough",
        max_turns=3,
    )

    while not step.done:
        response, latency = next(CANNED_RESPONSES)
        print(f"\nAgent: {step.prompt}")
        print(f"Caregiver (scripted): {response}")
        step = agent.submit_response(thread_id, response, latency)

    print("\n--- Audit Session Log ---")
    print(step.session_log.model_dump_json(indent=2))

    assert len(step.session_log.turns) >= 1, "expected at least one turn to be logged"
    assert all(turn.feedback for turn in step.session_log.turns), "every turn should have feedback"


def run_incident_smoke_test() -> None:
    patient = PatientProfile(patient_id="smoke-test-home-safety-002", name="Grandma Chen", stage=DiseaseStage.MILD)
    incident_event = IncidentEvent(hazard_type=HazardType.FIRE_GAS, location="Kitchen", source_device="Gas detector")

    agent = HomeSafetyProtectionAgent(provider="gemini")
    thread_id, step = agent.start_incident_session(patient=patient, incident_event=incident_event, max_turns=2)

    while not step.done:
        print(f"\nAgent: {step.prompt}")
        print("Caregiver (scripted): Turned off the gas and opened the windows")
        step = agent.submit_response(thread_id, "Turned off the gas and opened the windows", 4.0)

    print("\n--- Incident Session Log ---")
    print(step.session_log.model_dump_json(indent=2))

    assert step.session_log.emergency_incident is True, "fire/gas incidents must be flagged as emergencies"
    assert step.safety_alert, "an emergency incident must produce a safety alert"


def main() -> None:
    run_audit_smoke_test()
    run_incident_smoke_test()
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
