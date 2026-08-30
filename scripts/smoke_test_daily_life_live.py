"""Non-interactive end-to-end smoke test against the real Gemini API. Not part of the pytest suite."""

import itertools

from dotenv import load_dotenv

from agents.daily_life_routine_management import (
    DailyLifeRoutineManagementAgent,
    DeviationEvent,
    DeviationType,
    DiseaseStage,
    PatientProfile,
)
from agents.daily_life_routine_management.models import CheckpointType
from agents.daily_life_routine_management.llm_logging import configure_logging

load_dotenv()
configure_logging()

CANNED_RESPONSES = itertools.cycle(
    [
        ("The blue cardigan, please", 3.0),
        ("Water, thank you", 2.5),
        ("Gave the medication just now, watched her take it", 4.0),
    ]
)


def run_checkpoint_smoke_test() -> None:
    patient = PatientProfile(
        patient_id="smoke-test-daily-life-001",
        name="Grandma Chen",
        stage=DiseaseStage.MILD,
        medication_names=["Donepezil"],
    )

    agent = DailyLifeRoutineManagementAgent(provider="gemini")
    thread_id, step = agent.start_session(
        patient=patient,
        schedule_activity="07:30-08:30 Morning routine, hygiene & breakfast",
        max_turns=3,
    )

    while not step.done:
        response, latency = next(CANNED_RESPONSES)
        print(f"\nAgent: {step.prompt}")
        print(f"Caregiver (scripted): {response}")
        step = agent.submit_response(thread_id, response, latency)

    print("\n--- Checkpoint Session Log ---")
    print(step.session_log.model_dump_json(indent=2))

    assert len(step.session_log.turns) >= 1, "expected at least one turn to be logged"
    assert all(turn.feedback for turn in step.session_log.turns), "every turn should have feedback"


def run_deviation_smoke_test() -> None:
    patient = PatientProfile(patient_id="smoke-test-daily-life-002", name="Grandma Chen", stage=DiseaseStage.MILD)
    deviation_event = DeviationEvent(
        deviation_type=DeviationType.ENVIRONMENT_CHANGE,
        checkpoint_type=CheckpointType.BEDTIME,
        description="A new home-care aide is staying overnight for the first time.",
    )

    agent = DailyLifeRoutineManagementAgent(provider="gemini")
    thread_id, step = agent.start_deviation_session(patient=patient, deviation_event=deviation_event, max_turns=2)

    while not step.done:
        print(f"\nAgent: {step.prompt}")
        print("Caregiver (scripted): Adjusted the evening routine to introduce her gently")
        step = agent.submit_response(thread_id, "Adjusted the evening routine to introduce her gently", 4.0)

    print("\n--- Deviation Session Log ---")
    print(step.session_log.model_dump_json(indent=2))

    assert step.session_log.major_deviation is True, "environment changes must be flagged as major deviations"
    assert step.routine_alert, "a major deviation must produce a routine alert"


def main() -> None:
    run_checkpoint_smoke_test()
    run_deviation_smoke_test()
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
