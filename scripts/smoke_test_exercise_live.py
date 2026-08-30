"""Non-interactive end-to-end smoke test against the real Gemini API. Not part of the pytest suite."""

import itertools

from dotenv import load_dotenv

from agents.exercise_motor_coordination_training import (
    DiseaseStage,
    ExerciseMotorCoordinationTrainingAgent,
    MobilityLevel,
    PatientProfile,
)
from agents.exercise_motor_coordination_training.llm_logging import configure_logging

load_dotenv()
configure_logging()

CANNED_RESPONSES = itertools.cycle(
    [
        ("Walked for a few minutes, feeling okay", 3.0),
        ("Counted to ten while marching, a bit slow", 4.5),
        ("Finished the cool-down stretch", 2.0),
    ]
)


def main() -> None:
    patient = PatientProfile(
        patient_id="smoke-test-exercise-001",
        name="Grandma Chen",
        stage=DiseaseStage.MILD,
        mobility_level=MobilityLevel.INDEPENDENT,
        preferred_exercise_types=["walking", "tai chi"],
        indoor_outdoor_preference="outdoor",
    )

    agent = ExerciseMotorCoordinationTrainingAgent(provider="gemini")
    thread_id, step = agent.start_session(
        patient=patient,
        schedule_activity="08:30-09:30 Outdoor aerobic exercise",
        max_turns=2,
    )

    while not step.done:
        response, latency = next(CANNED_RESPONSES)
        print(f"\nAgent: {step.prompt}")
        print(f"Patient (scripted): {response}")
        step = agent.submit_response(thread_id, response, latency)

    print("\n--- Session Log ---")
    print(step.session_log.model_dump_json(indent=2))

    assert len(step.session_log.turns) >= 1, "expected at least one turn to be logged"
    assert all(turn.feedback for turn in step.session_log.turns), "every turn should have feedback"
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
