"""Non-interactive end-to-end smoke test against the real Gemini API. Not part of the pytest suite."""

import itertools

from dotenv import load_dotenv

from agents.social_creative_engagement_training import (
    PatientProfile,
    SocialContact,
    SocialCreativeEngagementTrainingAgent,
)
from agents.social_creative_engagement_training.llm_logging import configure_logging

load_dotenv()
configure_logging()

CANNED_RESPONSES = itertools.cycle(
    [
        ("Picked up a bead and threaded it, smiling", 3.0),
        ("Threaded another bead slowly", 4.5),
        ("Finished the necklace, feeling proud", 2.0),
    ]
)


def main() -> None:
    patient = PatientProfile(
        patient_id="smoke-test-social-001",
        name="Grandma Chen",
        preferred_crafts=["bead stringing"],
        preferred_songs=["folk songs"],
        social_contacts=[SocialContact(name="Mrs. Lee", relationship="neighbor")],
    )

    agent = SocialCreativeEngagementTrainingAgent(provider="gemini")
    thread_id, step = agent.start_session(
        patient=patient,
        schedule_activity="14:30-15:30 Fine motor skills & art therapy",
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
