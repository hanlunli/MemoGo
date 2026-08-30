import os
import time

from dotenv import load_dotenv

from agents.social_creative_engagement_training import (
    DiseaseStage,
    PatientProfile,
    SocialContact,
    SocialCreativeEngagementTrainingAgent,
)
from agents.social_creative_engagement_training.llm_logging import configure_logging

load_dotenv()
configure_logging()

PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")


def main() -> None:
    if PROVIDER == "gemini" and not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("Set GOOGLE_API_KEY in a .env file (see .env.example) before running the demo.")

    patient = PatientProfile(
        patient_id="p-001",
        name="Grandma Chen",
        stage=DiseaseStage.MILD,
        preferred_crafts=["bead stringing", "origami"],
        preferred_songs=["folk songs"],
        social_contacts=[SocialContact(name="Mrs. Lee", relationship="neighbor")],
    )

    agent = SocialCreativeEngagementTrainingAgent(provider=PROVIDER)
    thread_id, step = agent.start_session(
        patient=patient,
        schedule_activity="14:30-15:30 Fine motor skills & art therapy",
        max_turns=3,
    )

    while not step.done:
        print(f"\nAgent: {step.prompt}")
        start = time.monotonic()
        response = input("You: ")
        latency = time.monotonic() - start
        step = agent.submit_response(thread_id, response, latency)

    print("\n--- Session Log ---")
    print(step.session_log.model_dump_json(indent=2))
    if step.alert:
        print("\n--- URGENT ALERT ---")
        print(step.alert)
    elif step.caregiver_summary:
        print("\n--- Caregiver Summary ---")
        print(step.caregiver_summary)


if __name__ == "__main__":
    main()
