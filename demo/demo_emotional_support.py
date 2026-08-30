import os
import time

from dotenv import load_dotenv

from agents.emotional_support_comfort import DiseaseStage, EmotionalSupportComfortAgent, PatientProfile
from agents.emotional_support_comfort.llm_logging import configure_logging

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
        known_triggers=["unfamiliar visitors"],
        calming_preferences=["soft music from the 1960s", "family photo album"],
        independence_tasks=["folding clothes", "watering plants"],
        emergency_contacts=["Daughter Amy"],
    )

    agent = EmotionalSupportComfortAgent(provider=PROVIDER)
    thread_id, step = agent.start_session(
        patient=patient,
        schedule_activity="16:00-16:30 Sundowning prevention window",
        max_turns=6,
    )

    while not step.done:
        print(f"\nAgent: {step.prompt}")
        start = time.monotonic()
        response = input("You (caregiver): ")
        latency = time.monotonic() - start
        step = agent.submit_response(thread_id, response, latency)

    print("\n--- Session Log ---")
    print(step.session_log.model_dump_json(indent=2))
    if step.safety_alert:
        print("\n--- SAFETY ALERT ---")
        print(step.safety_alert)
    elif step.caregiver_summary:
        print("\n--- Caregiver Summary ---")
        print(step.caregiver_summary)


if __name__ == "__main__":
    main()
