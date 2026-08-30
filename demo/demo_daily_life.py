import os
import time

from dotenv import load_dotenv

from agents.daily_life_routine_management import DailyLifeRoutineManagementAgent, DiseaseStage, PatientProfile
from agents.daily_life_routine_management.llm_logging import configure_logging

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
        dietary_restrictions=["easy to chew"],
        medication_names=["Donepezil"],
    )

    agent = DailyLifeRoutineManagementAgent(provider=PROVIDER)
    thread_id, step = agent.start_session(
        patient=patient,
        schedule_activity="07:30-08:30 Morning routine, hygiene & breakfast",
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
    if step.routine_alert:
        print("\n--- ROUTINE ALERT ---")
        print(step.routine_alert)
    elif step.caregiver_summary:
        print("\n--- Caregiver Summary ---")
        print(step.caregiver_summary)


if __name__ == "__main__":
    main()
