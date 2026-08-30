import os
import time

from dotenv import load_dotenv

from agents.cognitive_brain_training import CognitiveBrainTrainingAgent, DiseaseStage, PatientProfile
from agents.cognitive_brain_training.llm_logging import configure_logging

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
        biography="Retired schoolteacher, raised three children, loves gardening and opera.",
        preferences=["gardening", "classic songs"],
    )

    agent = CognitiveBrainTrainingAgent(provider=PROVIDER)
    thread_id, step = agent.start_session(
        patient=patient,
        schedule_activity="09:30-10:30 Targeted cognitive training",
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
    if step.caregiver_summary:
        print("\n--- Caregiver Summary ---")
        print(step.caregiver_summary)


if __name__ == "__main__":
    main()
