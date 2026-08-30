import os
import time

from dotenv import load_dotenv

from agents.companionship import CompanionshipAgent, DiseaseStage, PatientProfile
from agents.companionship.llm_logging import configure_logging
from agents.companionship.models import DistortionType, RealityDistortionEvent

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
        reminiscence_background=["her years running the family noodle shop"],
        sensitive_topics_to_avoid=["losing her husband"],
        caregiver_name="Amy",
        days_since_last_respite=9,
    )

    agent = CompanionshipAgent(provider=PROVIDER)

    session_type = os.environ.get("COMPANIONSHIP_DEMO_MODE", "scheduled")
    if session_type == "distortion":
        distortion_event = RealityDistortionEvent(
            distortion_type=DistortionType.WANTS_TO_GO_HOME,
            patient_statement="I need to go home now, where are my shoes?",
        )
        thread_id, step = agent.start_distortion_session(patient=patient, distortion_event=distortion_event, max_turns=3)
    else:
        thread_id, step = agent.start_session(
            patient=patient,
            schedule_activity="13:00-14:30 Reminiscence & social engagement",
            max_turns=8,
        )

    while not step.done:
        print(f"\nAgent: {step.prompt}")
        start = time.monotonic()
        response = input("You (caregiver): ")
        latency = time.monotonic() - start
        step = agent.submit_response(thread_id, response, latency)

    print("\n--- Session Log ---")
    print(step.session_log.model_dump_json(indent=2))
    if step.caregiver_alert:
        print("\n--- CAREGIVER ALERT ---")
        print(step.caregiver_alert)
    elif step.caregiver_summary:
        print("\n--- Caregiver Summary ---")
        print(step.caregiver_summary)


if __name__ == "__main__":
    main()
