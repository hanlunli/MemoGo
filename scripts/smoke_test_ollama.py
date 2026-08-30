"""Non-interactive end-to-end smoke test against a local Ollama server. Not part of the pytest suite.

Requires Ollama running locally with the model pulled first: `ollama pull llama3.3`
"""

import itertools

from dotenv import load_dotenv

from agents.cognitive_brain_training import CognitiveBrainTrainingAgent, DiseaseStage, PatientProfile
from agents.cognitive_brain_training.llm_logging import configure_logging

load_dotenv()
configure_logging()

CANNED_RESPONSES = itertools.cycle(
    [
        ("42", 3.0),
        ("I remember gardening with my mother in the spring", 4.5),
        ("apple, banana, orange", 2.0),
    ]
)


def main() -> None:
    patient = PatientProfile(
        patient_id="smoke-test-ollama-001",
        name="Grandma Chen",
        stage=DiseaseStage.MILD,
        biography="Retired schoolteacher, raised three children, loves gardening and opera.",
        preferences=["gardening", "classic songs"],
    )

    agent = CognitiveBrainTrainingAgent(provider="ollama")
    thread_id, step = agent.start_session(
        patient=patient,
        schedule_activity="09:30-10:30 Targeted cognitive training",
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
