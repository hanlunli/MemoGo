"""Non-interactive end-to-end smoke test against a local Ollama server. Not part of the pytest suite.

Requires Ollama running locally with the model pulled first: `ollama pull llama3.3`
"""

import itertools

from dotenv import load_dotenv

from agents.adl_training import ADLTrainingAgent, DiseaseStage, PatientProfile
from agents.adl_training.llm_logging import configure_logging

load_dotenv()
configure_logging()

CANNED_RESPONSES = itertools.cycle(
    [
        ("Okay, picked it up", 3.0),
        ("Done, folded it", 4.5),
        ("Put it on the stack", 2.0),
    ]
)


def main() -> None:
    patient = PatientProfile(
        patient_id="smoke-test-adl-ollama-001",
        name="Grandma Chen",
        stage=DiseaseStage.MILD,
        preferred_adl_tasks=["folding clothes"],
    )

    agent = ADLTrainingAgent(provider="ollama")
    thread_id, step = agent.start_session(
        patient=patient,
        schedule_activity="10:30-11:30 Household chores",
        max_turns=3,
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
