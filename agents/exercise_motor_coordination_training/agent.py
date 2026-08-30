from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langgraph.types import Command

from .graph import build_session_graph
from .llm_logging import LLMUsageLoggingHandler
from .models import PatientProfile, SessionLog

Provider = Literal["ollama", "gemini"]

DEFAULT_MODELS: dict[Provider, str] = {
    "ollama": "llama3.3",
    "gemini": "gemini-2.5-flash",
}


def _build_llm(provider: Provider, model: str, temperature: float, llm_kwargs: dict) -> BaseChatModel:
    callbacks = [LLMUsageLoggingHandler()]
    if provider == "ollama":
        return ChatOllama(model=model, temperature=temperature, callbacks=callbacks, **llm_kwargs)
    if provider == "gemini":
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, callbacks=callbacks, **llm_kwargs)
    raise ValueError(f"Unknown provider: {provider!r}")


@dataclass
class SessionStep:
    done: bool
    prompt: Optional[str] = None
    session_log: Optional[SessionLog] = None
    caregiver_summary: Optional[str] = None
    safety_alert: Optional[str] = None


class ExerciseMotorCoordinationTrainingAgent:
    def __init__(
        self,
        provider: Provider = "ollama",
        model: Optional[str] = None,
        temperature: float = 0.4,
        **llm_kwargs,
    ):
        self._llm = _build_llm(provider, model or DEFAULT_MODELS[provider], temperature, llm_kwargs)
        self._graph = build_session_graph(self._llm)

    def start_session(
        self,
        patient: PatientProfile,
        schedule_activity: str,
        max_turns: int = 4,
        target_duration_min: int = 30,
        starting_intensity: int = 2,
        thread_id: Optional[str] = None,
    ) -> tuple[str, SessionStep]:
        thread_id = thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {
            "patient": patient,
            "schedule_activity": schedule_activity,
            "max_turns": max_turns,
            "target_duration_min": target_duration_min,
            "turn_count": 0,
            "intensity": starting_intensity,
            "fatigue_detected": False,
        }
        result = self._graph.invoke(initial_state, config=config)
        return thread_id, self._to_step(result)

    def submit_response(self, thread_id: str, response_text: str, response_latency_s: float) -> SessionStep:
        config = {"configurable": {"thread_id": thread_id}}
        result = self._graph.invoke(
            Command(resume={"response": response_text, "latency": response_latency_s}),
            config=config,
        )
        return self._to_step(result)

    @staticmethod
    def _to_step(result: dict) -> SessionStep:
        interrupts = result.get("__interrupt__")
        if interrupts:
            return SessionStep(done=False, prompt=interrupts[0].value["prompt"])
        return SessionStep(
            done=True,
            session_log=result["session_log"],
            caregiver_summary=result.get("caregiver_summary"),
            safety_alert=result.get("safety_alert"),
        )
