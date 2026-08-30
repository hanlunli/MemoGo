from __future__ import annotations

import logging
import random
from datetime import datetime
from typing import Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from .models import (
    DiseaseStage,
    ExerciseItem,
    MemoryItem,
    ModuleId,
    PatientProfile,
    ProgressTrend,
    SessionLog,
    SessionTurn,
)
from .modules import (
    STAGE_DIFFICULTY_CEILING,
    CaregiverReporter,
    CognitiveExerciseGenerator,
    FeedbackEncouragementLayer,
    RealityOrientationEngine,
    ReminiscenceSessionManager,
    adjust_difficulty,
    assess_engagement,
    resolve_module,
)
from .prompts import ResponseAssessment

logger = logging.getLogger("cognitive_brain_training.graph")

DEFAULT_MEMORY = MemoryItem(media_type="story", theme="family", decade="1980s")
EXERCISE_TYPES = (
    "arithmetic",
    "word association",
    "picture recognition",
    "mahjong-style tile matching",
    "simple logic puzzle",
)

_CHECKPOINT_SERDE = JsonPlusSerializer(
    allowed_msgpack_modules=[
        DiseaseStage,
        ModuleId,
        PatientProfile,
        MemoryItem,
        ExerciseItem,
        SessionTurn,
        SessionLog,
        ProgressTrend,
    ]
)


class SessionState(TypedDict, total=False):
    patient: PatientProfile
    schedule_activity: str
    memory_items: list[MemoryItem]
    module: ModuleId
    current_prompt: str
    current_exercise: Optional[ExerciseItem]
    patient_response: Optional[str]
    response_latency_s: float
    difficulty: int
    turn_count: int
    max_turns: int
    fatigue_detected: bool
    engagement_score: float
    session_log: SessionLog
    caregiver_summary: str


def build_session_graph(llm: BaseChatModel):
    orientation_engine = RealityOrientationEngine(llm)
    reminiscence_manager = ReminiscenceSessionManager(llm)
    exercise_generator = CognitiveExerciseGenerator(llm)
    feedback_layer = FeedbackEncouragementLayer(llm)
    caregiver_reporter = CaregiverReporter(llm)

    def schedule_check(state: SessionState) -> SessionState:
        module = resolve_module(state["schedule_activity"])
        session_log = SessionLog(
            patient_id=state["patient"].patient_id,
            module=module,
            started_at=datetime.now().isoformat(timespec="seconds"),
        )
        return {"module": module, "session_log": session_log}

    def prepare_content(state: SessionState) -> SessionState:
        patient = state["patient"]
        module = state["module"]
        exercise: Optional[ExerciseItem] = None

        if module == ModuleId.REALITY_ORIENTATION:
            prompt = orientation_engine.generate_prompt(patient)
        elif module == ModuleId.REMINISCENCE:
            memories = state.get("memory_items") or [DEFAULT_MEMORY]
            memory = random.choice(memories)
            prompt = reminiscence_manager.generate_prompt(patient, memory)
        else:
            exercise = exercise_generator.generate(
                domain="memory",
                difficulty=state.get("difficulty", 2),
                exercise_type=random.choice(EXERCISE_TYPES),
            )
            prompt = exercise.content

        return {"current_prompt": prompt, "current_exercise": exercise}

    def deliver_and_capture(state: SessionState) -> SessionState:
        response = interrupt({"prompt": state["current_prompt"]})
        return {"patient_response": response["response"], "response_latency_s": response["latency"]}

    def adaptive_adjustment(state: SessionState) -> SessionState:
        response_text = state.get("patient_response")
        latency = state.get("response_latency_s", 0.0)
        fatigue_detected, engagement_score = assess_engagement(response_text, latency)

        exercise = state.get("current_exercise")
        expected_answer = exercise.answer if exercise else None
        try:
            assessment = feedback_layer.generate(
                prompt=state["current_prompt"],
                expected_answer=expected_answer,
                response=response_text or "",
            )
        except Exception:
            logger.exception("Feedback generation failed; using a fallback response")
            assessment = ResponseAssessment(correct=None, feedback="Thank you for your answer!")

        ceiling = STAGE_DIFFICULTY_CEILING[state["patient"].stage]
        new_difficulty = adjust_difficulty(
            state.get("difficulty", 2), ceiling, assessment.correct, fatigue_detected
        )

        session_log = state["session_log"]
        session_log.turns.append(
            SessionTurn(
                module=state["module"],
                prompt=state["current_prompt"],
                patient_response=response_text,
                feedback=assessment.feedback,
                correct=assessment.correct,
            )
        )
        session_log.engagement_score = engagement_score
        session_log.fatigue_detected = fatigue_detected

        return {
            "difficulty": new_difficulty,
            "fatigue_detected": fatigue_detected,
            "engagement_score": engagement_score,
            "turn_count": state.get("turn_count", 0) + 1,
            "session_log": session_log,
        }

    def session_close(state: SessionState) -> SessionState:
        session_log = state["session_log"]
        session_log.ended_reason = (
            "fatigue_detected" if state.get("fatigue_detected") else "max_turns_reached"
        )
        try:
            summary = caregiver_reporter.summarize(session_log.model_dump_json())
        except Exception:
            logger.exception("Caregiver summary generation failed; returning session without it")
            summary = None
        return {"session_log": session_log, "caregiver_summary": summary}

    def route_after_adjustment(state: SessionState) -> str:
        if state.get("fatigue_detected") or state.get("turn_count", 0) >= state.get("max_turns", 5):
            return "session_close"
        return "prepare_content"

    graph = StateGraph(SessionState)
    graph.add_node("schedule_check", schedule_check)
    graph.add_node("prepare_content", prepare_content)
    graph.add_node("deliver_and_capture", deliver_and_capture)
    graph.add_node("adaptive_adjustment", adaptive_adjustment)
    graph.add_node("session_close", session_close)

    graph.set_entry_point("schedule_check")
    graph.add_edge("schedule_check", "prepare_content")
    graph.add_edge("prepare_content", "deliver_and_capture")
    graph.add_edge("deliver_and_capture", "adaptive_adjustment")
    graph.add_conditional_edges(
        "adaptive_adjustment",
        route_after_adjustment,
        {"prepare_content": "prepare_content", "session_close": "session_close"},
    )
    graph.add_edge("session_close", END)

    return graph.compile(checkpointer=InMemorySaver(serde=_CHECKPOINT_SERDE))
