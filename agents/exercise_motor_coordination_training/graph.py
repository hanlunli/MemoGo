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
    ExerciseSessionItem,
    IncidentFlag,
    MobilityLevel,
    ModuleId,
    MotorSignalLog,
    PatientProfile,
    ProgressTrend,
    SessionLog,
    SessionTurn,
)
from .modules import (
    AEROBIC_EXERCISE_TYPES,
    MOTOR_TASKS,
    STAGE_INTENSITY_CEILING,
    AerobicExerciseSessionEngine,
    CaregiverReporter,
    DualTaskTrainingCoordinator,
    FeedbackEncouragementLayer,
    adjust_intensity,
    assess_exertion,
    detect_safety_incident,
    is_safety_stop,
    resolve_module,
)
from .prompts import MotorResponseAssessment

logger = logging.getLogger("exercise_motor_coordination_training.graph")

AEROBIC_PHASES = ("warm-up", "sustain", "cool-down")

_CHECKPOINT_SERDE = JsonPlusSerializer(
    allowed_msgpack_modules=[
        DiseaseStage,
        ModuleId,
        MobilityLevel,
        IncidentFlag,
        PatientProfile,
        ExerciseSessionItem,
        MotorSignalLog,
        SessionTurn,
        SessionLog,
        ProgressTrend,
    ]
)


class SessionState(TypedDict, total=False):
    patient: PatientProfile
    schedule_activity: str
    module: ModuleId
    exercise_type: str
    motor_task: str
    current_prompt: str
    current_session_item: Optional[ExerciseSessionItem]
    has_cognitive_task: bool
    patient_response: Optional[str]
    response_latency_s: float
    intensity: int
    turn_count: int
    max_turns: int
    target_duration_min: int
    fatigue_detected: bool
    incident_flag: IncidentFlag
    engagement_score: float
    session_log: SessionLog
    caregiver_summary: Optional[str]
    safety_alert: Optional[str]


def _select_aerobic_type(patient: PatientProfile) -> str:
    preferred = [p for p in patient.preferred_exercise_types if p in AEROBIC_EXERCISE_TYPES]
    return random.choice(preferred) if preferred else random.choice(AEROBIC_EXERCISE_TYPES)


def build_session_graph(llm: BaseChatModel):
    aerobic_engine = AerobicExerciseSessionEngine(llm)
    dual_task_coordinator = DualTaskTrainingCoordinator(llm)
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
        turn_count = state.get("turn_count", 0)

        if module == ModuleId.AEROBIC:
            exercise_type = state.get("exercise_type") or _select_aerobic_type(patient)
            phase = AEROBIC_PHASES[min(turn_count, len(AEROBIC_PHASES) - 1)]
            prompt = aerobic_engine.generate_instruction(patient, exercise_type, phase)
            has_cognitive_task = False
            session_item = ExerciseSessionItem(
                exercise_type="aerobic",
                motor_task=exercise_type,
                cognitive_task=None,
                target_duration_min=state.get("target_duration_min", 30),
                intensity_level=state.get("intensity", 2),
            )
        else:
            motor_task = state.get("motor_task") or random.choice(MOTOR_TASKS)
            include_cognitive_task = turn_count > 0
            prompt, cognitive_task = dual_task_coordinator.generate_instruction(
                patient, motor_task, include_cognitive_task
            )
            has_cognitive_task = cognitive_task is not None
            session_item = ExerciseSessionItem(
                exercise_type="dual-task",
                motor_task=motor_task,
                cognitive_task=cognitive_task,
                target_duration_min=state.get("target_duration_min", 20),
                intensity_level=state.get("intensity", 2),
            )

        return {
            "current_prompt": prompt,
            "current_session_item": session_item,
            "has_cognitive_task": has_cognitive_task,
            "exercise_type": session_item.motor_task,
            "motor_task": session_item.motor_task,
        }

    def deliver_and_capture(state: SessionState) -> SessionState:
        response = interrupt({"prompt": state["current_prompt"]})
        return {"patient_response": response["response"], "response_latency_s": response["latency"]}

    def adaptive_adjustment(state: SessionState) -> SessionState:
        response_text = state.get("patient_response")
        latency = state.get("response_latency_s", 0.0)
        fatigue_detected, engagement_score = assess_exertion(response_text, latency)
        incident_flag = detect_safety_incident(response_text)

        try:
            assessment = feedback_layer.generate(
                prompt=state["current_prompt"],
                has_cognitive_task=state.get("has_cognitive_task", False),
                response=response_text or "",
            )
        except Exception:
            logger.exception("Feedback generation failed; using a fallback response")
            assessment = MotorResponseAssessment(sustained=None, feedback="Great effort, thank you!")

        ceiling = STAGE_INTENSITY_CEILING[state["patient"].stage]
        new_intensity = adjust_intensity(
            state.get("intensity", 2), ceiling, assessment.sustained, fatigue_detected, incident_flag
        )

        max_turns = max(state.get("max_turns", 4), 1)
        slice_min = state.get("target_duration_min", 30) / max_turns

        session_log = state["session_log"]
        session_log.turns.append(
            SessionTurn(
                module=state["module"],
                prompt=state["current_prompt"],
                patient_response=response_text,
                feedback=assessment.feedback,
                sustained=assessment.sustained,
                motor_signal=MotorSignalLog(incident_flag=incident_flag),
            )
        )
        session_log.engagement_score = engagement_score
        session_log.fatigue_detected = fatigue_detected
        session_log.safety_incident = is_safety_stop(incident_flag)
        session_log.duration_achieved_min = round(session_log.duration_achieved_min + slice_min, 1)

        return {
            "intensity": new_intensity,
            "fatigue_detected": fatigue_detected,
            "incident_flag": incident_flag,
            "engagement_score": engagement_score,
            "turn_count": state.get("turn_count", 0) + 1,
            "session_log": session_log,
        }

    def session_close(state: SessionState) -> SessionState:
        session_log = state["session_log"]
        incident_flag = state.get("incident_flag", IncidentFlag.NONE)
        if is_safety_stop(incident_flag):
            session_log.ended_reason = "safety_incident"
        elif state.get("fatigue_detected"):
            session_log.ended_reason = "fatigue_detected"
        else:
            session_log.ended_reason = "max_turns_reached"

        summary: Optional[str] = None
        safety_alert: Optional[str] = None
        try:
            if session_log.safety_incident:
                safety_alert = caregiver_reporter.safety_alert(session_log.model_dump_json())
            else:
                summary = caregiver_reporter.summarize(session_log.model_dump_json())
        except Exception:
            logger.exception("Caregiver report generation failed; returning session without it")

        return {"session_log": session_log, "caregiver_summary": summary, "safety_alert": safety_alert}

    def route_after_adjustment(state: SessionState) -> str:
        incident_flag = state.get("incident_flag", IncidentFlag.NONE)
        if (
            is_safety_stop(incident_flag)
            or state.get("fatigue_detected")
            or state.get("turn_count", 0) >= state.get("max_turns", 4)
        ):
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
