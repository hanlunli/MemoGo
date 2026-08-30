from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from .models import (
    CheckpointStatus,
    CheckpointType,
    DeviationEvent,
    DeviationSeverity,
    DeviationType,
    DiseaseStage,
    MobilityLevel,
    ModuleId,
    PatientProfile,
    ProgressTrend,
    RoutineCheckpoint,
    RoutineSignalLog,
    SessionLog,
    SessionTurn,
)
from .modules import (
    CaregiverReporter,
    CheckpointAssessmentLayer,
    CheckpointDeliveryEngine,
    DeviationAssessmentLayer,
    DeviationResponseEngine,
    WakeUpOrientationOpener,
    assess_deviation_acknowledgment,
    assess_medication_supervision,
    build_checkpoint_sequence,
    classify_deviation_severity,
    detect_double_meal_request,
    resolve_checkpoint_status,
    resolve_checkpoints,
)
from .prompts import CheckpointAssessment, DeviationAssessment

logger = logging.getLogger("daily_life_routine_management.graph")

_CHECKPOINT_SERDE = JsonPlusSerializer(
    allowed_msgpack_modules=[
        DiseaseStage,
        ModuleId,
        MobilityLevel,
        CheckpointType,
        DeviationType,
        DeviationSeverity,
        CheckpointStatus,
        PatientProfile,
        RoutineCheckpoint,
        DeviationEvent,
        RoutineSignalLog,
        SessionTurn,
        SessionLog,
        ProgressTrend,
    ]
)


class SessionState(TypedDict, total=False):
    patient: PatientProfile
    trigger_type: str
    schedule_activity: Optional[str]
    deviation_event: Optional[DeviationEvent]
    module: ModuleId
    severity: DeviationSeverity
    checkpoints: list[RoutineCheckpoint]
    checkpoint_index: int
    current_prompt: str
    current_checkpoint: Optional[RoutineCheckpoint]
    is_deviation_turn: bool
    caregiver_response: Optional[str]
    response_latency_s: float
    turn_count: int
    max_turns: int
    consecutive_no_response_count: int
    session_log: SessionLog
    caregiver_summary: Optional[str]
    routine_alert: Optional[str]


def build_session_graph(llm: BaseChatModel):
    wake_up_orientation_opener = WakeUpOrientationOpener(llm)
    checkpoint_delivery_engine = CheckpointDeliveryEngine(llm)
    deviation_response_engine = DeviationResponseEngine(llm)
    checkpoint_assessment_layer = CheckpointAssessmentLayer(llm)
    deviation_assessment_layer = DeviationAssessmentLayer(llm)
    caregiver_reporter = CaregiverReporter(llm)

    def trigger_check(state: SessionState) -> SessionState:
        patient = state["patient"]
        if state.get("trigger_type") == "deviation_report":
            event = state["deviation_event"]
            severity = classify_deviation_severity(event.deviation_type)
            session_log = SessionLog(
                patient_id=patient.patient_id,
                module=ModuleId.DEVIATION_RESPONSE,
                trigger_type="deviation_report",
                started_at=datetime.now().isoformat(timespec="seconds"),
                event_handled=f"{event.deviation_type.value} at {event.checkpoint_type.value} checkpoint",
            )
            return {
                "module": ModuleId.DEVIATION_RESPONSE,
                "severity": severity,
                "checkpoints": [],
                "checkpoint_index": 0,
                "session_log": session_log,
            }

        checkpoint_types = resolve_checkpoints(state.get("schedule_activity", ""))
        checkpoints = build_checkpoint_sequence(patient, checkpoint_types)
        session_log = SessionLog(
            patient_id=patient.patient_id,
            module=ModuleId.ROUTINE_CHECKPOINT,
            trigger_type="scheduled_checkpoint",
            started_at=datetime.now().isoformat(timespec="seconds"),
        )
        return {
            "module": ModuleId.ROUTINE_CHECKPOINT,
            "checkpoints": checkpoints,
            "checkpoint_index": 0,
            "session_log": session_log,
        }

    def prepare_content(state: SessionState) -> SessionState:
        if state["module"] == ModuleId.DEVIATION_RESPONSE:
            prompt = deviation_response_engine.generate(state["deviation_event"], state["severity"])
            return {"current_prompt": prompt, "current_checkpoint": None, "is_deviation_turn": True}

        checkpoint = state["checkpoints"][state.get("checkpoint_index", 0)]
        if checkpoint.checkpoint_type == CheckpointType.WAKE_UP:
            prompt = wake_up_orientation_opener.generate(state["patient"])
        else:
            prompt = checkpoint_delivery_engine.generate(state["patient"], checkpoint)
        return {"current_prompt": prompt, "current_checkpoint": checkpoint, "is_deviation_turn": False}

    def deliver_and_capture(state: SessionState) -> SessionState:
        response = interrupt({"prompt": state["current_prompt"]})
        return {"caregiver_response": response["response"], "response_latency_s": response["latency"]}

    def adaptive_adjustment(state: SessionState) -> SessionState:
        response_text = state.get("caregiver_response")
        latency = state.get("response_latency_s", 0.0)
        session_log = state["session_log"]

        if state.get("is_deviation_turn"):
            severity = state["severity"]
            _, needs_followup = assess_deviation_acknowledgment(response_text, latency)

            try:
                assessment = deviation_assessment_layer.generate(state["current_prompt"], response_text or "")
            except Exception:
                logger.exception("Deviation assessment generation failed; using a fallback response")
                assessment = DeviationAssessment(
                    resolved=None, feedback="Thank you — this has been logged and flagged for follow-up."
                )

            session_log.turns.append(
                SessionTurn(
                    module=ModuleId.DEVIATION_RESPONSE,
                    prompt=state["current_prompt"],
                    caregiver_response=response_text,
                    feedback=assessment.feedback,
                    checkpoint_resolved=assessment.resolved,
                    signal=RoutineSignalLog(
                        status=resolve_checkpoint_status(assessment.resolved),
                        resolved=assessment.resolved,
                        needs_followup=needs_followup,
                    ),
                )
            )
            session_log.major_deviation = severity == DeviationSeverity.MAJOR
            session_log.needs_followup = needs_followup

            return {
                "session_log": session_log,
                "checkpoint_index": state.get("checkpoint_index", 0) + 1,
                "turn_count": state.get("turn_count", 0) + 1,
            }

        checkpoint: RoutineCheckpoint = state["current_checkpoint"]
        consecutive_no_response_count = (
            0 if response_text else state.get("consecutive_no_response_count", 0) + 1
        )

        if checkpoint.checkpoint_type == CheckpointType.WAKE_UP:
            feedback = "Thanks for checking in with me!"
            resolved: Optional[bool] = True
        else:
            try:
                assessment = checkpoint_assessment_layer.generate(state["current_prompt"], response_text or "")
            except Exception:
                logger.exception("Checkpoint assessment generation failed; using a fallback response")
                assessment = CheckpointAssessment(resolved=None, feedback="Thank you for confirming.")
            feedback = assessment.feedback
            resolved = assessment.resolved

            if checkpoint.checkpoint_type == CheckpointType.MEDICATION:
                if not assess_medication_supervision(response_text):
                    resolved = False
                    feedback = "A dose can only be logged once it's taken under your direct supervision."
                    session_log.missed_medication_supervision = True
            elif checkpoint.checkpoint_type == CheckpointType.MEAL:
                if detect_double_meal_request(response_text):
                    resolved = False
                    feedback = "This meal window is already logged as served — please check before offering more food."
                    session_log.double_meal_prevented = True

        status = resolve_checkpoint_status(resolved)
        session_log.turns.append(
            SessionTurn(
                module=ModuleId.ROUTINE_CHECKPOINT,
                prompt=state["current_prompt"],
                caregiver_response=response_text,
                feedback=feedback,
                checkpoint_resolved=resolved,
                signal=RoutineSignalLog(status=status, resolved=resolved),
            )
        )
        session_log.checkpoints_completed += 1
        status_key = status.value
        session_log.checkpoint_status_counts[status_key] = session_log.checkpoint_status_counts.get(status_key, 0) + 1

        return {
            "session_log": session_log,
            "checkpoint_index": state.get("checkpoint_index", 0) + 1,
            "turn_count": state.get("turn_count", 0) + 1,
            "consecutive_no_response_count": consecutive_no_response_count,
        }

    def session_close(state: SessionState) -> SessionState:
        session_log = state["session_log"]
        module = state["module"]

        if module == ModuleId.DEVIATION_RESPONSE:
            session_log.ended_reason = "major_deviation" if session_log.major_deviation else "deviation_closed"
        elif state.get("consecutive_no_response_count", 0) >= 2:
            session_log.ended_reason = "no_response_escalation"
        elif state.get("checkpoint_index", 0) >= len(state.get("checkpoints", [])):
            session_log.ended_reason = "checkpoints_completed"
        else:
            session_log.ended_reason = "max_turns_reached"

        alert_worthy = (
            session_log.missed_medication_supervision
            or session_log.double_meal_prevented
            or (module == ModuleId.DEVIATION_RESPONSE and session_log.major_deviation)
            or session_log.ended_reason == "no_response_escalation"
        )

        summary: Optional[str] = None
        routine_alert: Optional[str] = None
        try:
            if alert_worthy:
                routine_alert = caregiver_reporter.routine_alert(session_log.model_dump_json())
            else:
                summary = caregiver_reporter.summarize(session_log.model_dump_json())
        except Exception:
            logger.exception("Caregiver report generation failed; returning session without it")

        return {"session_log": session_log, "caregiver_summary": summary, "routine_alert": routine_alert}

    def route_after_adjustment(state: SessionState) -> str:
        if state["module"] == ModuleId.DEVIATION_RESPONSE:
            return "session_close"
        if (
            state.get("consecutive_no_response_count", 0) >= 2
            or state.get("checkpoint_index", 0) >= len(state.get("checkpoints", []))
            or state.get("turn_count", 0) >= state.get("max_turns", 8)
        ):
            return "session_close"
        return "prepare_content"

    graph = StateGraph(SessionState)
    graph.add_node("trigger_check", trigger_check)
    graph.add_node("prepare_content", prepare_content)
    graph.add_node("deliver_and_capture", deliver_and_capture)
    graph.add_node("adaptive_adjustment", adaptive_adjustment)
    graph.add_node("session_close", session_close)

    graph.set_entry_point("trigger_check")
    graph.add_edge("trigger_check", "prepare_content")
    graph.add_edge("prepare_content", "deliver_and_capture")
    graph.add_edge("deliver_and_capture", "adaptive_adjustment")
    graph.add_conditional_edges(
        "adaptive_adjustment",
        route_after_adjustment,
        {"prepare_content": "prepare_content", "session_close": "session_close"},
    )
    graph.add_edge("session_close", END)

    return graph.compile(checkpointer=InMemorySaver(serde=_CHECKPOINT_SERDE))
