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
    ActivityStepType,
    DiseaseStage,
    DistortionType,
    ModuleId,
    PatientProfile,
    ProgressTrend,
    RealityDistortionEvent,
    ReminiscenceBoxItem,
    SelfCareNudgeType,
    SessionLog,
    SessionTurn,
    StageActivityStep,
)
from .modules import (
    CaregiverReporter,
    DistortionResponseCoach,
    DistortionValidationAssessmentLayer,
    ReminiscenceEngagementAssessmentLayer,
    ReminiscenceSessionEngine,
    SelfCareNudgeEngine,
    StageActivityAssessmentLayer,
    StageActivityEngine,
    REMINISCENCE_STEP_COUNT,
    assess_validation_signal,
    build_reminiscence_box,
    build_stage_activity_sequence,
    classify_distortion_type,
    detect_nudge_acceptance,
    detects_fatigue_signal,
    resolve_engagement_status,
    resolve_reminiscence_theme,
    select_self_care_nudge,
)
from .prompts import DistortionValidationAssessment, ReminiscenceEngagementAssessment, StageActivityAssessment

logger = logging.getLogger("companionship.graph")

_CHECKPOINT_SERDE = JsonPlusSerializer(
    allowed_msgpack_modules=[
        DiseaseStage,
        ModuleId,
        ActivityStepType,
        SelfCareNudgeType,
        DistortionType,
        PatientProfile,
        StageActivityStep,
        ReminiscenceBoxItem,
        RealityDistortionEvent,
        SessionTurn,
        SessionLog,
        ProgressTrend,
    ]
)


class SessionState(TypedDict, total=False):
    patient: PatientProfile
    trigger_type: str
    distortion_event: Optional[RealityDistortionEvent]
    module: ModuleId
    distortion_type: Optional[DistortionType]
    steps: list[StageActivityStep]
    step_index: int
    current_step: Optional[StageActivityStep]
    current_self_care_nudge_type: Optional[SelfCareNudgeType]
    reminiscence_active: bool
    reminiscence_step_index: int
    reminiscence_theme: Optional[str]
    reminiscence_box: Optional[ReminiscenceBoxItem]
    current_prompt: str
    caregiver_response: Optional[str]
    response_latency_s: float
    turn_count: int
    max_turns: int
    distortion_attempt_index: int
    last_validated_signal: Optional[bool]
    session_log: SessionLog
    caregiver_summary: Optional[str]
    caregiver_alert: Optional[str]


def build_session_graph(llm: BaseChatModel):
    stage_activity_engine = StageActivityEngine(llm)
    self_care_engine = SelfCareNudgeEngine(llm)
    reminiscence_engine = ReminiscenceSessionEngine(llm)
    distortion_coach = DistortionResponseCoach(llm)
    stage_activity_assessment_layer = StageActivityAssessmentLayer(llm)
    reminiscence_assessment_layer = ReminiscenceEngagementAssessmentLayer(llm)
    distortion_assessment_layer = DistortionValidationAssessmentLayer(llm)
    caregiver_reporter = CaregiverReporter(llm)

    def trigger_check(state: SessionState) -> SessionState:
        patient = state["patient"]
        if state.get("trigger_type") == "event_driven":
            event = state["distortion_event"]
            distortion_type = event.distortion_type or classify_distortion_type(event.patient_statement)
            session_log = SessionLog(
                patient_id=patient.patient_id,
                module=ModuleId.DISTORTION_RESPONSE,
                trigger_type="event_driven",
                started_at=datetime.now().isoformat(timespec="seconds"),
            )
            return {
                "module": ModuleId.DISTORTION_RESPONSE,
                "distortion_type": distortion_type,
                "distortion_attempt_index": 0,
                "last_validated_signal": None,
                "session_log": session_log,
            }

        steps = build_stage_activity_sequence(patient)
        session_log = SessionLog(
            patient_id=patient.patient_id,
            module=ModuleId.SCHEDULED_COMPANIONSHIP,
            trigger_type="scheduled_companionship",
            started_at=datetime.now().isoformat(timespec="seconds"),
        )
        return {
            "module": ModuleId.SCHEDULED_COMPANIONSHIP,
            "steps": steps,
            "step_index": 0,
            "reminiscence_active": False,
            "reminiscence_step_index": 0,
            "session_log": session_log,
        }

    def prepare_content(state: SessionState) -> SessionState:
        if state["module"] == ModuleId.DISTORTION_RESPONSE:
            prompt = distortion_coach.generate(
                distortion_type=state["distortion_type"],
                attempt_number=state.get("distortion_attempt_index", 0) + 1,
                patient_statement=state["distortion_event"].patient_statement,
            )
            return {"current_prompt": prompt}

        patient = state["patient"]

        if state.get("reminiscence_active"):
            prompt = reminiscence_engine.generate_step(
                patient,
                state.get("reminiscence_step_index", 0),
                state["reminiscence_theme"],
                state["reminiscence_box"],
            )
            return {"current_prompt": prompt}

        step = state["steps"][state.get("step_index", 0)]

        if step.step_type == ActivityStepType.REMINISCENCE_THERAPY:
            theme = resolve_reminiscence_theme(patient)
            box = build_reminiscence_box(theme)
            prompt = reminiscence_engine.generate_step(patient, 0, theme, box)
            return {
                "current_prompt": prompt,
                "current_step": step,
                "reminiscence_active": True,
                "reminiscence_step_index": 0,
                "reminiscence_theme": theme,
                "reminiscence_box": box,
            }

        if step.step_type == ActivityStepType.SELF_CARE_NUDGE:
            nudge_type = select_self_care_nudge(patient)
            prompt = self_care_engine.generate(patient, nudge_type)
            return {"current_prompt": prompt, "current_step": step, "current_self_care_nudge_type": nudge_type}

        prompt = stage_activity_engine.generate(patient, step)
        return {"current_prompt": prompt, "current_step": step}

    def deliver_and_capture(state: SessionState) -> SessionState:
        response = interrupt({"prompt": state["current_prompt"]})
        return {"caregiver_response": response["response"], "response_latency_s": response["latency"]}

    def adaptive_adjustment(state: SessionState) -> SessionState:
        response_text = state.get("caregiver_response")
        latency = state.get("response_latency_s", 0.0)
        session_log = state["session_log"]

        if state["module"] == ModuleId.DISTORTION_RESPONSE:
            settled = assess_validation_signal(response_text, latency)

            try:
                assessment: DistortionValidationAssessment = distortion_assessment_layer.generate(
                    state["current_prompt"], response_text or ""
                )
                feedback = assessment.feedback
            except Exception:
                logger.exception("Distortion-validation assessment generation failed; using a fallback response")
                feedback = "Understood — keep validating gently, I'm logging this."

            session_log.turns.append(
                SessionTurn(
                    module=ModuleId.DISTORTION_RESPONSE,
                    prompt=state["current_prompt"],
                    caregiver_response=response_text,
                    feedback=feedback,
                    item_addressed=settled,
                )
            )

            return {
                "session_log": session_log,
                "last_validated_signal": settled,
                "distortion_attempt_index": state.get("distortion_attempt_index", 0) + 1,
                "turn_count": state.get("turn_count", 0) + 1,
            }

        if state.get("reminiscence_active"):
            fatigue = detects_fatigue_signal(response_text)

            try:
                reminiscence_assessment: ReminiscenceEngagementAssessment = reminiscence_assessment_layer.generate(
                    state["current_prompt"], response_text or ""
                )
                engaged, feedback = reminiscence_assessment.engaged, reminiscence_assessment.feedback
            except Exception:
                logger.exception("Reminiscence engagement assessment generation failed; using a fallback response")
                engaged, feedback = None, "Thank you for sharing this moment together."

            session_log.turns.append(
                SessionTurn(
                    module=ModuleId.SCHEDULED_COMPANIONSHIP,
                    prompt=state["current_prompt"],
                    caregiver_response=response_text,
                    feedback=feedback,
                    item_addressed=engaged,
                )
            )

            next_reminiscence_index = state.get("reminiscence_step_index", 0) + 1
            if fatigue or next_reminiscence_index >= REMINISCENCE_STEP_COUNT:
                session_log.reminiscence_session_conducted = True
                session_log.reminiscence_theme_used = state.get("reminiscence_theme")
                return {
                    "session_log": session_log,
                    "reminiscence_active": False,
                    "reminiscence_step_index": 0,
                    "step_index": state.get("step_index", 0) + 1,
                    "turn_count": state.get("turn_count", 0) + 1,
                }
            return {
                "session_log": session_log,
                "reminiscence_step_index": next_reminiscence_index,
                "turn_count": state.get("turn_count", 0) + 1,
            }

        step: StageActivityStep = state["current_step"]
        if step.step_type == ActivityStepType.SELF_CARE_NUDGE:
            accepted = detect_nudge_acceptance(response_text)
            session_log.self_care_nudge_delivered = state["current_self_care_nudge_type"].value
            session_log.self_care_nudge_accepted = accepted
            session_log.turns.append(
                SessionTurn(
                    module=ModuleId.SCHEDULED_COMPANIONSHIP,
                    prompt=state["current_prompt"],
                    caregiver_response=response_text,
                    feedback="Thank you for taking care of yourself too.",
                    item_addressed=accepted,
                )
            )
        else:
            try:
                activity_assessment: StageActivityAssessment = stage_activity_assessment_layer.generate(
                    state["current_prompt"], response_text or ""
                )
                engaged, feedback = activity_assessment.engaged, activity_assessment.feedback
            except Exception:
                logger.exception("Stage-activity assessment generation failed; using a fallback response")
                engaged, feedback = None, "Thank you for spending this time together."

            session_log.turns.append(
                SessionTurn(
                    module=ModuleId.SCHEDULED_COMPANIONSHIP,
                    prompt=state["current_prompt"],
                    caregiver_response=response_text,
                    feedback=feedback,
                    item_addressed=engaged,
                )
            )
            if engaged:
                session_log.activities_completed += 1
            status_key = resolve_engagement_status(engaged)
            session_log.activity_status_counts[status_key] = session_log.activity_status_counts.get(status_key, 0) + 1

        return {
            "session_log": session_log,
            "step_index": state.get("step_index", 0) + 1,
            "turn_count": state.get("turn_count", 0) + 1,
        }

    def session_close(state: SessionState) -> SessionState:
        session_log = state["session_log"]
        module = state["module"]

        summary: Optional[str] = None
        alert: Optional[str] = None

        if module == ModuleId.DISTORTION_RESPONSE:
            settled = state.get("last_validated_signal")
            attempts = state.get("distortion_attempt_index", 0)
            session_log.distortion_attempts = attempts

            if settled is True:
                session_log.ended_reason = "validated_and_settled"
                session_log.distortion_handled = f"Settled after {attempts} attempt(s)."
            else:
                session_log.ended_reason = "unresolved_after_attempts"
                session_log.distortion_handled = f"Not settled after {attempts} attempt(s)."

            try:
                if settled is True:
                    summary = caregiver_reporter.summarize(session_log.model_dump_json())
                else:
                    alert = caregiver_reporter.unresolved_alert(session_log.model_dump_json())
            except Exception:
                logger.exception("Caregiver report generation failed; returning session without it")
        else:
            session_log.ended_reason = (
                "companionship_completed"
                if state.get("step_index", 0) >= len(state.get("steps", []))
                else "max_turns_reached"
            )
            try:
                summary = caregiver_reporter.summarize(session_log.model_dump_json())
            except Exception:
                logger.exception("Caregiver report generation failed; returning session without it")

        return {"session_log": session_log, "caregiver_summary": summary, "caregiver_alert": alert}

    def route_after_adjustment(state: SessionState) -> str:
        if state["module"] == ModuleId.DISTORTION_RESPONSE:
            if state.get("last_validated_signal") is True:
                return "session_close"
            if state.get("distortion_attempt_index", 0) >= state.get("max_turns", 3):
                return "session_close"
            return "prepare_content"

        if state.get("turn_count", 0) >= state.get("max_turns", 8):
            return "session_close"

        if state.get("reminiscence_active"):
            return "prepare_content"

        if state.get("step_index", 0) >= len(state.get("steps", [])):
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
