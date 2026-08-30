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
    ADLTaskStep,
    AssistanceLevel,
    DiseaseStage,
    IncidentFlag,
    MobilityLevel,
    ModuleId,
    PatientProfile,
    ProgressTrend,
    SessionLog,
    SessionTurn,
    TaskSignalLog,
)
from .modules import (
    ENVIRONMENTAL_CUE_STEP,
    ORIENTATION_STEP,
    STAGE_BASELINE_ASSISTANCE,
    STAGE_FLOOR_ASSISTANCE,
    TASK_ROOM_LABELS,
    CaregiverReporter,
    EnvironmentalCueEngine,
    FeedbackEncouragementLayer,
    MorningOrientationOpener,
    StepInstructionEngine,
    adjust_assistance_level,
    assess_frustration,
    build_step_sequence,
    detect_hazard_incident,
    enforce_hazard_floor,
    is_safety_stop,
    resolve_module,
    select_task,
)
from .prompts import StepAssessment

logger = logging.getLogger("adl_training.graph")

_CHECKPOINT_SERDE = JsonPlusSerializer(
    allowed_msgpack_modules=[
        DiseaseStage,
        ModuleId,
        MobilityLevel,
        AssistanceLevel,
        IncidentFlag,
        PatientProfile,
        ADLTaskStep,
        TaskSignalLog,
        SessionTurn,
        SessionLog,
        ProgressTrend,
    ]
)


class SessionState(TypedDict, total=False):
    patient: PatientProfile
    schedule_activity: str
    module: ModuleId
    task_name: str
    steps: list[tuple[str, bool]]
    step_index: int
    current_prompt: str
    current_is_hazard_step: bool
    is_orientation_turn: bool
    is_environmental_cue_turn: bool
    patient_response: Optional[str]
    response_latency_s: float
    assistance_level: AssistanceLevel
    turn_count: int
    max_turns: int
    consecutive_frustration_count: int
    frustration_detected: bool
    incident_flag: IncidentFlag
    engagement_score: float
    session_log: SessionLog
    caregiver_summary: Optional[str]
    hazard_alert: Optional[str]


def build_session_graph(llm: BaseChatModel):
    step_instruction_engine = StepInstructionEngine(llm)
    orientation_opener = MorningOrientationOpener(llm)
    environmental_cue_engine = EnvironmentalCueEngine(llm)
    feedback_layer = FeedbackEncouragementLayer(llm)
    caregiver_reporter = CaregiverReporter(llm)

    def schedule_check(state: SessionState) -> SessionState:
        module = resolve_module(state["schedule_activity"])
        task_name = select_task(module, state["patient"])
        steps = build_step_sequence(module, task_name, state["schedule_activity"])
        session_log = SessionLog(
            patient_id=state["patient"].patient_id,
            module=module,
            task_name=task_name,
            started_at=datetime.now().isoformat(timespec="seconds"),
        )
        return {
            "module": module,
            "task_name": task_name,
            "steps": steps,
            "step_index": 0,
            "session_log": session_log,
        }

    def prepare_content(state: SessionState) -> SessionState:
        patient = state["patient"]
        steps = state["steps"]
        step_description, is_hazard_step = steps[state.get("step_index", 0)]

        if step_description == ORIENTATION_STEP:
            prompt = orientation_opener.generate(patient)
            return {
                "current_prompt": prompt,
                "current_is_hazard_step": False,
                "is_orientation_turn": True,
                "is_environmental_cue_turn": False,
            }

        if step_description == ENVIRONMENTAL_CUE_STEP:
            room_label = TASK_ROOM_LABELS[state["task_name"]]
            prompt = environmental_cue_engine.generate(patient, room_label)
            return {
                "current_prompt": prompt,
                "current_is_hazard_step": False,
                "is_orientation_turn": False,
                "is_environmental_cue_turn": True,
            }

        assistance_level = state.get("assistance_level") or STAGE_BASELINE_ASSISTANCE[patient.stage]
        assistance_level = enforce_hazard_floor(assistance_level, is_hazard_step)
        prompt = step_instruction_engine.generate(
            patient, state["task_name"], step_description, assistance_level, is_hazard_step
        )
        return {
            "current_prompt": prompt,
            "current_is_hazard_step": is_hazard_step,
            "is_orientation_turn": False,
            "is_environmental_cue_turn": False,
            "assistance_level": assistance_level,
        }

    def deliver_and_capture(state: SessionState) -> SessionState:
        response = interrupt({"prompt": state["current_prompt"]})
        return {"patient_response": response["response"], "response_latency_s": response["latency"]}

    def adaptive_adjustment(state: SessionState) -> SessionState:
        response_text = state.get("patient_response")
        latency = state.get("response_latency_s", 0.0)
        frustration_detected, engagement_score = assess_frustration(response_text, latency)
        incident_flag = detect_hazard_incident(response_text)

        patient = state["patient"]
        is_orientation_turn = state.get("is_orientation_turn", False)
        is_environmental_cue_turn = state.get("is_environmental_cue_turn", False)
        assistance_level = state.get("assistance_level", AssistanceLevel.INDEPENDENT)

        if is_orientation_turn:
            feedback = "Thanks for checking in with me!"
            step_completed: Optional[bool] = True
            # Orientation has no assistance tier of its own; leave state["assistance_level"]
            # unset so the first real step still starts from the patient's stage baseline.
            new_assistance_level = state.get("assistance_level")
        elif is_environmental_cue_turn:
            feedback = "Great, you found it!"
            step_completed = True
            new_assistance_level = state.get("assistance_level")
        else:
            try:
                assessment = feedback_layer.generate(
                    prompt=state["current_prompt"],
                    assistance_level=assistance_level,
                    response=response_text or "",
                )
            except Exception:
                logger.exception("Feedback generation failed; using a fallback response")
                assessment = StepAssessment(completed=None, feedback="Great effort, thank you!")

            feedback = assessment.feedback
            step_completed = assessment.completed
            floor = STAGE_FLOOR_ASSISTANCE[patient.stage]
            new_assistance_level = adjust_assistance_level(
                assistance_level, floor, step_completed, frustration_detected, incident_flag
            )

        session_log = state["session_log"]
        session_log.turns.append(
            SessionTurn(
                module=state["module"],
                prompt=state["current_prompt"],
                patient_response=response_text,
                feedback=feedback,
                step_completed=step_completed,
                signal=TaskSignalLog(
                    assistance_level_used=assistance_level,
                    incident_flag=incident_flag,
                    frustration_detected=frustration_detected,
                ),
            )
        )
        session_log.engagement_score = engagement_score
        session_log.frustration_detected = frustration_detected
        session_log.hazard_incident = is_safety_stop(incident_flag)

        step_index = state.get("step_index", 0)
        should_advance = is_orientation_turn or is_environmental_cue_turn or step_completed is True
        if should_advance and not is_safety_stop(incident_flag):
            step_index += 1
            if not is_orientation_turn and not is_environmental_cue_turn:
                session_log.steps_completed += 1
                level_key = assistance_level.value
                session_log.assistance_level_counts[level_key] = (
                    session_log.assistance_level_counts.get(level_key, 0) + 1
                )

        consecutive_frustration_count = (
            state.get("consecutive_frustration_count", 0) + 1 if frustration_detected else 0
        )

        update: SessionState = {
            "frustration_detected": frustration_detected,
            "incident_flag": incident_flag,
            "engagement_score": engagement_score,
            "turn_count": state.get("turn_count", 0) + 1,
            "step_index": step_index,
            "consecutive_frustration_count": consecutive_frustration_count,
            "session_log": session_log,
        }
        if new_assistance_level is not None:
            update["assistance_level"] = new_assistance_level
        return update

    def session_close(state: SessionState) -> SessionState:
        session_log = state["session_log"]
        incident_flag = state.get("incident_flag", IncidentFlag.NONE)
        steps = state.get("steps", [])

        if is_safety_stop(incident_flag):
            session_log.ended_reason = "hazard_incident"
        elif state.get("consecutive_frustration_count", 0) >= 2:
            session_log.ended_reason = "sustained_frustration"
        elif state.get("step_index", 0) >= len(steps):
            session_log.ended_reason = "task_completed"
        else:
            session_log.ended_reason = "max_turns_reached"

        summary: Optional[str] = None
        hazard_alert: Optional[str] = None
        try:
            if session_log.hazard_incident:
                hazard_alert = caregiver_reporter.hazard_alert(session_log.model_dump_json())
            else:
                summary = caregiver_reporter.summarize(session_log.model_dump_json())
        except Exception:
            logger.exception("Caregiver report generation failed; returning session without it")

        return {"session_log": session_log, "caregiver_summary": summary, "hazard_alert": hazard_alert}

    def route_after_adjustment(state: SessionState) -> str:
        incident_flag = state.get("incident_flag", IncidentFlag.NONE)
        steps = state.get("steps", [])
        if (
            is_safety_stop(incident_flag)
            or state.get("consecutive_frustration_count", 0) >= 2
            or state.get("step_index", 0) >= len(steps)
            or state.get("turn_count", 0) >= state.get("max_turns", 8)
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
