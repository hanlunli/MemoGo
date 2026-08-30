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

from agents.cognitive_brain_training.models import MemoryItem

from .models import (
    ActivitySignalLog,
    CraftActivityItem,
    DiseaseStage,
    FineMotorLevel,
    IncidentFlag,
    ModuleId,
    MusicSessionItem,
    PatientProfile,
    ProgressTrend,
    SessionLog,
    SessionTurn,
    SocialContact,
)
from .modules import (
    DEFAULT_SONG_MEMORY,
    STAGE_COMPLEXITY_CEILING,
    CaregiverReporter,
    CraftsHorticultureActivityEngine,
    FeedbackEncouragementLayer,
    MusicTherapySessionEngine,
    SocialInteractionFacilitator,
    adjust_complexity,
    assess_mood_and_engagement,
    build_craft_step_sequence,
    choose_participation_mode,
    detect_material_hazard,
    is_safety_stop,
    resolve_module,
    select_conversation_topic,
    select_contact,
    select_craft_activity,
)
from .prompts import ParticipationAssessment

logger = logging.getLogger("social_creative_engagement_training.graph")

_CHECKPOINT_SERDE = JsonPlusSerializer(
    allowed_msgpack_modules=[
        DiseaseStage,
        ModuleId,
        FineMotorLevel,
        IncidentFlag,
        SocialContact,
        PatientProfile,
        MemoryItem,
        MusicSessionItem,
        CraftActivityItem,
        ActivitySignalLog,
        SessionTurn,
        SessionLog,
        ProgressTrend,
    ]
)


class SessionState(TypedDict, total=False):
    patient: PatientProfile
    schedule_activity: str
    memory_items: Optional[list[MemoryItem]]
    module: ModuleId
    activity_type: str
    steps: list[tuple[str, bool]]
    step_index: int
    contact: Optional[SocialContact]
    topic: str
    current_prompt: str
    current_is_hazard_step: bool
    patient_response: Optional[str]
    response_latency_s: float
    complexity: int
    turn_count: int
    max_turns: int
    consecutive_agitation_count: int
    agitation_detected: bool
    incident_flag: IncidentFlag
    engagement_score: float
    session_log: SessionLog
    caregiver_summary: Optional[str]
    urgent_alert: Optional[str]


def build_session_graph(llm: BaseChatModel):
    music_engine = MusicTherapySessionEngine(llm)
    crafts_engine = CraftsHorticultureActivityEngine(llm)
    social_facilitator = SocialInteractionFacilitator(llm)
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
        complexity = state.get("complexity", 2)

        if module == ModuleId.MUSIC_THERAPY:
            memories = state.get("memory_items") or [DEFAULT_SONG_MEMORY]
            memory = random.choice(memories)
            participation_mode = choose_participation_mode(complexity, state.get("agitation_detected", False))
            prompt = music_engine.generate_instruction(patient, memory, participation_mode)
            session_item = MusicSessionItem(
                theme=memory.theme, decade=memory.decade, participation_mode=participation_mode
            )
            return {
                "current_prompt": prompt,
                "current_is_hazard_step": False,
                "current_session_item": session_item,
            }

        if module == ModuleId.CRAFTS_HORTICULTURE:
            activity_type = state.get("activity_type") or select_craft_activity(patient)
            steps = state.get("steps") or build_craft_step_sequence(activity_type, patient.stage)
            step_index = state.get("step_index", 0)
            step_description, is_hazard_step = steps[min(step_index, len(steps) - 1)]
            prompt = crafts_engine.generate(patient, activity_type, step_description, is_hazard_step)
            return {
                "current_prompt": prompt,
                "current_is_hazard_step": is_hazard_step,
                "activity_type": activity_type,
                "steps": steps,
            }

        contact = state.get("contact") if "contact" in state else select_contact(patient)
        topic = state.get("topic") or select_conversation_topic(patient)
        prompt = social_facilitator.generate(patient, contact, topic)
        return {"current_prompt": prompt, "current_is_hazard_step": False, "contact": contact, "topic": topic}

    def deliver_and_capture(state: SessionState) -> SessionState:
        response = interrupt({"prompt": state["current_prompt"]})
        return {"patient_response": response["response"], "response_latency_s": response["latency"]}

    def adaptive_adjustment(state: SessionState) -> SessionState:
        response_text = state.get("patient_response")
        latency = state.get("response_latency_s", 0.0)
        module = state["module"]
        agitation_detected, engagement_score = assess_mood_and_engagement(response_text, latency)
        incident_flag = detect_material_hazard(response_text)

        try:
            assessment = feedback_layer.generate(
                prompt=state["current_prompt"], module=module, response=response_text or ""
            )
        except Exception:
            logger.exception("Feedback generation failed; using a fallback response")
            assessment = ParticipationAssessment(participated=None, feedback="Thank you for joining in!")

        ceiling = STAGE_COMPLEXITY_CEILING[state["patient"].stage]
        new_complexity = adjust_complexity(
            state.get("complexity", 2), ceiling, assessment.participated, agitation_detected, incident_flag
        )

        session_log = state["session_log"]
        session_log.turns.append(
            SessionTurn(
                module=module,
                prompt=state["current_prompt"],
                patient_response=response_text,
                feedback=assessment.feedback,
                participation_confirmed=assessment.participated,
                signal=ActivitySignalLog(incident_flag=incident_flag, agitation_detected=agitation_detected),
            )
        )
        session_log.engagement_score = engagement_score
        session_log.agitation_detected = agitation_detected
        session_log.hazard_incident = is_safety_stop(incident_flag)

        step_index = state.get("step_index", 0)
        if module == ModuleId.CRAFTS_HORTICULTURE and assessment.participated is True and not is_safety_stop(
            incident_flag
        ):
            step_index += 1

        consecutive_agitation_count = (
            state.get("consecutive_agitation_count", 0) + 1 if agitation_detected else 0
        )

        return {
            "complexity": new_complexity,
            "agitation_detected": agitation_detected,
            "incident_flag": incident_flag,
            "engagement_score": engagement_score,
            "turn_count": state.get("turn_count", 0) + 1,
            "step_index": step_index,
            "consecutive_agitation_count": consecutive_agitation_count,
            "session_log": session_log,
        }

    def session_close(state: SessionState) -> SessionState:
        session_log = state["session_log"]
        incident_flag = state.get("incident_flag", IncidentFlag.NONE)
        module = state["module"]
        steps = state.get("steps", [])
        sustained_agitation = state.get("consecutive_agitation_count", 0) >= 2

        if is_safety_stop(incident_flag):
            session_log.ended_reason = "hazard_incident"
        elif sustained_agitation:
            session_log.ended_reason = "sustained_agitation"
        elif module == ModuleId.CRAFTS_HORTICULTURE and state.get("step_index", 0) >= len(steps):
            session_log.ended_reason = "activity_completed"
        else:
            session_log.ended_reason = "max_turns_reached"

        summary: Optional[str] = None
        urgent_alert: Optional[str] = None
        try:
            if session_log.hazard_incident or sustained_agitation:
                urgent_alert = caregiver_reporter.urgent_alert(session_log.model_dump_json())
            else:
                summary = caregiver_reporter.summarize(session_log.model_dump_json())
        except Exception:
            logger.exception("Caregiver report generation failed; returning session without it")

        return {"session_log": session_log, "caregiver_summary": summary, "urgent_alert": urgent_alert}

    def route_after_adjustment(state: SessionState) -> str:
        incident_flag = state.get("incident_flag", IncidentFlag.NONE)
        module = state["module"]
        steps = state.get("steps", [])
        if (
            is_safety_stop(incident_flag)
            or state.get("consecutive_agitation_count", 0) >= 2
            or (module == ModuleId.CRAFTS_HORTICULTURE and state.get("step_index", 0) >= len(steps))
            or state.get("turn_count", 0) >= state.get("max_turns", 5)
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
