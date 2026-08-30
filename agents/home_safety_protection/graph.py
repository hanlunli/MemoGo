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
    ChecklistItem,
    ChecklistStatus,
    DiseaseStage,
    HazardType,
    IncidentEvent,
    MobilityLevel,
    ModuleId,
    PatientProfile,
    ProgressTrend,
    SafetySignalLog,
    Severity,
    SessionLog,
    SessionTurn,
    WanderingRiskLevel,
)
from .modules import (
    CaregiverReporter,
    ChecklistItemEngine,
    IncidentAssessmentLayer,
    IncidentTriageEngine,
    ItemAssessmentLayer,
    assess_incident_acknowledgment,
    build_checklist_sequence,
    classify_incident_severity,
    escalate_risk_level,
    recommend_device_for_risk,
    resolve_checklist_status,
    resolve_module,
)
from .prompts import IncidentAssessment, ItemAssessment

logger = logging.getLogger("home_safety_protection.graph")

_CHECKPOINT_SERDE = JsonPlusSerializer(
    allowed_msgpack_modules=[
        DiseaseStage,
        ModuleId,
        MobilityLevel,
        WanderingRiskLevel,
        HazardType,
        Severity,
        ChecklistStatus,
        PatientProfile,
        ChecklistItem,
        IncidentEvent,
        SafetySignalLog,
        SessionTurn,
        SessionLog,
        ProgressTrend,
    ]
)


class SessionState(TypedDict, total=False):
    patient: PatientProfile
    trigger_type: str
    schedule_activity: Optional[str]
    incident_event: Optional[IncidentEvent]
    module: ModuleId
    severity: Severity
    steps: list[ChecklistItem]
    step_index: int
    current_prompt: str
    current_item: Optional[ChecklistItem]
    is_incident_turn: bool
    caregiver_response: Optional[str]
    response_latency_s: float
    turn_count: int
    max_turns: int
    wandering_risk_level: WanderingRiskLevel
    escalated_to_contact: bool
    session_log: SessionLog
    caregiver_summary: Optional[str]
    safety_alert: Optional[str]


def build_session_graph(llm: BaseChatModel):
    checklist_item_engine = ChecklistItemEngine(llm)
    incident_triage_engine = IncidentTriageEngine(llm)
    item_assessment_layer = ItemAssessmentLayer(llm)
    incident_assessment_layer = IncidentAssessmentLayer(llm)
    caregiver_reporter = CaregiverReporter(llm)

    def trigger_check(state: SessionState) -> SessionState:
        patient = state["patient"]
        if state.get("trigger_type") == "event_driven":
            event = state["incident_event"]
            severity = classify_incident_severity(event.hazard_type)
            session_log = SessionLog(
                patient_id=patient.patient_id,
                module=ModuleId.INCIDENT_RESPONSE,
                trigger_type="event_driven",
                started_at=datetime.now().isoformat(timespec="seconds"),
                event_handled=f"{event.hazard_type.value} at {event.location} ({event.source_device})",
            )
            return {
                "module": ModuleId.INCIDENT_RESPONSE,
                "severity": severity,
                "steps": [],
                "step_index": 0,
                "wandering_risk_level": patient.wandering_risk_level,
                "session_log": session_log,
            }

        module = resolve_module(state.get("schedule_activity", ""))
        steps = build_checklist_sequence(patient.home_rooms)
        session_log = SessionLog(
            patient_id=patient.patient_id,
            module=module,
            trigger_type="scheduled_audit",
            started_at=datetime.now().isoformat(timespec="seconds"),
        )
        return {
            "module": module,
            "steps": steps,
            "step_index": 0,
            "wandering_risk_level": patient.wandering_risk_level,
            "session_log": session_log,
        }

    def prepare_content(state: SessionState) -> SessionState:
        if state["module"] == ModuleId.INCIDENT_RESPONSE:
            prompt = incident_triage_engine.generate(state["incident_event"], state["severity"])
            return {"current_prompt": prompt, "current_item": None, "is_incident_turn": True}

        item = state["steps"][state.get("step_index", 0)]
        prompt = checklist_item_engine.generate(state["patient"], item)
        return {"current_prompt": prompt, "current_item": item, "is_incident_turn": False}

    def deliver_and_capture(state: SessionState) -> SessionState:
        response = interrupt({"prompt": state["current_prompt"]})
        return {"caregiver_response": response["response"], "response_latency_s": response["latency"]}

    def adaptive_adjustment(state: SessionState) -> SessionState:
        response_text = state.get("caregiver_response")
        latency = state.get("response_latency_s", 0.0)
        session_log = state["session_log"]

        if state.get("is_incident_turn"):
            event: IncidentEvent = state["incident_event"]
            severity = state["severity"]
            acknowledged, escalate_to_contact = assess_incident_acknowledgment(response_text, latency)

            try:
                assessment = incident_assessment_layer.generate(state["current_prompt"], response_text or "")
            except Exception:
                logger.exception("Incident assessment generation failed; using a fallback response")
                assessment = IncidentAssessment(
                    resolved=None, feedback="Thank you — this has been logged and flagged for follow-up."
                )

            new_risk = escalate_risk_level(state["wandering_risk_level"], event.hazard_type)
            session_log.turns.append(
                SessionTurn(
                    module=ModuleId.INCIDENT_RESPONSE,
                    prompt=state["current_prompt"],
                    caregiver_response=response_text,
                    feedback=assessment.feedback,
                    item_resolved=assessment.resolved,
                    signal=SafetySignalLog(
                        severity=severity, resolved=assessment.resolved, escalated_to_contact=escalate_to_contact
                    ),
                )
            )
            session_log.emergency_incident = severity == Severity.EMERGENCY
            session_log.escalated_to_contact = escalate_to_contact
            session_log.wandering_risk_level_after = new_risk
            session_log.device_recommendation = recommend_device_for_risk(new_risk)

            return {
                "session_log": session_log,
                "wandering_risk_level": new_risk,
                "escalated_to_contact": escalate_to_contact,
                "step_index": state.get("step_index", 0) + 1,
                "turn_count": state.get("turn_count", 0) + 1,
            }

        item: ChecklistItem = state["current_item"]
        try:
            assessment = item_assessment_layer.generate(state["current_prompt"], response_text or "")
        except Exception:
            logger.exception("Checklist item assessment generation failed; using a fallback response")
            assessment = ItemAssessment(resolved=None, feedback="Thank you for checking on this.")

        status = resolve_checklist_status(assessment.resolved)
        session_log.turns.append(
            SessionTurn(
                module=ModuleId.HOME_SAFETY_AUDIT,
                prompt=state["current_prompt"],
                caregiver_response=response_text,
                feedback=assessment.feedback,
                item_resolved=assessment.resolved,
                signal=SafetySignalLog(severity=Severity.INFO, resolved=assessment.resolved),
            )
        )
        session_log.checklist_items_reviewed += 1
        status_key = status.value
        session_log.checklist_status_counts[status_key] = session_log.checklist_status_counts.get(status_key, 0) + 1

        return {
            "session_log": session_log,
            "step_index": state.get("step_index", 0) + 1,
            "turn_count": state.get("turn_count", 0) + 1,
        }

    def session_close(state: SessionState) -> SessionState:
        session_log = state["session_log"]
        module = state["module"]

        if module == ModuleId.INCIDENT_RESPONSE:
            session_log.ended_reason = "emergency_incident" if session_log.emergency_incident else "incident_closed"
        elif state.get("step_index", 0) >= len(state.get("steps", [])):
            session_log.ended_reason = "audit_completed"
        else:
            session_log.ended_reason = "max_turns_reached"

        summary: Optional[str] = None
        safety_alert: Optional[str] = None
        try:
            if module == ModuleId.INCIDENT_RESPONSE and session_log.emergency_incident:
                safety_alert = caregiver_reporter.emergency_alert(session_log.model_dump_json())
            else:
                summary = caregiver_reporter.summarize(session_log.model_dump_json())
        except Exception:
            logger.exception("Caregiver report generation failed; returning session without it")

        return {"session_log": session_log, "caregiver_summary": summary, "safety_alert": safety_alert}

    def route_after_adjustment(state: SessionState) -> str:
        if state["module"] == ModuleId.INCIDENT_RESPONSE:
            return "session_close"
        if state.get("step_index", 0) >= len(state.get("steps", [])) or state.get("turn_count", 0) >= state.get(
            "max_turns", 8
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
