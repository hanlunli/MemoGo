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
    DiseaseStage,
    HabitType,
    ModuleId,
    OutburstEvent,
    OutburstSeverity,
    OutburstSignalLog,
    PatientProfile,
    PreventionHabitItem,
    ProgressTrend,
    RedirectionType,
    SessionLog,
    SessionTurn,
    SundowningJournalEntry,
    SundowningRiskLevel,
)
from .modules import (
    CaregiverReporter,
    DeescalationAssessmentLayer,
    DeescalationCoach,
    HabitAssessmentLayer,
    IndependenceAssessmentLayer,
    IndependenceTaskEngine,
    PreventionHabitEngine,
    assess_calming_signal,
    build_journal_entry,
    build_prevention_sequence,
    classify_outburst_severity,
    matches_known_trigger,
    resolve_checklist_status,
    select_redirection,
)
from .prompts import DeescalationAssessment, HabitAssessment, IndependenceAssessment

logger = logging.getLogger("emotional_support_comfort.graph")

_CHECKPOINT_SERDE = JsonPlusSerializer(
    allowed_msgpack_modules=[
        DiseaseStage,
        ModuleId,
        SundowningRiskLevel,
        HabitType,
        OutburstSeverity,
        RedirectionType,
        PatientProfile,
        PreventionHabitItem,
        OutburstEvent,
        OutburstSignalLog,
        SessionTurn,
        SessionLog,
        SundowningJournalEntry,
        ProgressTrend,
    ]
)


class SessionState(TypedDict, total=False):
    patient: PatientProfile
    trigger_type: str
    outburst_event: Optional[OutburstEvent]
    module: ModuleId
    severity: Optional[OutburstSeverity]
    steps: list[PreventionHabitItem]
    step_index: int
    current_prompt: str
    current_item: Optional[PreventionHabitItem]
    current_redirection_type: Optional[RedirectionType]
    is_outburst_turn: bool
    caregiver_response: Optional[str]
    response_latency_s: float
    turn_count: int
    max_turns: int
    used_redirections: list[RedirectionType]
    last_calmed_signal: Optional[bool]
    session_log: SessionLog
    caregiver_summary: Optional[str]
    safety_alert: Optional[str]


def build_session_graph(llm: BaseChatModel):
    habit_engine = PreventionHabitEngine(llm)
    independence_engine = IndependenceTaskEngine(llm)
    deescalation_coach = DeescalationCoach(llm)
    habit_assessment_layer = HabitAssessmentLayer(llm)
    independence_assessment_layer = IndependenceAssessmentLayer(llm)
    deescalation_assessment_layer = DeescalationAssessmentLayer(llm)
    caregiver_reporter = CaregiverReporter(llm)

    def trigger_check(state: SessionState) -> SessionState:
        patient = state["patient"]
        if state.get("trigger_type") == "event_driven":
            event = state["outburst_event"]
            severity = classify_outburst_severity(event)
            session_log = SessionLog(
                patient_id=patient.patient_id,
                module=ModuleId.OUTBURST_DE_ESCALATION,
                trigger_type="event_driven",
                started_at=datetime.now().isoformat(timespec="seconds"),
                event_handled=f"{severity.value} — preceding: {event.preceding_event or 'not reported'}",
            )
            return {
                "module": ModuleId.OUTBURST_DE_ESCALATION,
                "severity": severity,
                "steps": [],
                "step_index": 0,
                "used_redirections": [],
                "last_calmed_signal": None,
                "session_log": session_log,
            }

        steps = build_prevention_sequence(patient)
        session_log = SessionLog(
            patient_id=patient.patient_id,
            module=ModuleId.SUNDOWNING_PREVENTION,
            trigger_type="scheduled_prevention",
            started_at=datetime.now().isoformat(timespec="seconds"),
        )
        return {
            "module": ModuleId.SUNDOWNING_PREVENTION,
            "steps": steps,
            "step_index": 0,
            "session_log": session_log,
        }

    def prepare_content(state: SessionState) -> SessionState:
        if state["module"] == ModuleId.OUTBURST_DE_ESCALATION:
            patient = state["patient"]
            used = state.get("used_redirections", [])
            redirection_type = select_redirection(patient, used)
            redirection_content = deescalation_coach.render_redirection_content(patient, redirection_type)
            prompt = deescalation_coach.generate(
                patient=patient,
                severity=state["severity"],
                attempt_number=state.get("step_index", 0) + 1,
                redirection_type=redirection_type,
                redirection_content=redirection_content,
                preceding_event=state["outburst_event"].preceding_event,
            )
            return {"current_prompt": prompt, "current_redirection_type": redirection_type, "is_outburst_turn": True}

        item = state["steps"][state.get("step_index", 0)]
        if item.habit_type == HabitType.INDEPENDENCE_TASK_OFFER:
            prompt = independence_engine.generate(state["patient"], item.description)
        else:
            prompt = habit_engine.generate(state["patient"], item)
        return {"current_prompt": prompt, "current_item": item, "is_outburst_turn": False}

    def deliver_and_capture(state: SessionState) -> SessionState:
        response = interrupt({"prompt": state["current_prompt"]})
        return {"caregiver_response": response["response"], "response_latency_s": response["latency"]}

    def adaptive_adjustment(state: SessionState) -> SessionState:
        response_text = state.get("caregiver_response")
        latency = state.get("response_latency_s", 0.0)
        session_log = state["session_log"]

        if state.get("is_outburst_turn"):
            redirection_type = state["current_redirection_type"]
            calmed = assess_calming_signal(response_text, latency)

            try:
                assessment: DeescalationAssessment = deescalation_assessment_layer.generate(
                    state["current_prompt"], response_text or ""
                )
                feedback = assessment.feedback
            except Exception:
                logger.exception("De-escalation assessment generation failed; using a fallback response")
                feedback = "Understood — keep going gently, I'm logging this."

            used = state.get("used_redirections", []) + [redirection_type]
            session_log.turns.append(
                SessionTurn(
                    module=ModuleId.OUTBURST_DE_ESCALATION,
                    prompt=state["current_prompt"],
                    caregiver_response=response_text,
                    feedback=feedback,
                    item_addressed=calmed,
                    signal=OutburstSignalLog(severity=state["severity"], calmed=calmed),
                )
            )
            session_log.redirection_history.append(redirection_type.value)

            return {
                "session_log": session_log,
                "used_redirections": used,
                "last_calmed_signal": calmed,
                "step_index": state.get("step_index", 0) + 1,
                "turn_count": state.get("turn_count", 0) + 1,
            }

        item: PreventionHabitItem = state["current_item"]
        if item.habit_type == HabitType.INDEPENDENCE_TASK_OFFER:
            try:
                independence_assessment: IndependenceAssessment = independence_assessment_layer.generate(
                    state["current_prompt"], response_text or ""
                )
                accepted, feedback = independence_assessment.accepted, independence_assessment.feedback
            except Exception:
                logger.exception("Independence-task assessment generation failed; using a fallback response")
                accepted, feedback = None, "Thank you for trying — every bit of participation counts."

            session_log.independence_task_offered = item.description
            session_log.independence_task_accepted = accepted
            session_log.turns.append(
                SessionTurn(
                    module=ModuleId.SUNDOWNING_PREVENTION,
                    prompt=state["current_prompt"],
                    caregiver_response=response_text,
                    feedback=feedback,
                    item_addressed=accepted,
                )
            )
        else:
            try:
                habit_assessment: HabitAssessment = habit_assessment_layer.generate(
                    state["current_prompt"], response_text or ""
                )
                addressed, feedback = habit_assessment.addressed, habit_assessment.feedback
            except Exception:
                logger.exception("Habit assessment generation failed; using a fallback response")
                addressed, feedback = None, "Thank you for checking on this."

            session_log.turns.append(
                SessionTurn(
                    module=ModuleId.SUNDOWNING_PREVENTION,
                    prompt=state["current_prompt"],
                    caregiver_response=response_text,
                    feedback=feedback,
                    item_addressed=addressed,
                )
            )
            session_log.prevention_items_reviewed += 1
            status_key = resolve_checklist_status(addressed)
            session_log.prevention_status_counts[status_key] = session_log.prevention_status_counts.get(status_key, 0) + 1

        return {
            "session_log": session_log,
            "step_index": state.get("step_index", 0) + 1,
            "turn_count": state.get("turn_count", 0) + 1,
        }

    def session_close(state: SessionState) -> SessionState:
        session_log = state["session_log"]
        module = state["module"]

        summary: Optional[str] = None
        safety_alert: Optional[str] = None

        if module == ModuleId.OUTBURST_DE_ESCALATION:
            calmed = state.get("last_calmed_signal")
            attempts = state.get("step_index", 0)
            session_log.deescalation_attempts = attempts
            # session_close for this module is only reached once calmed is True or attempts are
            # exhausted (see route_after_adjustment), so "not calmed" here always means escalate.
            session_log.outburst_active = calmed is not True
            session_log.escalated_to_contact = calmed is not True

            # Backfill the escalation flag onto the turn that actually triggered it — otherwise
            # every per-turn signal reads False even on the escalating turn.
            if session_log.turns and session_log.turns[-1].signal is not None:
                session_log.turns[-1].signal.escalated_to_contact = session_log.escalated_to_contact

            if calmed is True:
                session_log.ended_reason = "deescalated"
                outcome = f"De-escalated after {attempts} attempt(s)."
            else:
                session_log.ended_reason = "escalated_unresolved"
                outcome = f"Not calmed after {attempts} attempt(s); escalated to caregiver contact."

            session_log.journal_entry = build_journal_entry(
                event=state["outburst_event"],
                intervention_used=", ".join(session_log.redirection_history) or "none",
                outcome=outcome,
                started_at=session_log.started_at,
                matched_known_trigger=matches_known_trigger(state["outburst_event"], state["patient"]),
            )

            try:
                if session_log.escalated_to_contact:
                    safety_alert = caregiver_reporter.urgent_alert(session_log.model_dump_json())
                else:
                    summary = caregiver_reporter.summarize(session_log.model_dump_json())
            except Exception:
                logger.exception("Caregiver report generation failed; returning session without it")
        else:
            session_log.ended_reason = (
                "prevention_completed" if state.get("step_index", 0) >= len(state.get("steps", [])) else "max_turns_reached"
            )
            try:
                summary = caregiver_reporter.summarize(session_log.model_dump_json())
            except Exception:
                logger.exception("Caregiver report generation failed; returning session without it")

        return {"session_log": session_log, "caregiver_summary": summary, "safety_alert": safety_alert}

    def route_after_adjustment(state: SessionState) -> str:
        if state["module"] == ModuleId.OUTBURST_DE_ESCALATION:
            if state.get("last_calmed_signal") is True:
                return "session_close"
            if state.get("step_index", 0) >= state.get("max_turns", 3):
                return "session_close"
            return "prepare_content"

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
