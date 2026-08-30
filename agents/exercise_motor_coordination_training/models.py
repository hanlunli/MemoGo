from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from agents.shared.enums import DiseaseStage


class ModuleId(str, Enum):
    AEROBIC = "aerobic-exercise-session-engine"
    DUAL_TASK = "dual-task-training-coordinator"


class MobilityLevel(str, Enum):
    INDEPENDENT = "independent"
    NEEDS_SUPERVISION = "needs_supervision"
    USES_ASSISTIVE_DEVICE = "uses_assistive_device"


class IncidentFlag(str, Enum):
    NONE = "none"
    NEAR_FALL = "near_fall"
    FALL = "fall"
    DISTRESS = "distress"


class PatientProfile(BaseModel):
    patient_id: str
    name: str
    stage: DiseaseStage
    mobility_level: MobilityLevel = MobilityLevel.INDEPENDENT
    physical_limitations: list[str] = Field(default_factory=list)
    preferred_exercise_types: list[str] = Field(default_factory=list)
    indoor_outdoor_preference: str = "either"
    language: str = "en"


class ExerciseSessionItem(BaseModel):
    exercise_type: str
    motor_task: str
    cognitive_task: Optional[str] = None
    target_duration_min: int
    intensity_level: int


class MotorSignalLog(BaseModel):
    steadiness: Optional[str] = None
    support_needed: bool = False
    exertion_level: Optional[str] = None
    incident_flag: IncidentFlag = IncidentFlag.NONE


class SessionTurn(BaseModel):
    module: ModuleId
    prompt: str
    patient_response: Optional[str] = None
    feedback: Optional[str] = None
    sustained: Optional[bool] = None
    motor_signal: Optional[MotorSignalLog] = None


class SessionLog(BaseModel):
    patient_id: str
    module: ModuleId
    started_at: str
    turns: list[SessionTurn] = Field(default_factory=list)
    duration_achieved_min: float = 0.0
    engagement_score: float = 1.0
    fatigue_detected: bool = False
    safety_incident: bool = False
    ended_reason: Optional[str] = None


class ProgressTrend(BaseModel):
    domain: str
    rolling_duration_min: float
    rolling_engagement: float
    dual_task_sustain_rate: Optional[float] = None
    timeframe: str
