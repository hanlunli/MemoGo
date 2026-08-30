from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DiseaseStage(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class ModuleId(str, Enum):
    SCHEDULED_COMPANIONSHIP = "scheduled-companionship"
    DISTORTION_RESPONSE = "distortion-response"


class ActivityStepType(str, Enum):
    STAGE_ACTIVITY = "stage_activity"
    REMINISCENCE_THERAPY = "reminiscence_therapy"
    SELF_CARE_NUDGE = "self_care_nudge"


class SelfCareNudgeType(str, Enum):
    EMOTIONAL_BOUNDARY_REFRAME = "emotional_boundary_reframe"
    RESPITE_REMINDER = "respite_reminder"


class DistortionType(str, Enum):
    MISIDENTIFICATION = "misidentification"
    WANTS_TO_GO_HOME = "wants_to_go_home"
    ACCUSATION_OR_SUSPICION = "accusation_or_suspicion"
    OTHER = "other"


class PatientProfile(BaseModel):
    patient_id: str
    name: str
    stage: DiseaseStage
    biography: str = ""
    reminiscence_background: list[str] = Field(default_factory=list)
    sensitive_topics_to_avoid: list[str] = Field(default_factory=list)
    accepts_physical_contact: bool = True
    caregiver_name: str = ""
    days_since_last_respite: int = 0
    known_distortion_patterns: list[str] = Field(default_factory=list)


class StageActivityStep(BaseModel):
    step_type: ActivityStepType
    activity_key: str
    description: str


class ReminiscenceBoxItem(BaseModel):
    visual: str
    auditory: str
    tactile_olfactory: str


class RealityDistortionEvent(BaseModel):
    distortion_type: Optional[DistortionType] = None
    patient_statement: str = ""
    description: str = ""


class SessionTurn(BaseModel):
    module: ModuleId
    prompt: str
    caregiver_response: Optional[str] = None
    feedback: Optional[str] = None
    item_addressed: Optional[bool] = None


class SessionLog(BaseModel):
    patient_id: str
    module: ModuleId
    trigger_type: str
    started_at: str
    turns: list[SessionTurn] = Field(default_factory=list)
    activities_completed: int = 0
    activity_status_counts: dict[str, int] = Field(default_factory=dict)
    reminiscence_session_conducted: bool = False
    reminiscence_theme_used: Optional[str] = None
    self_care_nudge_delivered: Optional[str] = None
    self_care_nudge_accepted: Optional[bool] = None
    distortion_attempts: int = 0
    distortion_handled: Optional[str] = None
    ended_reason: Optional[str] = None


class ProgressTrend(BaseModel):
    domain: str
    rolling_engagement_rate: float
    rolling_reminiscence_session_count: float
    timeframe: str
