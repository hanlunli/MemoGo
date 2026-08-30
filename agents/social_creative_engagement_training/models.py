from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DiseaseStage(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class ModuleId(str, Enum):
    MUSIC_THERAPY = "music-therapy-session-engine"
    CRAFTS_HORTICULTURE = "crafts-horticulture-activity-engine"
    SOCIAL_INTERACTION = "social-interaction-facilitator"


class FineMotorLevel(str, Enum):
    INDEPENDENT = "independent"
    MILD_TREMOR = "mild_tremor"
    NEEDS_ASSIST = "needs_assist"


class IncidentFlag(str, Enum):
    NONE = "none"
    CUT = "cut"
    BURN = "burn"
    CHOKING_RISK = "choking_risk"


class SocialContact(BaseModel):
    name: str
    relationship: str
    contact_method: str = "in_person"


class PatientProfile(BaseModel):
    patient_id: str
    name: str
    stage: DiseaseStage
    fine_motor_level: FineMotorLevel = FineMotorLevel.INDEPENDENT
    material_sensitivities: list[str] = Field(default_factory=list)
    preferred_songs: list[str] = Field(default_factory=list)
    preferred_crafts: list[str] = Field(default_factory=list)
    conversation_topics: list[str] = Field(default_factory=list)
    social_contacts: list[SocialContact] = Field(default_factory=list)
    language: str = "en"


class MusicSessionItem(BaseModel):
    theme: str
    decade: str
    participation_mode: str


class CraftActivityItem(BaseModel):
    activity_type: str
    step_description: str
    hazard_flag: bool = False


class ActivitySignalLog(BaseModel):
    incident_flag: IncidentFlag = IncidentFlag.NONE
    agitation_detected: bool = False


class SessionTurn(BaseModel):
    module: ModuleId
    prompt: str
    patient_response: Optional[str] = None
    feedback: Optional[str] = None
    participation_confirmed: Optional[bool] = None
    signal: Optional[ActivitySignalLog] = None


class SessionLog(BaseModel):
    patient_id: str
    module: ModuleId
    started_at: str
    turns: list[SessionTurn] = Field(default_factory=list)
    engagement_score: float = 1.0
    agitation_detected: bool = False
    hazard_incident: bool = False
    ended_reason: Optional[str] = None


class ProgressTrend(BaseModel):
    domain: str
    rolling_mood_improvement: float
    rolling_fine_motor_steadiness: Optional[float] = None
    social_withdrawal_frequency: Optional[float] = None
    timeframe: str
