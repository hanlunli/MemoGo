from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DiseaseStage(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class ModuleId(str, Enum):
    PERSONAL_CARE = "personal-care-routine"
    HOUSEHOLD_CHORES = "household-chores-routine"


class MobilityLevel(str, Enum):
    INDEPENDENT = "independent"
    NEEDS_SUPERVISION = "needs_supervision"
    USES_ASSISTIVE_DEVICE = "uses_assistive_device"


class AssistanceLevel(str, Enum):
    INDEPENDENT = "independent"
    VERBAL_CUE = "verbal_cue"
    DEMONSTRATION = "demonstration"
    PHYSICAL_ASSIST = "physical_assist"


class IncidentFlag(str, Enum):
    NONE = "none"
    BURN = "burn"
    CUT = "cut"
    FALL = "fall"
    DISTRESS = "distress"


class PatientProfile(BaseModel):
    patient_id: str
    name: str
    stage: DiseaseStage
    mobility_level: MobilityLevel = MobilityLevel.INDEPENDENT
    physical_limitations: list[str] = Field(default_factory=list)
    preferred_adl_tasks: list[str] = Field(default_factory=list)
    language: str = "en"


class ADLTaskStep(BaseModel):
    task_name: str
    step_index: int
    step_description: str
    assistance_level: AssistanceLevel
    is_hazard_step: bool = False


class TaskSignalLog(BaseModel):
    assistance_level_used: AssistanceLevel
    incident_flag: IncidentFlag = IncidentFlag.NONE
    frustration_detected: bool = False


class SessionTurn(BaseModel):
    module: ModuleId
    prompt: str
    patient_response: Optional[str] = None
    feedback: Optional[str] = None
    step_completed: Optional[bool] = None
    signal: Optional[TaskSignalLog] = None


class SessionLog(BaseModel):
    patient_id: str
    module: ModuleId
    task_name: str
    started_at: str
    turns: list[SessionTurn] = Field(default_factory=list)
    steps_completed: int = 0
    assistance_level_counts: dict[str, int] = Field(default_factory=dict)
    engagement_score: float = 1.0
    frustration_detected: bool = False
    hazard_incident: bool = False
    ended_reason: Optional[str] = None


class ProgressTrend(BaseModel):
    task_type: str
    rolling_independence_score: float
    rolling_completion_rate: float
    timeframe: str
