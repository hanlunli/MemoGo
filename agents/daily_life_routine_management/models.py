from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from agents.shared.enums import DiseaseStage


class ModuleId(str, Enum):
    ROUTINE_CHECKPOINT = "routine-checkpoint"
    DEVIATION_RESPONSE = "deviation-response"


class MobilityLevel(str, Enum):
    INDEPENDENT = "independent"
    NEEDS_SUPERVISION = "needs_supervision"
    USES_ASSISTIVE_DEVICE = "uses_assistive_device"


class CheckpointType(str, Enum):
    WAKE_UP = "wake_up"
    DRESSING = "dressing"
    MEAL = "meal"
    MEDICATION = "medication"
    WALK = "walk"
    BEDTIME = "bedtime"


class DeviationType(str, Enum):
    SCHEDULE_SLIP = "schedule_slip"
    ENVIRONMENT_CHANGE = "environment_change"


class DeviationSeverity(str, Enum):
    MODERATE = "moderate"
    MAJOR = "major"


class CheckpointStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    NEEDS_ATTENTION = "needs_attention"


class PatientProfile(BaseModel):
    patient_id: str
    name: str
    stage: DiseaseStage
    mobility_level: MobilityLevel = MobilityLevel.INDEPENDENT
    physical_limitations: list[str] = Field(default_factory=list)
    dietary_restrictions: list[str] = Field(default_factory=list)
    seasonal_outfit_set: list[str] = Field(
        default_factory=lambda: ["blue zip-up cardigan and pants", "grey velcro-strap tracksuit"]
    )
    medication_names: list[str] = Field(default_factory=list)
    language: str = "en"


class RoutineCheckpoint(BaseModel):
    checkpoint_type: CheckpointType
    choice_options: list[str] = Field(default_factory=list)


class DeviationEvent(BaseModel):
    deviation_type: DeviationType
    checkpoint_type: CheckpointType
    description: str = ""


class RoutineSignalLog(BaseModel):
    status: CheckpointStatus = CheckpointStatus.PENDING
    resolved: Optional[bool] = None
    needs_followup: bool = False


class SessionTurn(BaseModel):
    module: ModuleId
    prompt: str
    caregiver_response: Optional[str] = None
    feedback: Optional[str] = None
    checkpoint_resolved: Optional[bool] = None
    signal: Optional[RoutineSignalLog] = None


class SessionLog(BaseModel):
    patient_id: str
    module: ModuleId
    trigger_type: str
    started_at: str
    turns: list[SessionTurn] = Field(default_factory=list)
    checkpoints_completed: int = 0
    checkpoint_status_counts: dict[str, int] = Field(default_factory=dict)
    event_handled: Optional[str] = None
    major_deviation: bool = False
    missed_medication_supervision: bool = False
    double_meal_prevented: bool = False
    needs_followup: bool = False
    ended_reason: Optional[str] = None


class ProgressTrend(BaseModel):
    domain: str
    rolling_routine_adherence_rate: float
    rolling_medication_adherence_rate: float
    timeframe: str
