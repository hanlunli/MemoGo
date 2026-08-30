from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DiseaseStage(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class ModuleId(str, Enum):
    SUNDOWNING_PREVENTION = "sundowning-prevention"
    OUTBURST_DE_ESCALATION = "outburst-de-escalation"


class SundowningRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class HabitType(str, Enum):
    MORNING_SUNLIGHT = "morning_sunlight"
    NAP_CAFFEINE_SUGAR_LIMIT = "nap_caffeine_sugar_limit"
    ENVIRONMENT_LIGHTING = "environment_lighting"
    EVENING_ROUTINE_PROTECTION = "evening_routine_protection"
    BEDTIME_WARM_DRINK = "bedtime_warm_drink"
    INDEPENDENCE_TASK_OFFER = "independence_task_offer"


class OutburstSeverity(str, Enum):
    RESTLESSNESS = "restlessness"
    OUTBURST = "outburst"


class RedirectionType(str, Enum):
    FAMILIAR_ITEM = "familiar_item"
    MUSIC = "music"
    PHOTO_ALBUM = "photo_album"
    SNACK = "snack"
    HANDS_ON_TASK = "hands_on_task"


class PatientProfile(BaseModel):
    patient_id: str
    name: str
    stage: DiseaseStage
    biography: str = ""
    sundowning_risk_level: SundowningRiskLevel = SundowningRiskLevel.LOW
    typical_onset_time: str = "16:00"
    known_triggers: list[str] = Field(default_factory=list)
    calming_preferences: list[str] = Field(default_factory=list)
    accepts_physical_contact: bool = True
    independence_tasks: list[str] = Field(default_factory=lambda: ["folding clothes", "watering plants"])
    emergency_contacts: list[str] = Field(default_factory=list)
    language: str = "en"


class PreventionHabitItem(BaseModel):
    habit_type: HabitType
    description: str
    is_high_priority: bool = False


class OutburstEvent(BaseModel):
    preceding_event: str = ""
    environmental_factor: str = ""
    food_or_drink_intake: str = ""
    symptoms_observed: list[str] = Field(default_factory=list)
    description: str = ""


class OutburstSignalLog(BaseModel):
    severity: OutburstSeverity
    calmed: Optional[bool] = None
    escalated_to_contact: bool = False


class SessionTurn(BaseModel):
    module: ModuleId
    prompt: str
    caregiver_response: Optional[str] = None
    feedback: Optional[str] = None
    item_addressed: Optional[bool] = None
    signal: Optional[OutburstSignalLog] = None


class SundowningJournalEntry(BaseModel):
    date: str
    time_of_onset: str
    preceding_event: str
    food_or_drink_intake: str
    environmental_factor: str
    symptoms_observed: list[str] = Field(default_factory=list)
    intervention_used: str
    outcome: str


class SessionLog(BaseModel):
    patient_id: str
    module: ModuleId
    trigger_type: str
    started_at: str
    turns: list[SessionTurn] = Field(default_factory=list)
    prevention_items_reviewed: int = 0
    prevention_status_counts: dict[str, int] = Field(default_factory=dict)
    event_handled: Optional[str] = None
    outburst_active: bool = False
    escalated_to_contact: bool = False
    deescalation_attempts: int = 0
    redirection_history: list[str] = Field(default_factory=list)
    independence_task_offered: Optional[str] = None
    independence_task_accepted: Optional[bool] = None
    journal_entry: Optional[SundowningJournalEntry] = None
    ended_reason: Optional[str] = None


class ProgressTrend(BaseModel):
    domain: str
    rolling_deescalation_success_rate: float
    rolling_episode_frequency: float
    timeframe: str
