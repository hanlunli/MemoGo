from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from agents.shared.enums import DiseaseStage


class ModuleId(str, Enum):
    REALITY_ORIENTATION = "reality-orientation-engine"
    REMINISCENCE = "reminiscence-session-manager"
    EXERCISE = "cognitive-exercise-generator"


class PatientProfile(BaseModel):
    patient_id: str
    name: str
    stage: DiseaseStage
    biography: str = ""
    preferences: list[str] = Field(default_factory=list)
    triggers: list[str] = Field(default_factory=list)
    language: str = "en"


class MemoryItem(BaseModel):
    media_type: str
    theme: str
    decade: str
    description: str = ""


class ExerciseItem(BaseModel):
    type: str
    domain: str
    difficulty: int
    content: str
    answer: Optional[str] = None


class ProgressTrend(BaseModel):
    domain: str
    rolling_accuracy: float
    rolling_engagement: float
    timeframe: str


class SessionTurn(BaseModel):
    module: ModuleId
    prompt: str
    patient_response: Optional[str] = None
    feedback: Optional[str] = None
    correct: Optional[bool] = None
    domain: Optional[str] = None


class SessionLog(BaseModel):
    patient_id: str
    module: ModuleId
    started_at: str
    turns: list[SessionTurn] = Field(default_factory=list)
    engagement_score: float = 1.0
    fatigue_detected: bool = False
    ended_reason: Optional[str] = None
    progress_trends: list[ProgressTrend] = Field(default_factory=list)
