from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DiseaseStage(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class ModuleId(str, Enum):
    HOME_SAFETY_AUDIT = "home-safety-audit"
    INCIDENT_RESPONSE = "incident-response"


class MobilityLevel(str, Enum):
    INDEPENDENT = "independent"
    NEEDS_SUPERVISION = "needs_supervision"
    USES_ASSISTIVE_DEVICE = "uses_assistive_device"


class WanderingRiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class HazardType(str, Enum):
    FIRE_GAS = "fire_gas"
    FALL = "fall"
    WANDERING = "wandering"
    MEDICATION_CHEMICAL_ACCESS = "medication_chemical_access"


class Severity(str, Enum):
    INFO = "info"
    CAUTION = "caution"
    EMERGENCY = "emergency"


class ChecklistStatus(str, Enum):
    PENDING = "pending"
    DONE = "done"
    DEFERRED = "deferred"
    NEEDS_ATTENTION = "needs_attention"


class PatientProfile(BaseModel):
    patient_id: str
    name: str
    stage: DiseaseStage
    mobility_level: MobilityLevel = MobilityLevel.INDEPENDENT
    physical_limitations: list[str] = Field(default_factory=list)
    home_rooms: list[str] = Field(default_factory=lambda: ["Kitchen", "Bathroom", "Bedroom", "Hallway"])
    wandering_risk_level: WanderingRiskLevel = WanderingRiskLevel.LOW
    emergency_contacts: list[str] = Field(default_factory=list)
    language: str = "en"


class ChecklistItem(BaseModel):
    room: str
    hazard_type: HazardType
    mitigation_item: str
    is_high_priority: bool = False


class IncidentEvent(BaseModel):
    hazard_type: HazardType
    location: str
    source_device: str
    description: str = ""


class SafetySignalLog(BaseModel):
    severity: Severity = Severity.INFO
    resolved: Optional[bool] = None
    escalated_to_contact: bool = False


class SessionTurn(BaseModel):
    module: ModuleId
    prompt: str
    caregiver_response: Optional[str] = None
    feedback: Optional[str] = None
    item_resolved: Optional[bool] = None
    signal: Optional[SafetySignalLog] = None


class SessionLog(BaseModel):
    patient_id: str
    module: ModuleId
    trigger_type: str
    started_at: str
    turns: list[SessionTurn] = Field(default_factory=list)
    checklist_items_reviewed: int = 0
    checklist_status_counts: dict[str, int] = Field(default_factory=dict)
    event_handled: Optional[str] = None
    emergency_incident: bool = False
    escalated_to_contact: bool = False
    wandering_risk_level_after: Optional[WanderingRiskLevel] = None
    device_recommendation: Optional[str] = None
    ended_reason: Optional[str] = None


class ProgressTrend(BaseModel):
    domain: str
    rolling_checklist_completion_rate: float
    rolling_incident_frequency: float
    timeframe: str
