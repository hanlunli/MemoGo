from __future__ import annotations

from enum import Enum


class DiseaseStage(str, Enum):
    """Shared across every agent package instead of being redefined 8 times — the disease-stage
    taxonomy (mild/moderate/severe) is a single concept, not something each caregiving domain
    should be free to drift on independently."""

    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
