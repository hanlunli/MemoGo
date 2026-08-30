from __future__ import annotations

from agents.shared.llm_logging import DEFAULT_LOG_FILE, configure_logging
from agents.shared.llm_logging import LLMUsageLoggingHandler as _SharedLLMUsageLoggingHandler

__all__ = ["DEFAULT_LOG_FILE", "LLMUsageLoggingHandler", "configure_logging"]


class LLMUsageLoggingHandler(_SharedLLMUsageLoggingHandler):
    def __init__(self) -> None:
        super().__init__("social_creative_engagement_training.llm")
