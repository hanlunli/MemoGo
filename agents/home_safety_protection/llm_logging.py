from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

logger = logging.getLogger("home_safety_protection.llm")

DEFAULT_LOG_FILE = Path("logs") / "llm_calls.log"


def configure_logging(log_file: Path | str = DEFAULT_LOG_FILE, level: int = logging.INFO) -> None:
    """Log to both console and a file. Safe to call multiple times (no duplicate handlers)."""
    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(console_handler)

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    root.addHandler(file_handler)


def _format_messages(messages: list[BaseMessage]) -> str:
    return "\n".join(f"  [{message.type}] {message.content}" for message in messages)


class LLMUsageLoggingHandler(BaseCallbackHandler):
    """Logs the prompt, response, and token usage for every LLM call."""

    def __init__(self) -> None:
        self._start_times: dict[UUID, float] = {}

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._start_times[run_id] = time.monotonic()
        prompt_text = _format_messages(messages[0]) if messages else ""
        logger.info("LLM call -> prompt:\n%s", prompt_text)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        elapsed = time.monotonic() - self._start_times.pop(run_id, time.monotonic())
        generation = response.generations[0][0]
        message = getattr(generation, "message", None)

        if message is not None and getattr(message, "tool_calls", None):
            output_text = f"tool_calls={message.tool_calls}"
        else:
            output_text = getattr(message, "content", None) or generation.text

        usage = getattr(message, "usage_metadata", None) if message is not None else None
        if usage:
            token_summary = (
                f"input={usage.get('input_tokens')} "
                f"output={usage.get('output_tokens')} "
                f"total={usage.get('total_tokens')}"
            )
        else:
            token_summary = "unavailable"

        logger.info(
            "LLM call <- response (%.2fs):\n%s\nTokens: %s",
            elapsed,
            output_text,
            token_summary,
        )
