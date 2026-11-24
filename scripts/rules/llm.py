"""Compatibility exports. Chat lives in scripts.llm; rule parsing in scripts.proposals.extract."""

from scripts.llm.client import build_chat_kwargs, chat_complete, thinking_controls
from scripts.proposals.extract import SYSTEM_PROMPT, coerce_rules, request_rules, session_ids_in_transcript

__all__ = [
    "SYSTEM_PROMPT",
    "build_chat_kwargs",
    "chat_complete",
    "coerce_rules",
    "request_rules",
    "session_ids_in_transcript",
    "thinking_controls",
]
