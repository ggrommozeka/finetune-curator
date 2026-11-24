"""Shared OpenAI-compatible client, endpoint resolution, and JSON parsing."""

from scripts.llm.client import build_chat_kwargs, chat_complete, thinking_controls

__all__ = ["build_chat_kwargs", "chat_complete", "thinking_controls"]
