"""Skill files are written by the shared rules export. This module only re-exports it."""

from scripts.proposals.skills import render_skill_markdown, skill_id, write_skill

__all__ = ["render_skill_markdown", "skill_id", "write_skill"]
