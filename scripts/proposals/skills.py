"""OpenCode skill files written from approved skill proposals.

Layout: ``<dir>/<skill-id>/SKILL.md`` with YAML ``name`` and ``description``.
OpenCode discovers ``.opencode/skills/<id>/SKILL.md`` and ``~/.config/opencode/skills``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_SLUG = re.compile(r"[^a-z0-9]+")


def skill_id(title: str) -> str:
    slug = _SLUG.sub("-", (title or "").lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)[:64].strip("-")
    return slug or "skill"


def render_skill_markdown(proposal: dict) -> str:
    title = proposal.get("title") or proposal.get("text") or proposal.get("id") or "skill"
    identifier = skill_id(str(title))
    description = str(proposal.get("description") or proposal.get("text") or "Task-specific agent skill")
    description = " ".join(description.split())
    lines = [
        "---",
        f"name: {identifier}",
        f"description: {json.dumps(description, ensure_ascii=False)}",
        "---",
        "",
        f"# {title}",
        "",
        "## When to use",
        "",
        description,
        "",
        "## Steps",
        "",
    ]
    steps = proposal.get("steps") or []
    if steps:
        lines.extend(f"{index}. {step}" for index, step in enumerate(steps, 1))
    else:
        lines.append(f"1. {proposal.get('text') or description}")
    lines.extend(["", "## Examples", ""])
    examples = proposal.get("examples") or []
    if examples:
        lines.extend(f"- {example}" for example in examples)
    else:
        sessions = proposal.get("session_ids") or []
        if sessions:
            lines.append("Drawn from sessions: " + ", ".join(str(sid) for sid in sessions[:8]))
        else:
            lines.append("- No example captured.")
    lines.append("")
    return "\n".join(lines)


def write_skill(directory: Path, proposal: dict) -> Path:
    title = proposal.get("title") or proposal.get("text") or proposal.get("id") or "skill"
    folder = directory / skill_id(str(title))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "SKILL.md"
    path.write_text(render_skill_markdown(proposal), encoding="utf-8")
    meta = {
        "id": proposal.get("id"),
        "kind": "skill",
        "skill_id": skill_id(str(title)),
        "session_ids": proposal.get("session_ids") or [],
        "evidence": proposal.get("evidence") or "",
    }
    (folder / "skill.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
