"""OpenCode skill files: `.opencode/skills/<id>/SKILL.md` with name and description."""

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
    payload = proposal.get("payload") or {}
    identifier = skill_id(payload.get("title") or proposal.get("id") or "skill")
    description = (payload.get("description") or payload.get("text") or "Task-specific agent skill").replace("\n", " ")
    lines = [
        "---",
        f"name: {identifier}",
        f"description: {json.dumps(description, ensure_ascii=False)}",
        "---",
        "",
        f"# {payload.get('title') or identifier}",
        "",
        "## When to use",
        "",
        description,
        "",
        "## Steps",
        "",
    ]
    steps = payload.get("steps") or []
    if steps:
        for index, step in enumerate(steps, 1):
            lines.append(f"{index}. {step}")
    else:
        lines.append(f"1. {payload.get('text') or description}")
    lines.extend(["", "## Examples", ""])
    examples = payload.get("examples") or []
    evidence = (proposal.get("evidence") or {}).get("session_ids") or []
    if examples:
        for example in examples:
            lines.append(f"- {example}")
    elif evidence:
        lines.append("Drawn from sessions: " + ", ".join(evidence[:8]))
    else:
        lines.append("- No example captured.")
    lines.append("")
    return "\n".join(lines)


def write_skill(directory: Path, proposal: dict) -> Path:
    identifier = skill_id((proposal.get("payload") or {}).get("title") or proposal["id"])
    folder = directory / identifier
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "SKILL.md"
    path.write_text(render_skill_markdown(proposal), encoding="utf-8")
    meta = {
        "id": proposal["id"],
        "skill_id": identifier,
        "cluster_id": proposal.get("cluster_id"),
        "evidence": proposal.get("evidence") or {},
        "conflicts": proposal.get("conflicts") or [],
    }
    (folder / "skill.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
