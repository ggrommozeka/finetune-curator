"""Render approved harness rules as paste-ready Markdown and JSON."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.rules.merge import category_heading, category_rank
from scripts.rules.store import approved_rules, load_proposals, load_rule_decisions, public_rule


def render_markdown(rules: list[dict]) -> str:
    grouped: dict[str, list[dict]] = {}
    for rule in rules:
        grouped.setdefault(rule.get("category") or "other", []).append(rule)
    lines = [
        "# Agent harness rules",
        "",
        "These instructions correct recurring mistakes. Follow them in addition to the task.",
        "",
    ]
    if not rules:
        lines.append("_No approved rules._")
        lines.append("")
        return "\n".join(lines)

    for category in sorted(grouped, key=lambda name: (category_rank(name), name)):
        lines.append(f"## {category_heading(category)}")
        lines.append("")
        ordered = sorted(grouped[category], key=lambda rule: rule.get("id") or "")
        for rule in ordered:
            text = " ".join(str(rule.get("text") or "").split())
            if text.startswith("- "):
                text = text[2:]
            lines.append(f"- {text}")
        lines.append("")
    return "\n".join(lines)


def render_json(rules: list[dict]) -> str:
    document = {"rules": [public_rule(rule) for rule in rules]}
    return json.dumps(document, ensure_ascii=False, indent=2) + "\n"


def export_approved(
    proposals_path: Path,
    decisions_path: Path,
    markdown_path: Path,
    json_path: Path,
) -> dict:
    proposals = load_proposals(proposals_path)
    decisions = load_rule_decisions(decisions_path)
    rules = approved_rules(proposals, decisions)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(rules), encoding="utf-8")
    json_path.write_text(render_json(rules), encoding="utf-8")
    return {
        "approved": len(rules),
        "proposed": len(proposals.get("rules") or []),
        "markdown": str(markdown_path),
        "json": str(json_path),
    }
