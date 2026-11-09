#!/usr/bin/env python3
"""Export approved harness rules as Markdown and JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.rules.export import export_approved, export_skills


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Write approved harness rules to Markdown (AGENTS.md / system prompt) and JSON."
    )
    parser.add_argument("--proposals", default="data/rule_proposals.json")
    parser.add_argument("--decisions", default="data/rule_decisions.json")
    parser.add_argument("--markdown", default="data/approved_rules.md")
    parser.add_argument("--json", default="data/approved_rules.json")
    parser.add_argument(
        "--skills-dir",
        default=None,
        help="Also write approved skill proposals as OpenCode SKILL.md directories",
    )
    args = parser.parse_args(argv)

    proposals = Path(args.proposals)
    if not proposals.exists():
        raise SystemExit(f"Proposals file not found: {proposals}")
    result = export_approved(
        proposals,
        Path(args.decisions),
        Path(args.markdown),
        Path(args.json),
    )
    print(f"Approved rules: {result['approved']} / {result['proposed']} proposed")
    print(f"Markdown: {result['markdown']}")
    print(f"JSON:     {result['json']}")
    if args.skills_dir:
        skills = export_skills(proposals, Path(args.decisions), Path(args.skills_dir))
        print(f"Skills: {skills['skills']} -> {args.skills_dir}")


if __name__ == "__main__":
    main()
