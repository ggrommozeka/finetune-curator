#!/usr/bin/env python3
"""Export merged GitLab MR review comments and 'MRR fixes' commits.

The token is read from GITLAB_TOKEN and is never printed. Raw responses can be
cached under data/gitlab_cache (gitignored). Outputs stay local.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from scripts.gitlab.client import FixtureTransport, GitLabClient, GitLabError
from scripts.gitlab.examples import sft_to_dpo
from scripts.gitlab.export import export_records
from scripts.gitlab.link import DEFAULT_FIX_PATTERN, parse_user_time


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export merged GitLab merge requests: review discussions plus diffs of "
            "commits whose message matches 'MRR fixes'."
        )
    )
    parser.add_argument("--base-url", default=os.environ.get("GITLAB_BASE_URL"), help="Self-hosted GitLab URL")
    parser.add_argument("--project", action="append", default=[], help="Project id or group/path. Repeatable.")
    parser.add_argument("--group", action="append", default=[], help="Group id or path. Repeatable. Includes group MRs.")
    parser.add_argument("--since", default=None, help="Inclusive start (YYYY-MM-DD or ISO timestamp)")
    parser.add_argument("--until", default=None, help="Inclusive end (YYYY-MM-DD or ISO timestamp)")
    parser.add_argument("--reviewer", default=None, help="Only notes by this GitLab username (team lead)")
    parser.add_argument(
        "--fix-pattern",
        default=DEFAULT_FIX_PATTERN,
        help="Case-insensitive regex matched against fix commit messages (default: mrr\\s*fixes)",
    )
    parser.add_argument("--line-pad", type=int, default=2, help="Extra lines when matching a comment to a hunk")
    parser.add_argument("--max-hunk-chars", type=int, default=8000)
    parser.add_argument("--sft", default="data/gitlab_sft.jsonl", help="Unsloth messages JSONL")
    parser.add_argument("--dpo", default="data/gitlab_dpo.jsonl", help="prompt/chosen/rejected JSONL")
    parser.add_argument("--no-dpo", action="store_true")
    parser.add_argument("--cache-dir", default="data/gitlab_cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--ca-bundle", default=None, help="PEM file for a private CA")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS verification")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--anonymize", action="store_true", help="Strip author names and emails from the JSONL")
    parser.add_argument("--anonymize-urls", action="store_true", help="Also replace http(s) URLs")
    parser.add_argument(
        "--fixture",
        default=None,
        help="Replay a recorded API fixture instead of the network (tests and smoke runs)",
    )
    parser.add_argument(
        "--dpo-from",
        default=None,
        help="Build a DPO JSONL from an existing SFT JSONL and exit (no API calls)",
    )
    return parser


def _dpo_only(path: Path, destination: Path) -> None:
    rows = []
    for record in _read_jsonl(path):
        converted = sft_to_dpo(record)
        if converted:
            rows.append(converted)
    _write_jsonl(destination, rows)
    print(f"DPO: {len(rows)} -> {destination}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.dpo_from:
        _dpo_only(Path(args.dpo_from), Path(args.dpo))
        return
    if not args.project and not args.group:
        raise SystemExit("Pass --project and/or --group")
    if not args.base_url and not args.fixture:
        raise SystemExit("Pass --base-url or set GITLAB_BASE_URL")
    try:
        import re

        re.compile(args.fix_pattern)
    except re.error as exc:
        raise SystemExit(f"Invalid --fix-pattern: {exc}") from exc

    token = os.environ.get("GITLAB_TOKEN", "")
    if args.fixture:
        transport = FixtureTransport.from_file(Path(args.fixture))
        if not token:
            token = "gitlab-token-unset"
    else:
        transport = None
        if not token:
            raise SystemExit("Set GITLAB_TOKEN to a GitLab personal access token with api scope.")
    since = parse_user_time(args.since) if args.since else None
    until = parse_user_time(args.until, end_of_day=True) if args.until else None
    if since and until and since > until:
        raise SystemExit("--since is after --until")
    if args.insecure and args.ca_bundle:
        print("Warning: --insecure ignores --ca-bundle", file=sys.stderr)

    try:
        client = GitLabClient(
            args.base_url or "https://gitlab.example.invalid",
            token,
            verify=not args.insecure,
            ca_file=None if args.insecure else args.ca_bundle,
            cache_dir=None if args.no_cache else Path(args.cache_dir),
            transport=transport,
            timeout=args.timeout,
        )
        records, stats = export_records(
            client,
            projects=args.project,
            groups=args.group,
            since=since,
            until=until,
            reviewer=args.reviewer,
            fix_pattern=args.fix_pattern,
            line_pad=args.line_pad,
            max_chars=args.max_hunk_chars,
            anonymize=args.anonymize,
            anonymize_urls=args.anonymize_urls,
        )
    except GitLabError as exc:
        raise SystemExit(str(exc)) from exc

    sft_path = Path(args.sft)
    _write_jsonl(sft_path, records)
    print(
        "MRs: {merge_requests}  comments: {comments}  "
        "file+line: {file_line}  file: {file}  mr: {mr}  unlinked: {unlinked}".format(**stats)
    )
    print(f"SFT: {len(records)} -> {sft_path}")
    if not args.no_dpo:
        dpo_rows = [row for row in (sft_to_dpo(record) for record in records) if row]
        dpo_path = Path(args.dpo)
        _write_jsonl(dpo_path, dpo_rows)
        print(f"DPO: {len(dpo_rows)} -> {dpo_path}")


if __name__ == "__main__":
    main()
