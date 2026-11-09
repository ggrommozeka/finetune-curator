#!/usr/bin/env python3
"""Two-phase GitLab MR review export.

``sync`` talks to GitLab once and stores raw MRs under data/gitlab/ (gitignored).
``process``, ``list``, and ``export`` read only that store.

The token is read from GITLAB_TOKEN and is never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from scripts.gitlab.client import FixtureTransport, GitLabClient, GitLabError
from scripts.gitlab.examples import sft_to_dpo
from scripts.gitlab.link import DEFAULT_FIX_PATTERN, parse_user_time
from scripts.gitlab.store import GitlabStore, parse_mr_ref
from scripts.gitlab.sync import process_store, sync_store


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


def _store_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store", default="data/gitlab", help="Local store directory (gitignored)")


def _client_from_args(args) -> GitLabClient:
    if not args.base_url and not args.fixture:
        raise SystemExit("Pass --base-url or set GITLAB_BASE_URL")
    token = os.environ.get("GITLAB_TOKEN", "")
    if args.fixture:
        transport = FixtureTransport.from_file(Path(args.fixture))
        if not token:
            token = "gitlab-token-unset"
    else:
        transport = None
        if not token:
            raise SystemExit("Set GITLAB_TOKEN to a GitLab personal access token with api scope.")
    if args.insecure and args.ca_bundle:
        print("Warning: --insecure ignores --ca-bundle", file=sys.stderr)
    return GitLabClient(
        args.base_url or "https://gitlab.example.invalid",
        token,
        verify=not args.insecure,
        ca_file=None if args.insecure else args.ca_bundle,
        cache_dir=None,
        transport=transport,
        timeout=args.timeout,
    )


def _add_http_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default=os.environ.get("GITLAB_BASE_URL"), help="Self-hosted GitLab URL")
    parser.add_argument("--project", action="append", default=[], help="Project id or group/path. Repeatable.")
    parser.add_argument("--group", action="append", default=[], help="Group id or path. Repeatable.")
    parser.add_argument("--since", default=None, help="Inclusive start (YYYY-MM-DD or ISO timestamp)")
    parser.add_argument("--until", default=None, help="Inclusive end (YYYY-MM-DD or ISO timestamp)")
    parser.add_argument(
        "--refresh",
        action="append",
        default=[],
        help="Re-fetch one stored MR (project!iid or group/app!iid). Repeatable.",
    )
    parser.add_argument("--ca-bundle", default=None, help="PEM file for a private CA")
    parser.add_argument("--insecure", action="store_true", help="Disable TLS verification")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--fixture", default=None, help="Replay a recorded API fixture (no network)")


def _process_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mr", action="append", default=[], help="MR to process: project!iid. Repeatable.")
    parser.add_argument("--all", action="store_true", help="Process every MR in the store")
    parser.add_argument("--reviewer", default=None, help="Only notes by this GitLab username")
    parser.add_argument("--fix-pattern", default=DEFAULT_FIX_PATTERN, help="Case-insensitive regex for fix commits")
    parser.add_argument("--line-pad", type=int, default=2)
    parser.add_argument("--max-hunk-chars", type=int, default=8000)
    parser.add_argument("--anonymize", action="store_true", help="Strip author names and emails from examples")
    parser.add_argument("--anonymize-urls", action="store_true", help="Also replace http(s) URLs")
    parser.add_argument("--rules", action="store_true", help="Also propose harness rules from the new examples")
    parser.add_argument("--mock-response", default=None, help="Canned model reply for --rules (no network)")
    parser.add_argument("--rules-output", default="data/rule_proposals.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync GitLab MRs locally, then process them offline.")
    sub = parser.add_subparsers(dest="command", required=True)

    sync = sub.add_parser("sync", help="Fetch merged MRs into the local store")
    _store_arg(sync)
    _add_http_args(sync)

    listing = sub.add_parser("list", help="List stored MRs")
    _store_arg(listing)
    listing.add_argument("--project", default="")
    listing.add_argument("--author", default="")
    listing.add_argument("--reviewer", default="")
    listing.add_argument("--since", default="")
    listing.add_argument("--until", default="")
    listing.add_argument("--fixes-only", action="store_true")
    listing.add_argument("--min-comments", type=int, default=0)
    listing.add_argument("--fix-pattern", default=DEFAULT_FIX_PATTERN)

    process = sub.add_parser("process", help="Link comments and build examples offline")
    _store_arg(process)
    _process_flags(process)

    export = sub.add_parser("export", help="Write approved examples to SFT and DPO JSONL")
    _store_arg(export)
    export.add_argument("--mr", action="append", default=[])
    export.add_argument("--status", default="keep", choices=["keep", "discard", "undecided", "all"])
    export.add_argument("--sft", default="data/gitlab_sft.jsonl")
    export.add_argument("--dpo", default="data/gitlab_dpo.jsonl")
    export.add_argument("--no-dpo", action="store_true")
    export.add_argument("--dpo-from", default=None, help="Build DPO from an existing SFT JSONL and exit")
    export.add_argument("--rules", action="store_true")
    export.add_argument("--mock-response", default=None)
    export.add_argument("--rules-output", default="data/rule_proposals.json")
    return parser


def _compile_pattern(pattern: str) -> None:
    import re

    try:
        re.compile(pattern)
    except re.error as exc:
        raise SystemExit(f"Invalid --fix-pattern: {exc}") from exc


def _selected_refs(store: GitlabStore, args) -> list[str]:
    if args.all:
        return [item["ref"] for item in store.list_mrs(fix_pattern=args.fix_pattern)]
    if not args.mr:
        raise SystemExit("Pass --mr project!iid (repeatable) or --all")
    for ref in args.mr:
        parse_mr_ref(ref)
    return list(args.mr)


def _propose(records: list[dict], args) -> None:
    if not records:
        print("No examples to propose rules from.")
        return
    from scripts.rules.llm import chat_complete, thinking_controls
    from scripts.rules.propose import propose_rules
    from scripts.rules.store import upsert_proposals

    if args.mock_response:
        raw = Path(args.mock_response).read_text(encoding="utf-8")

        def complete_fn(_messages):
            return raw

        model = "mock"
        base_url = None
    else:
        import argparse

        from scripts.propose_rules import _resolve

        endpoint = _resolve(
            argparse.Namespace(
                base_url=None,
                api_key=None,
                model=None,
                from_opencode=None,
                opencode_config=None,
            )
        )
        if not endpoint.model:
            raise SystemExit(
                "Set FINETUNE_CURATOR_MODEL or OPENAI_MODEL for --rules, or pass --mock-response."
            )
        _suffix, extra = thinking_controls(True)

        def complete_fn(messages, extra_body=extra):
            return chat_complete(
                endpoint,
                messages,
                temperature=0.1,
                max_tokens=4096,
                timeout=600,
                extra_body=extra_body or None,
            )

        model = endpoint.model
        base_url = endpoint.base_url
    document = propose_rules(
        records,
        {"labels": {}, "notes": {}},
        complete_fn,
        sources=("gitlab_review",),
        model=model,
        base_url=base_url,
    )
    destination = Path(args.rules_output)
    for item in document.get("proposals") or []:
        item["source"] = "gitlab"
    upsert_proposals(destination, document.get("proposals") or [])
    print(f"Rules: {len(document.get('proposals') or [])} -> {destination}")


def _cmd_sync(args) -> None:
    if not args.project and not args.group and not args.refresh:
        raise SystemExit("Pass --project, --group, and/or --refresh project!iid")
    since = parse_user_time(args.since) if args.since else None
    until = parse_user_time(args.until, end_of_day=True) if args.until else None
    if since and until and since > until:
        raise SystemExit("--since is after --until")
    store = GitlabStore(args.store)
    try:
        client = _client_from_args(args)
        stats = sync_store(
            client,
            store,
            projects=args.project,
            groups=args.group,
            since=since,
            until=until,
            refresh=args.refresh,
        )
    except GitLabError as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        store.close()
    print(
        f"Fetched: {stats['fetched']}  skipped: {stats['skipped']}  refreshed: {stats['refreshed']}"
    )
    print(f"Store: {args.store}")


def _cmd_list(args) -> None:
    _compile_pattern(args.fix_pattern)
    store = GitlabStore(args.store)
    try:
        rows = store.list_mrs(
            project=args.project,
            author=args.author,
            reviewer=args.reviewer,
            since=args.since,
            until=args.until,
            fixes_only=args.fixes_only,
            min_comments=args.min_comments,
            fix_pattern=args.fix_pattern,
        )
    finally:
        store.close()
    if not rows:
        print("No stored merge requests.")
        return
    for row in rows:
        reviewers = ",".join(row["reviewers"]) or "-"
        print(
            f"{row['ref']}\t{row['project_path'] or row['project_id']}\t!{row['iid']}\t"
            f"{row['author_username']}\t{reviewers}\tcomments={row['comment_count']}\t"
            f"fixes={row['fix_count']}\t{row['merged_at']}\t{row['title']}"
        )


def _cmd_process(args) -> None:
    _compile_pattern(args.fix_pattern)
    store = GitlabStore(args.store)
    try:
        refs = _selected_refs(store, args)
        stats = process_store(
            store,
            refs,
            reviewer=args.reviewer,
            fix_pattern=args.fix_pattern,
            line_pad=args.line_pad,
            max_chars=args.max_hunk_chars,
            anonymize=args.anonymize,
            anonymize_urls=args.anonymize_urls,
        )
        print(
            "MRs: {merge_requests}  examples: {examples}  comments: {comments}  "
            "file+line: {file_line}  file: {file}  mr: {mr}  unlinked: {unlinked}".format(**stats)
        )
        if args.rules:
            keys = [store.resolve_ref(ref) for ref in refs]
            records = store.export_records(status="all", refs=keys)
            _propose(records, args)
    except KeyError as exc:
        raise SystemExit(f"MR not in the local store: {exc}") from exc
    finally:
        store.close()


def _cmd_export(args) -> None:
    if args.dpo_from:
        rows = []
        for record in _read_jsonl(Path(args.dpo_from)):
            converted = sft_to_dpo(record)
            if converted:
                rows.append(converted)
        _write_jsonl(Path(args.dpo), rows)
        print(f"DPO: {len(rows)} -> {args.dpo}")
        return
    store = GitlabStore(args.store)
    try:
        refs = [store.resolve_ref(ref) for ref in args.mr] if args.mr else None
        records = store.export_records(status=args.status, refs=refs)
    except KeyError as exc:
        raise SystemExit(f"MR not in the local store: {exc}") from exc
    finally:
        store.close()
    _write_jsonl(Path(args.sft), records)
    print(f"SFT: {len(records)} ({args.status}) -> {args.sft}")
    if not args.no_dpo:
        dpo_rows = [row for row in (sft_to_dpo(record) for record in records) if row]
        _write_jsonl(Path(args.dpo), dpo_rows)
        print(f"DPO: {len(dpo_rows)} -> {args.dpo}")
    if args.rules:
        _propose(records, args)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "sync":
        _cmd_sync(args)
    elif args.command == "list":
        _cmd_list(args)
    elif args.command == "process":
        _cmd_process(args)
    elif args.command == "export":
        _cmd_export(args)
    else:
        raise SystemExit(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
