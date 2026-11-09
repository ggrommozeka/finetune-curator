"""Fetch merged MRs and turn review comments into SFT records."""

from __future__ import annotations

from datetime import datetime

from scripts.gitlab.client import merge_request_path
from scripts.gitlab.examples import apply_anonymization, build_sft_record
from scripts.gitlab.link import DEFAULT_FIX_PATTERN, in_range, parse_time


def _mr_moment(mr: dict) -> datetime | None:
    for key in ("merged_at", "updated_at", "created_at"):
        raw = mr.get(key)
        if not raw:
            continue
        try:
            return parse_time(str(raw))
        except ValueError:
            continue
    return None


def list_merge_requests(client, projects: list[str], groups: list[str], since: datetime | None, until: datetime | None) -> list[dict]:
    params: dict = {
        "state": "merged",
        "scope": "all",
        "order_by": "updated_at",
        "sort": "desc",
    }
    if since is not None:
        params["updated_after"] = since.isoformat()
    if until is not None:
        params["updated_before"] = until.isoformat()
    found: list[dict] = []
    seen: set[tuple] = set()
    queries = [("project", ident) for ident in projects] + [("group", ident) for ident in groups]
    for kind, ident in queries:
        rows = client.get_all(merge_request_path(kind, ident), params)
        for mr in rows:
            moment = _mr_moment(mr)
            if moment is not None and not in_range(moment, since, until):
                continue
            key = (mr.get("project_id"), mr.get("iid"))
            if key in seen:
                continue
            seen.add(key)
            found.append(mr)
    return found


def records_from_bundle(
    mr: dict,
    discussions: list,
    commits: list,
    diffs_by_sha: dict[str, list],
    *,
    reviewer: str | None,
    fix_pattern: str,
    line_pad: int,
    max_chars: int,
    signal_config=None,
) -> tuple[list[dict], dict]:
    """Link comments to fixes using only data already in memory."""
    from scripts.gitlab.trace import build_traces

    traces = build_traces(
        discussions,
        commits,
        diffs_by_sha,
        mr,
        reviewer=reviewer,
        fix_pattern=fix_pattern,
        line_pad=line_pad,
    )
    records, stats = records_from_traces(mr, traces, max_chars=max_chars)
    from scripts.gitlab.polish import build_polish_records
    from scripts.gitlab.signals import SignalConfig

    config = signal_config or SignalConfig(team_leads=tuple(), mrr_pattern=fix_pattern)
    polish = build_polish_records(mr, commits, diffs_by_sha, discussions, config, max_chars=max_chars)
    records.extend(polish)
    stats["polish"] = len(polish)
    return records, stats


def records_from_traces(mr: dict, traces: list[dict], max_chars: int) -> tuple[list[dict], dict]:
    stats = {"comments": len(traces), "file_line": 0, "file": 0, "mr": 0, "unlinked": 0, "linked": 0}
    records = []
    for traced in traces:
        status = traced.get("status")
        if status == "linked":
            stats["linked"] += 1
        if status not in {"linked", "addressed_elsewhere"}:
            stats["unlinked"] += 1
            continue
        note = {
            "id": traced.get("note_id"),
            "body": traced.get("body") or "",
            "created_at": traced.get("created_at") or "",
            "resolved": traced.get("resolved"),
            "author": {
                "username": traced.get("author_username") or "",
                "name": traced.get("author_name") or "",
            },
        }
        anchor = traced.get("anchor") or {}
        linked = {
            "link": traced.get("link"),
            "confidence": traced.get("confidence"),
            "file": anchor.get("file"),
            "line_start": anchor.get("new_line_start") or anchor.get("old_line_start"),
            "line_end": anchor.get("new_line_end") or anchor.get("old_line_end"),
            "commit_sha": traced.get("commit_sha") or "",
            "commit_message": traced.get("commit_message") or "",
            "commit_created_at": traced.get("commit_created_at") or "",
            "before": traced.get("before") or "",
            "after": traced.get("after") or "",
            "diff": traced.get("diff") or "",
            "commit_author": traced.get("commit_author") or {},
        }
        record = build_sft_record(mr, note, linked, max_chars=max_chars)
        if record is None:
            stats["unlinked"] += 1
            continue
        record["gitlab"]["link_status"] = status
        record["gitlab"]["author_role"] = traced.get("author_role") or ""
        record["gitlab"]["resolved_by"] = traced.get("resolved_by") or ""
        record["gitlab"]["replies"] = traced.get("replies") or []
        record["gitlab"]["anchor"] = anchor
        record["gitlab"]["manual"] = bool(traced.get("manual"))
        kind = linked["link"]
        if kind in stats:
            stats[kind] += 1
        records.append(record)
    return records, stats


def records_for_mr(
    client,
    mr: dict,
    *,
    reviewer: str | None,
    fix_pattern: str,
    line_pad: int,
    max_chars: int,
    signal_config=None,
) -> tuple[list[dict], dict]:
    project_id = mr.get("project_id")
    iid = mr.get("iid")
    if project_id is None or iid is None:
        return [], {"comments": 0, "file_line": 0, "file": 0, "mr": 0, "unlinked": 0}
    base = f"/api/v4/projects/{project_id}/merge_requests/{iid}"
    discussions = client.get_all(base + "/discussions")
    commits = client.get_all(base + "/commits")
    diffs_by_sha: dict[str, list] = {}
    for commit in commits:
        sha = str(commit.get("id") or "")
        if not sha:
            continue
        diffs_by_sha[sha] = client.get_all_optional(f"/api/v4/projects/{project_id}/repository/commits/{sha}/diff")
    return records_from_bundle(
        mr,
        discussions,
        commits,
        diffs_by_sha,
        reviewer=reviewer,
        fix_pattern=fix_pattern,
        line_pad=line_pad,
        max_chars=max_chars,
        signal_config=signal_config,
    )


def export_records(
    client,
    *,
    projects: list[str],
    groups: list[str],
    since: datetime | None = None,
    until: datetime | None = None,
    reviewer: str | None = None,
    fix_pattern: str = DEFAULT_FIX_PATTERN,
    line_pad: int = 2,
    max_chars: int = 8000,
    anonymize: bool = False,
    anonymize_urls: bool = False,
    signal_config=None,
) -> tuple[list[dict], dict]:
    mrs = list_merge_requests(client, projects, groups, since, until)
    records: list[dict] = []
    totals = {"merge_requests": len(mrs), "comments": 0, "file_line": 0, "file": 0, "mr": 0, "unlinked": 0, "polish": 0}
    for mr in mrs:
        batch, stats = records_for_mr(
            client,
            mr,
            reviewer=reviewer,
            fix_pattern=fix_pattern,
            line_pad=line_pad,
            max_chars=max_chars,
            signal_config=signal_config,
        )
        records.extend(batch)
        for key in ("comments", "file_line", "file", "mr", "unlinked", "polish"):
            totals[key] += stats.get(key, 0)
    if anonymize or anonymize_urls:
        records = apply_anonymization(records, strip_urls=anonymize_urls)
    else:
        for record in records:
            record.pop("_authors", None)
    return records, totals
