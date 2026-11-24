"""Fetch merged MRs and turn review comments into SFT records."""

from __future__ import annotations

from datetime import datetime

from scripts.ingest.gitlab.client import merge_request_path
from scripts.ingest.gitlab.examples import apply_anonymization, build_sft_record
from scripts.ingest.gitlab.link import in_range, parse_time
from scripts.ingest.gitlab.signals import DEFAULT_MRR_PATTERN, SignalConfig


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


def linked_review(
    mr: dict,
    discussions: list,
    commits: list,
    diffs_by_sha: dict[str, list],
    config: SignalConfig,
    *,
    line_pad: int = 2,
    files: list | None = None,
    overrides: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """Traces for comment fixes, and the commit roles.

    Team-lead and MRR commits are omitted from the link. The same function is
    used by the live exporter and by offline ``process_store``.
    """
    from scripts.ingest.gitlab.classify import classify_commits, commits_for_comment_fixes
    from scripts.ingest.gitlab.trace import build_traces

    roles = classify_commits(commits, mr, discussions, config)
    traces = build_traces(
        discussions,
        commits_for_comment_fixes(commits, roles),
        diffs_by_sha,
        mr,
        line_pad=line_pad,
        files=files,
        overrides=overrides,
    )
    return traces, roles


def records_from_bundle(
    mr: dict,
    discussions: list,
    commits: list,
    diffs_by_sha: dict[str, list],
    *,
    line_pad: int,
    max_chars: int,
    signal_config: SignalConfig | None = None,
    files: list | None = None,
    overrides: dict | None = None,
    complete_fn=None,
) -> tuple[list[dict], dict, list[dict], list[dict]]:
    """Return records, stats, traces, and roles. No reviewer filter."""
    from scripts.ingest.gitlab.polish import build_polish_records

    config = signal_config or SignalConfig(team_leads=tuple(), mrr_pattern=DEFAULT_MRR_PATTERN)
    traces, roles = linked_review(
        mr,
        discussions,
        commits,
        diffs_by_sha,
        config,
        line_pad=line_pad,
        files=files,
        overrides=overrides,
    )
    records, stats = records_from_traces(mr, traces, max_chars=max_chars)
    polish = build_polish_records(
        mr, commits, diffs_by_sha, discussions, config, max_chars=max_chars, complete_fn=complete_fn
    )
    records.extend(polish)
    stats["polish"] = len(polish)
    return records, stats, traces, roles


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
        record["gitlab"]["reviewer"] = traced.get("reviewer") or traced.get("author_username") or ""
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
    line_pad: int,
    max_chars: int,
    signal_config: SignalConfig | None = None,
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
    records, stats, _traces, _roles = records_from_bundle(
        mr,
        discussions,
        commits,
        diffs_by_sha,
        line_pad=line_pad,
        max_chars=max_chars,
        signal_config=signal_config,
    )
    return records, stats


def export_records(
    client,
    *,
    projects: list[str],
    groups: list[str],
    since: datetime | None = None,
    until: datetime | None = None,
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
