"""Fetch merged MRs and turn review comments into SFT records."""

from __future__ import annotations

from datetime import datetime

from scripts.gitlab.client import merge_request_path
from scripts.gitlab.examples import apply_anonymization, build_sft_record
from scripts.gitlab.link import (
    DEFAULT_FIX_PATTERN,
    fix_commits,
    flatten_notes,
    in_range,
    link_comment,
    parse_time,
)


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


def records_for_mr(
    client,
    mr: dict,
    *,
    reviewer: str | None,
    fix_pattern: str,
    line_pad: int,
    max_chars: int,
) -> tuple[list[dict], dict]:
    project_id = mr.get("project_id")
    iid = mr.get("iid")
    stats = {"comments": 0, "file_line": 0, "file": 0, "mr": 0, "unlinked": 0}
    if project_id is None or iid is None:
        return [], stats
    base = f"/api/v4/projects/{project_id}/merge_requests/{iid}"
    discussions = client.get_all(base + "/discussions")
    commits = client.get_all(base + "/commits")
    notes = flatten_notes(discussions, reviewer)
    stats["comments"] = len(notes)
    fixes = fix_commits(commits, fix_pattern)
    diffs_by_sha: dict[str, list] = {}
    for commit in fixes:
        sha = str(commit.get("id") or "")
        if not sha:
            continue
        diffs_by_sha[sha] = client.get_all(f"/api/v4/projects/{project_id}/repository/commits/{sha}/diff")
    records = []
    for note in notes:
        linked = link_comment(note, fixes, diffs_by_sha, line_pad=line_pad)
        if not linked:
            stats["unlinked"] += 1
            continue
        commit = next((item for item in fixes if str(item.get("id") or "") == linked.get("commit_sha")), {})
        linked["commit_author"] = commit.get("author") if isinstance(commit.get("author"), dict) else {
            "name": commit.get("author_name"),
            "email": commit.get("author_email"),
        }
        record = build_sft_record(mr, note, linked, max_chars=max_chars)
        if record is None:
            stats["unlinked"] += 1
            continue
        kind = linked["link"]
        stats[kind] = stats.get(kind, 0) + 1
        records.append(record)
    return records, stats


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
) -> tuple[list[dict], dict]:
    mrs = list_merge_requests(client, projects, groups, since, until)
    records: list[dict] = []
    totals = {"merge_requests": len(mrs), "comments": 0, "file_line": 0, "file": 0, "mr": 0, "unlinked": 0}
    for mr in mrs:
        batch, stats = records_for_mr(
            client,
            mr,
            reviewer=reviewer,
            fix_pattern=fix_pattern,
            line_pad=line_pad,
            max_chars=max_chars,
        )
        records.extend(batch)
        for key in ("comments", "file_line", "file", "mr", "unlinked"):
            totals[key] += stats.get(key, 0)
    if anonymize or anonymize_urls:
        records = apply_anonymization(records, strip_urls=anonymize_urls)
    else:
        for record in records:
            record.pop("_authors", None)
    return records, totals
