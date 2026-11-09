"""Associate review comments with the MRR-fix commit that addressed them."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

FILE_LINE_CONFIDENCE = 0.8
FILE_CONFIDENCE = 0.55
MR_CONFIDENCE = 0.3

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
DEFAULT_FIX_PATTERN = r"\bmrr\b"


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    text: str
    before: str
    after: str


def is_fix_message(message: str, pattern: str = DEFAULT_FIX_PATTERN) -> bool:
    if not message or not pattern:
        return False
    return re.search(pattern, message, flags=re.IGNORECASE) is not None


def parse_time(value: str) -> datetime:
    text = (value or "").strip()
    if not text:
        raise ValueError("empty timestamp")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def parse_user_time(value: str, *, end_of_day: bool = False) -> datetime:
    """Parse a CLI date or timestamp. A bare date is UTC midnight, or end of that day."""
    text = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        hour, minute, second = (23, 59, 59) if end_of_day else (0, 0, 0)
        return datetime.fromisoformat(f"{text}T{hour:02d}:{minute:02d}:{second:02d}+00:00")
    return parse_time(text)


def in_range(moment: datetime, since: datetime | None, until: datetime | None) -> bool:
    if since and moment < since:
        return False
    if until and moment > until:
        return False
    return True


def _span(start: int, count: int) -> tuple[int, int]:
    if count <= 0:
        return start, start
    return start, start + count - 1


def _before_after(lines: list[str]) -> tuple[str, str]:
    before: list[str] = []
    after: list[str] = []
    for line in lines:
        if line.startswith("@@") or line.startswith("\\"):
            continue
        if line.startswith("+"):
            after.append(line[1:])
        elif line.startswith("-"):
            before.append(line[1:])
        elif line.startswith(" "):
            before.append(line[1:])
            after.append(line[1:])
        else:
            before.append(line)
            after.append(line)
    return "\n".join(before).strip("\n"), "\n".join(after).strip("\n")


def parse_hunks(diff: str) -> list[Hunk]:
    if not diff:
        return []
    hunks: list[Hunk] = []
    header: re.Match | None = None
    bucket: list[str] = []

    def flush() -> None:
        nonlocal header, bucket
        if header is None:
            return
        old_start = int(header.group(1))
        old_count = int(header.group(2) if header.group(2) is not None else 1)
        new_start = int(header.group(3))
        new_count = int(header.group(4) if header.group(4) is not None else 1)
        before, after = _before_after(bucket)
        hunks.append(
            Hunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                text="\n".join(bucket),
                before=before,
                after=after,
            )
        )
        header = None
        bucket = []

    for line in diff.splitlines():
        match = _HUNK.match(line)
        if match:
            flush()
            header = match
            bucket = [line]
        elif header is not None:
            bucket.append(line)
    flush()
    return hunks


def _overlaps(hunk: Hunk, line0: int, line1: int, pad: int) -> bool:
    old0, old1 = _span(hunk.old_start, hunk.old_count)
    new0, new1 = _span(hunk.new_start, hunk.new_count)
    return (line0 - pad) <= old1 and (old0 - pad) <= line1 or (line0 - pad) <= new1 and (new0 - pad) <= line1


def comment_target(note: dict) -> tuple[str | None, int | None, int | None]:
    """Return (path, first_line, last_line). Lines are absent for MR-level notes."""
    position = note.get("position") or {}
    if not isinstance(position, dict):
        position = {}
    path = position.get("new_path") or position.get("old_path") or None
    numbers: list[int] = []
    line_range = position.get("line_range") or {}
    if isinstance(line_range, dict):
        for edge in (line_range.get("start"), line_range.get("end")):
            if not isinstance(edge, dict):
                continue
            for key in ("new_line", "old_line"):
                if edge.get(key):
                    numbers.append(int(edge[key]))
    for key in ("new_line", "old_line"):
        if position.get(key):
            numbers.append(int(position[key]))
    if not numbers:
        return path, None, None
    return path, min(numbers), max(numbers)


def _diff_paths(diff: dict) -> set[str]:
    paths = set()
    for key in ("old_path", "new_path"):
        if diff.get(key):
            paths.add(str(diff[key]))
    return paths


def _hunks_for(diffs: list[dict], path: str | None) -> tuple[list[Hunk], list[Hunk]]:
    """Return (line-match candidates' hunks are filtered by caller, all hunks on path)."""
    on_path: list[Hunk] = []
    everything: list[Hunk] = []
    for diff in diffs or []:
        hunks = parse_hunks(diff.get("diff") or "")
        everything.extend(hunks)
        if path and path in _diff_paths(diff):
            on_path.extend(hunks)
    return on_path, everything


def link_comment(
    note: dict,
    fix_commits: list[dict],
    diffs_by_sha: dict[str, list[dict]],
    *,
    line_pad: int = 2,
) -> dict | None:
    """Link one note. ``MRR fixes`` commits are preferred; other later commits still count.

    Line numbers are replayed through each later diff, so a fix is found after
    earlier commits shift the anchored line. Returns None when nothing after
    the comment can be tied to a hunk.
    """
    from scripts.gitlab.trace import trace_thread

    traced = trace_thread(
        {"root": note, "replies": [], "discussion_id": ""},
        fix_commits,
        diffs_by_sha,
        {},
        line_pad=line_pad,
    )
    if traced["status"] in {"not_addressed", "unclear"} and not traced.get("diff"):
        return None
    if not traced.get("before") and not traced.get("after") and not traced.get("diff"):
        return None
    anchor = traced.get("anchor") or {}
    return {
        "link": traced["link"],
        "status": traced["status"],
        "confidence": traced["confidence"],
        "file": anchor.get("file"),
        "line_start": anchor.get("new_line_start") or anchor.get("old_line_start"),
        "line_end": anchor.get("new_line_end") or anchor.get("old_line_end"),
        "commit_sha": traced.get("commit_sha") or "",
        "commit_message": traced.get("commit_message") or "",
        "commit_created_at": traced.get("commit_created_at") or "",
        "before": traced.get("before") or "",
        "after": traced.get("after") or "",
        "diff": traced.get("diff") or "",
    }


def flatten_notes(discussions: list[dict], reviewer: str | None = None) -> list[dict]:
    """Diff and discussion notes, skipping system notes and other authors."""
    wanted = reviewer.lower() if reviewer else None
    notes: list[dict] = []
    for discussion in discussions or []:
        for note in discussion.get("notes") or []:
            if not isinstance(note, dict) or note.get("system"):
                continue
            body = str(note.get("body") or "").strip()
            if not body:
                continue
            username = str((note.get("author") or {}).get("username") or "")
            if wanted and username.lower() != wanted:
                continue
            notes.append(note)
    return notes


def fix_commits(commits: list[dict], pattern: str = DEFAULT_FIX_PATTERN) -> list[dict]:
    found = []
    for commit in commits or []:
        message = "\n".join(
            part for part in (commit.get("message"), commit.get("title")) if part
        )
        if is_fix_message(message, pattern):
            found.append(commit)
    return found
