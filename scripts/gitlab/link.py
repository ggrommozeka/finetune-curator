"""Associate review comments with the MRR-fix commit that addressed them."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

FILE_LINE_CONFIDENCE = 0.9
FILE_CONFIDENCE = 0.55
MR_CONFIDENCE = 0.3

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
DEFAULT_FIX_PATTERN = r"mrr\s*fixes"


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
    """Link one note to the earliest later fix that best explains it.

    Preference: file + line overlap, then same file, then any later MRR-fix commit.
    Returns None when no fix commit is strictly after the comment.
    """
    path, line0, line1 = comment_target(note)
    try:
        note_time = parse_time(str(note.get("created_at") or ""))
    except ValueError:
        return None
    later = []
    for commit in fix_commits:
        try:
            commit_time = parse_time(str(commit.get("created_at") or commit.get("committed_date") or ""))
        except ValueError:
            continue
        if commit_time > note_time:
            later.append(commit)
    later.sort(key=lambda commit: parse_time(str(commit.get("created_at") or commit.get("committed_date"))))
    if not later:
        return None

    line_hit = None
    file_hit = None
    for commit in later:
        sha = str(commit.get("id") or commit.get("sha") or "")
        on_path, everything = _hunks_for(diffs_by_sha.get(sha) or [], path)
        if path and line0 is not None:
            matched = [hunk for hunk in on_path if _overlaps(hunk, line0, line1 or line0, line_pad)]
            if matched and line_hit is None:
                line_hit = (commit, matched, sha)
        if path and on_path and file_hit is None:
            file_hit = (commit, on_path, sha)
        if line_hit:
            break

    if line_hit:
        commit, hunks, sha = line_hit
        link = "file_line"
        confidence = FILE_LINE_CONFIDENCE
    elif file_hit:
        commit, hunks, sha = file_hit
        link = "file"
        confidence = FILE_CONFIDENCE
    else:
        commit = later[0]
        sha = str(commit.get("id") or commit.get("sha") or "")
        _on_path, hunks = _hunks_for(diffs_by_sha.get(sha) or [], None)
        link = "mr"
        confidence = MR_CONFIDENCE

    before = "\n\n".join(hunk.before for hunk in hunks if hunk.before).strip()
    after = "\n\n".join(hunk.after for hunk in hunks if hunk.after).strip()
    diff_text = "\n".join(hunk.text for hunk in hunks if hunk.text).strip()
    if not before and not after and not diff_text:
        return None
    return {
        "link": link,
        "confidence": confidence,
        "file": path,
        "line_start": line0,
        "line_end": line1,
        "commit_sha": sha,
        "commit_message": commit.get("message") or commit.get("title") or "",
        "commit_created_at": commit.get("created_at") or commit.get("committed_date"),
        "before": before,
        "after": after or diff_text,
        "diff": diff_text,
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
