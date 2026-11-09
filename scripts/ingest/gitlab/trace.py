"""Comment anchors and how later commits move or change those lines.

Every human review note keeps the diff version it was attached to (SHAs, path,
old/new line range, snippet). Later commits are replayed in order so a fix is
found even when earlier commits shifted the line. Commit titles are ignored:
a fix is a later commit that touches the anchored file or lines.
"""

from __future__ import annotations

from scripts.ingest.gitlab.link import (
    FILE_CONFIDENCE,
    FILE_LINE_CONFIDENCE,
    Hunk,
    _span,
    comment_target,
    parse_hunks,
    parse_time,
)

OTHER_LINE_CONFIDENCE = 0.7
MANUAL_CONFIDENCE = 1.0


def author_role(mr: dict | None, username: str) -> str:
    mr = mr or {}
    username = username or ""
    author = (mr.get("author") or {}).get("username") or ""
    if username and username == author:
        return "author"
    reviewers = {(item or {}).get("username") or "" for item in (mr.get("reviewers") or [])}
    if username and username in reviewers:
        return "reviewer"
    assignees = {(item or {}).get("username") or "" for item in (mr.get("assignees") or [])}
    if username and username in assignees:
        return "assignee"
    return "commenter"


def extract_anchor(note: dict) -> dict:
    position = note.get("position") if isinstance(note.get("position"), dict) else {}
    line_range = position.get("line_range") if isinstance(position.get("line_range"), dict) else {}
    start = line_range.get("start") if isinstance(line_range.get("start"), dict) else {}
    end = line_range.get("end") if isinstance(line_range.get("end"), dict) else {}

    def nums(*values) -> list[int]:
        found = []
        for value in values:
            if value is None or value == "":
                continue
            found.append(int(value))
        return found

    old_lines = nums(start.get("old_line"), end.get("old_line"), position.get("old_line"))
    new_lines = nums(start.get("new_line"), end.get("new_line"), position.get("new_line"))
    path, line0, line1 = comment_target(note)
    return {
        "base_sha": position.get("base_sha") or "",
        "start_sha": position.get("start_sha") or "",
        "head_sha": position.get("head_sha") or "",
        "old_path": position.get("old_path") or "",
        "new_path": position.get("new_path") or "",
        "file": path or "",
        "old_line_start": min(old_lines) if old_lines else None,
        "old_line_end": max(old_lines) if old_lines else None,
        "new_line_start": min(new_lines) if new_lines else (line0 if not old_lines else None),
        "new_line_end": max(new_lines) if new_lines else (line1 if not old_lines else None),
        "snippet": "",
    }


def slice_lines(content: str, start: int | None, end: int | None, pad: int = 2) -> str:
    if not content or start is None:
        return ""
    rows = content.splitlines()
    last = end or start
    begin = max(1, start - pad)
    stop = min(len(rows), last + pad)
    if begin > len(rows):
        return ""
    return "\n".join(rows[begin - 1 : stop])


def snippet_for_anchor(anchor: dict, files: list[dict] | None) -> str:
    path = anchor.get("file") or ""
    if not path:
        return ""
    start = anchor.get("new_line_start") or anchor.get("old_line_start")
    end = anchor.get("new_line_end") or anchor.get("old_line_end") or start
    head = anchor.get("head_sha") or ""
    candidates = []
    for item in files or []:
        if item.get("path") != path:
            continue
        if head and item.get("sha") != head:
            continue
        candidates.append(item)
    if not candidates:
        for item in files or []:
            if item.get("path") == path and item.get("side") == "before":
                candidates.append(item)
                break
    if not candidates:
        return ""
    return slice_lines(candidates[0].get("content") or "", start, end)


def map_line(line: int, hunks: list[Hunk]) -> tuple[int, bool, Hunk | None]:
    """Map one old-file line through hunks. Hunks use the pre-commit line numbers."""
    adjusted = line
    for hunk in sorted(hunks, key=lambda item: (item.old_start, item.old_count)):
        if hunk.old_count == 0:
            if line > hunk.old_start:
                adjusted += hunk.new_count
            continue
        old_last = hunk.old_start + hunk.old_count - 1
        if line < hunk.old_start:
            break
        if line <= old_last:
            if hunk.new_count <= 0:
                return max(hunk.new_start, 1), True, hunk
            offset = line - hunk.old_start
            return hunk.new_start + min(offset, hunk.new_count - 1), True, hunk
        adjusted += hunk.new_count - hunk.old_count
    return adjusted, False, None


def map_range(start: int, end: int, hunks: list[Hunk], pad: int = 0) -> tuple[int, int, bool, Hunk | None]:
    """Map an inclusive range. ``pad`` widens only the touch test, not the stored lines."""
    touch_start = max(1, start - pad)
    touch_end = end + pad
    touched = False
    touching: Hunk | None = None
    for hunk in hunks:
        if hunk.old_count == 0:
            if touch_start <= hunk.old_start <= touch_end:
                touched = True
                touching = touching or hunk
            continue
        old0, old1 = _span(hunk.old_start, hunk.old_count)
        if not (touch_end < old0 or touch_start > old1):
            touched = True
            touching = touching or hunk
    new_start, _touch_s, hunk_s = map_line(start, hunks)
    new_end, _touch_e, hunk_e = map_line(end, hunks)
    if new_end < new_start:
        new_start, new_end = new_end, new_start
    return new_start, new_end, touched, touching or hunk_s or hunk_e


def _commit_time(commit: dict):
    return parse_time(str(commit.get("created_at") or commit.get("committed_date") or ""))


def _message(commit: dict) -> str:
    return "\n".join(part for part in (commit.get("message"), commit.get("title")) if part)


def _sha(commit: dict) -> str:
    return str(commit.get("id") or commit.get("sha") or "")


def _path_hunks(diffs: list[dict], path: str) -> tuple[list[Hunk], str]:
    """Hunks on ``path`` and the path name after this commit (follows renames)."""
    hunks: list[Hunk] = []
    renamed = path
    for diff in diffs or []:
        old_path = diff.get("old_path") or ""
        new_path = diff.get("new_path") or ""
        if path not in (old_path, new_path):
            continue
        hunks.extend(parse_hunks(diff.get("diff") or ""))
        if old_path == path and new_path and new_path != path:
            renamed = new_path
    return hunks, renamed


def _join_hunks(hunks: list[Hunk]) -> tuple[str, str, str]:
    before = "\n\n".join(hunk.before for hunk in hunks if hunk.before).strip()
    after = "\n\n".join(hunk.after for hunk in hunks if hunk.after).strip()
    diff = "\n".join(hunk.text for hunk in hunks if hunk.text).strip()
    return before, after, diff


def iter_threads(discussions: list[dict], reviewer: str | None = None) -> list[dict]:
    """One traceable comment per discussion root. Later notes in the thread are replies.

    A reviewer filter keeps threads whose root author matches. It does not drop
    replies from other people. System notes are never roots.
    """
    wanted = reviewer.lower() if reviewer else None
    threads = []
    for discussion in discussions or []:
        humans = []
        for note in discussion.get("notes") or []:
            if not isinstance(note, dict) or note.get("system"):
                continue
            if not str(note.get("body") or "").strip():
                continue
            humans.append(note)
        if not humans:
            continue
        root = humans[0]
        username = str((root.get("author") or {}).get("username") or "")
        if wanted and username.lower() != wanted:
            continue
        threads.append(
            {
                "discussion_id": str(discussion.get("id") or ""),
                "root": root,
                "replies": humans[1:],
            }
        )
    return threads


def _reply_view(note: dict) -> dict:
    author = note.get("author") or {}
    return {
        "id": note.get("id"),
        "author_username": author.get("username") or "",
        "author_name": author.get("name") or "",
        "body": str(note.get("body") or "").strip(),
        "created_at": note.get("created_at") or "",
    }


def trace_thread(
    thread: dict,
    commits: list[dict],
    diffs_by_sha: dict[str, list],
    mr: dict | None = None,
    *,
    fix_pattern: str = r"mrr\s*fixes",
    line_pad: int = 2,
    files: list[dict] | None = None,
) -> dict:
    note = thread["root"]
    author = note.get("author") or {}
    username = author.get("username") or ""
    anchor = extract_anchor(note)
    anchor["snippet"] = snippet_for_anchor(anchor, files)
    resolved_by = (note.get("resolved_by") or {}).get("username") or ""
    try:
        note_time = _commit_time({"created_at": note.get("created_at")})
    except ValueError:
        note_time = None

    later = []
    if note_time is not None:
        for commit in commits or []:
            try:
                if _commit_time(commit) > note_time:
                    later.append(commit)
            except ValueError:
                continue
        later.sort(key=_commit_time)

    path = anchor.get("file") or ""
    line_start = anchor.get("new_line_start") or anchor.get("old_line_start")
    line_end = anchor.get("new_line_end") or anchor.get("old_line_end") or line_start
    changed_by: list[dict] = []
    file_hits: list[dict] = []

    if path and line_start is not None:
        current_path = path
        current_start, current_end = line_start, line_end
        for commit in later:
            sha = _sha(commit)
            hunks, renamed = _path_hunks(diffs_by_sha.get(sha) or [], current_path)
            if not hunks and renamed == current_path:
                continue
            new_start, new_end, touched, hunk = map_range(current_start, current_end, hunks, pad=line_pad)
            entry = {
                "sha": sha,
                "message": _message(commit),
                "created_at": commit.get("created_at") or commit.get("committed_date") or "",
                "hunk_index": hunks.index(hunk) if hunk in hunks else 0,
                "mapped_line_start": new_start,
                "mapped_line_end": new_end,
            }
            if touched and hunk is not None:
                before, after, diff = _join_hunks([hunk])
                entry.update({"before": before, "after": after or diff, "diff": diff})
                changed_by.append(entry)
            elif hunks:
                before, after, diff = _join_hunks(hunks)
                entry.update({"before": before, "after": after or diff, "diff": diff})
                file_hits.append(entry)
            current_start, current_end = new_start, new_end
            current_path = renamed
    elif path:
        for commit in later:
            sha = _sha(commit)
            hunks, _renamed = _path_hunks(diffs_by_sha.get(sha) or [], path)
            if not hunks:
                continue
            before, after, diff = _join_hunks(hunks)
            file_hits.append(
                {
                    "sha": sha,
                    "message": _message(commit),
                    "created_at": commit.get("created_at") or commit.get("committed_date") or "",
                    "hunk_index": 0,
                    "before": before,
                    "after": after or diff,
                    "diff": diff,
                }
            )

    primary = None
    status = "unclear"
    confidence = 0.0
    link = "none"
    resolved = bool(note.get("resolved"))
    if changed_by:
        primary = changed_by[0]
        status = "linked"
        link = "file_line"
        confidence = min(0.95, FILE_LINE_CONFIDENCE + (0.1 if resolved else 0))
    elif file_hits:
        primary = file_hits[0]
        status = "addressed_elsewhere"
        link = "file"
        confidence = FILE_CONFIDENCE
    else:
        status = "not_addressed" if note_time is not None else "unclear"
        confidence = 0.0

    if not anchor["snippet"] and primary and primary.get("before"):
        anchor["snippet"] = primary["before"]

    commit = {}
    if primary and primary.get("sha"):
        commit = next((item for item in commits or [] if _sha(item) == primary["sha"]), {})
    commit_author = commit.get("author") if isinstance(commit.get("author"), dict) else {
        "name": commit.get("author_name"),
        "email": commit.get("author_email"),
    }

    return {
        "note_id": str(note.get("id")),
        "discussion_id": thread.get("discussion_id") or "",
        "author_username": username,
        "reviewer": username,
        "author_name": author.get("name") or "",
        "author_role": author_role(mr, username),
        "body": str(note.get("body") or "").strip(),
        "created_at": note.get("created_at") or "",
        "resolved": bool(note.get("resolved")),
        "resolved_by": (note.get("resolved_by") or {}).get("username") or "",
        "replies": [_reply_view(reply) for reply in thread.get("replies") or []],
        "anchor": anchor,
        "status": status,
        "confidence": confidence,
        "link": link,
        "manual": False,
        "commit_sha": (primary or {}).get("sha") or "",
        "commit_message": (primary or {}).get("message") or "",
        "commit_created_at": (primary or {}).get("created_at") or "",
        "hunk_index": (primary or {}).get("hunk_index") or 0,
        "before": (primary or {}).get("before") or "",
        "after": (primary or {}).get("after") or "",
        "diff": (primary or {}).get("diff") or "",
        "changed_by": changed_by,
        "commit_author": commit_author or {},
    }


def _all_hunks(diffs: list[dict]) -> tuple[str, list[Hunk]]:
    hunks: list[Hunk] = []
    for diff in diffs or []:
        hunks.extend(parse_hunks(diff.get("diff") or ""))
    return "", hunks


def apply_override(trace: dict, override: dict | None, commits: list[dict], diffs_by_sha: dict[str, list]) -> dict:
    """Manual link wins over the automatic one. The override row is kept across reprocessing."""
    if not override:
        return trace
    status = (override.get("status") or "").lower()
    updated = dict(trace)
    updated["manual"] = True
    updated["link"] = "manual"
    if status in {"not_addressed", "unclear", "addressed_elsewhere"} and not override.get("commit_sha"):
        updated["status"] = status
        updated["confidence"] = 0 if status != "addressed_elsewhere" else trace.get("confidence") or 0
        if status == "not_addressed":
            updated["commit_sha"] = ""
            updated["before"] = ""
            updated["after"] = ""
            updated["diff"] = ""
        return updated
    sha = override.get("commit_sha") or ""
    path = (trace.get("anchor") or {}).get("file") or ""
    hunks, _renamed = _path_hunks(diffs_by_sha.get(sha) or [], path) if path else ([], path)
    if not hunks:
        _, hunks = _all_hunks(diffs_by_sha.get(sha) or [])
    index = int(override.get("hunk_index") or 0)
    if hunks:
        index = min(max(index, 0), len(hunks) - 1)
        chosen = [hunks[index]]
    else:
        chosen = []
    before, after, diff = _join_hunks(chosen)
    commit = next((item for item in commits or [] if _sha(item) == sha), {})
    updated.update(
        {
            "status": "linked" if status in {"", "linked"} else status,
            "confidence": MANUAL_CONFIDENCE,
            "commit_sha": sha,
            "commit_message": _message(commit),
            "commit_created_at": commit.get("created_at") or "",
            "hunk_index": index,
            "before": before,
            "after": after or diff,
            "diff": diff,
            "commit_author": commit.get("author")
            if isinstance(commit.get("author"), dict)
            else {"name": commit.get("author_name"), "email": commit.get("author_email")},
        }
    )
    return updated


def build_traces(
    discussions: list,
    commits: list,
    diffs_by_sha: dict[str, list],
    mr: dict | None = None,
    *,
    reviewer: str | None = None,
    fix_pattern: str = r"mrr\s*fixes",
    line_pad: int = 2,
    files: list[dict] | None = None,
    overrides: dict[str, dict] | None = None,
) -> list[dict]:
    traces = []
    for thread in iter_threads(discussions, reviewer):
        traced = trace_thread(
            thread,
            commits,
            diffs_by_sha,
            mr,
            fix_pattern=fix_pattern,
            line_pad=line_pad,
            files=files,
        )
        override = (overrides or {}).get(traced["note_id"])
        traces.append(apply_override(traced, override, commits, diffs_by_sha))
    return traces


def summarize_traces(traces: list[dict]) -> dict:
    def empty() -> dict:
        return {
            "comments": 0,
            "linked": 0,
            "addressed_elsewhere": 0,
            "not_addressed": 0,
            "unclear": 0,
        }

    total = empty()
    by_reviewer: dict[str, dict] = {}
    by_file: dict[str, dict] = {}
    for trace in traces:
        status = trace.get("status") or "unclear"
        if status not in total:
            status = "unclear"
        total["comments"] += 1
        total[status] += 1
        reviewer = trace.get("author_username") or "(unknown)"
        file_name = (trace.get("anchor") or {}).get("file") or "(no file)"
        for bucket, key in ((by_reviewer, reviewer), (by_file, file_name)):
            row = bucket.setdefault(key, empty())
            row["comments"] += 1
            row[status] += 1
    total["with_fix_share"] = (total["linked"] / total["comments"]) if total["comments"] else 0.0
    return {"total": total, "by_reviewer": by_reviewer, "by_file": by_file}
