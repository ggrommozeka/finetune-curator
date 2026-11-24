"""Style pairs and review samples from one GitLab merge request."""

from __future__ import annotations

import difflib

from scripts.datasets.context import review_context
from scripts.datasets.format import clip_diff_context
from scripts.datasets.sample import NO_ISSUES, REVIEW_SYSTEM, STYLE_SYSTEM, Sample, _with_system
from scripts.ingest.gitlab.classify import classify_commits
from scripts.ingest.gitlab.link import parse_hunks
from scripts.ingest.gitlab.polish import iter_hunks
from scripts.ingest.gitlab.signals import SignalConfig, load_signal_config
from scripts.ingest.gitlab.export import linked_review
from scripts.ingest.gitlab.trace import extract_anchor

def _style_messages(before: str, after: str, *, file_name: str, why: str) -> list[dict]:
    user = (
        "Rewrite this code so it follows the project's conventions.\n\n"
        f"Why: {why.strip() or 'match the surrounding project style'}\n"
        f"File: {file_name or '(file)'}\n\n"
        f"Code before:\n```\n{before}\n```"
    )
    return _with_system(STYLE_SYSTEM, [{"role": "user", "content": user}, {"role": "assistant", "content": after}])


def style_from_pair(
    *,
    sample_id: str,
    group: str,
    machine: str,
    kind: str,
    before: str,
    after: str,
    file_name: str,
    why: str,
) -> Sample | None:
    if not before.strip() or not after.strip() or before.strip() == after.strip():
        return None
    messages = _style_messages(before, after, file_name=file_name, why=why)
    prompt = messages[:-1]
    return Sample(
        id=sample_id,
        task="style",
        source="gitlab",
        machine=machine,
        group=group,
        kind=kind,
        messages=messages,
        dpo={"prompt": prompt, "chosen": [messages[-1]], "rejected": [{"role": "assistant", "content": before}]},
        gold=after,
    )


def _pack_review(
    *,
    diff: str,
    file_name: str,
    line_start: int | None,
    line_end: int | None,
    mr: dict,
    issues: list,
    base_files: dict[str, str],
    conventions: str,
    context_tokens: int,
    review_context_lines: int | None,
) -> str:
    """One packed review prompt, shared by comments, replies, and negative hunks."""
    clipped = clip_diff_context(diff, line_start, line_end, review_context_lines) if review_context_lines else diff
    others = [(path, text) for path, text in base_files.items() if path and path != file_name and text]
    return review_context(
        diff=clipped,
        title=str(mr.get("title") or ""),
        description=str(mr.get("description") or ""),
        issues=issues,
        base_file=base_files.get(file_name) or "",
        line_start=line_start,
        conventions=conventions,
        max_tokens=context_tokens,
        other_files=others,
    )


def _line_span(anchor: dict) -> tuple[int | None, int | None]:
    start = anchor.get("new_line_start") or anchor.get("old_line_start")
    end = anchor.get("new_line_end") or anchor.get("old_line_end") or start
    return start, end


def _overlaps(start: int, end: int, anchor: dict, pad: int = 2) -> bool:
    line_start, line_end = _line_span(anchor)
    if line_start is None:
        return False
    line_end = line_end or line_start
    return not (end < line_start - pad or start > line_end + pad)


_NEGATIVE_SKIP = {"author_fix", "team_lead_mrr", "team_lead_other"}


def _excluded_touch(role: dict) -> bool:
    return bool(role.get("mrr")) or role.get("role") in _NEGATIVE_SKIP


def _hunk_body(hunk) -> list[str]:
    return hunk.text.splitlines()[1:]


def _base_from_hunks(hunks) -> list[str]:
    """Old file lines implied by the first commit, when no base snapshot was stored."""
    lines: list[str] = []
    for hunk in hunks:
        if hunk.old_count <= 0:
            continue
        before: list[str] = []
        for raw in _hunk_body(hunk):
            if not raw or raw.startswith("+") or raw.startswith("\\"):
                continue
            before.append(raw[1:] if raw[0] in "- " else raw)
        start = max(hunk.old_start - 1, 0)
        while len(lines) < start:
            lines.append("")
        lines[start : start + len(before)] = before
        while len(lines) < start + hunk.old_count:
            lines.append("")
    return lines


def _apply_hunk(lines: list[str], blame: list, tainted: set[int], hunk, sha: str, excluded: bool):
    """Replay one hunk whose old line numbers still match ``lines``.

    ``tainted`` holds final-file lines a lead or fix commit touched. Callers apply
    every hunk of one commit from the bottom of the file upward, so an earlier
    hunk does not move the lines this hunk and its taint update still name.
    """
    if hunk.old_count == 0:
        prefix = min(max(hunk.old_start, 0), len(lines))
        old_count = 0
    else:
        prefix = min(max(hunk.old_start - 1, 0), len(lines))
        old_count = hunk.old_count
    produced: list[str] = []
    produced_blame: list = []
    cursor = prefix
    for raw in _hunk_body(hunk):
        if not raw or raw.startswith("\\"):
            continue
        mark = raw[0]
        if mark == "+":
            produced.append(raw[1:])
            produced_blame.append(sha)
        elif mark == "-":
            cursor += 1
        else:
            text = raw[1:] if mark == " " else raw
            produced.append(text)
            produced_blame.append(blame[cursor] if cursor < len(blame) else None)
            cursor += 1
    end_old = prefix + old_count
    delta = len(produced) - old_count
    remapped: set[int] = set()
    for line in tainted:
        index = line - 1
        if index < prefix:
            remapped.add(line)
        elif index >= prefix + old_count:
            remapped.add(line + delta)
    if excluded:
        if produced:
            remapped.update(range(prefix + 1, prefix + 1 + len(produced)))
        else:
            remapped.add(max(prefix, 1))
            remapped.add(prefix + 1)
    merged_lines = lines[:prefix] + produced + lines[end_old:]
    merged_blame = blame[:prefix] + produced_blame + blame[end_old:]
    return merged_lines, merged_blame, remapped


def _bottom_up(hunks: list) -> list:
    """Later hunks in a commit keep the pre-commit line numbers. Apply the lowest last."""
    return [
        hunk
        for _index, hunk in sorted(
            enumerate(hunks),
            key=lambda item: (item[1].old_start, item[1].old_count, item[0]),
            reverse=True,
        )
    ]


def _apply_commit(lines: list[str], blame: list, tainted: set[int], hunks: list, sha: str, excluded: bool):
    for hunk in _bottom_up(hunks):
        lines, blame, tainted = _apply_hunk(lines, blame, tainted, hunk, sha, excluded)
    return lines, blame, tainted


def _added_blame(hunk, blame: list) -> list:
    number = hunk.new_start
    found = []
    for raw in _hunk_body(hunk):
        if not raw or raw.startswith("\\"):
            continue
        if raw.startswith("+"):
            found.append(blame[number - 1] if 0 < number <= len(blame) else None)
            number += 1
        elif raw.startswith("-"):
            continue
        else:
            number += 1
    return found


def _net_negative_hunks(commits: list[dict], diffs: dict, roles: dict, base_files: dict) -> list[tuple[str, object]]:
    """Hunks of the MR's base-to-head diff that still belong to untouched author code."""
    ordered = sorted(
        commits or [],
        key=lambda commit: (
            str(commit.get("created_at") or commit.get("committed_date") or ""),
            str(commit.get("id") or commit.get("sha") or ""),
        ),
    )
    by_path: dict[str, list[tuple[str, bool, list]]] = {}
    for commit in ordered:
        sha = str(commit.get("id") or commit.get("sha") or "")
        role = roles.get(sha) or {}
        excluded = _excluded_touch(role)
        for diff in diffs.get(sha) or []:
            new_path = diff.get("new_path") or ""
            old_path = diff.get("old_path") or ""
            path = new_path if new_path and new_path != "/dev/null" else old_path
            if not path or path == "/dev/null":
                continue
            by_path.setdefault(path, []).append((sha, excluded, diff))
    kept: list[tuple[str, object]] = []
    for path, touches in by_path.items():
        first_hunks = []
        for _sha, _excluded, diff in touches:
            first_hunks.extend(parse_hunks(diff.get("diff") or ""))
            if first_hunks:
                break
        base_text = base_files.get(path) or ""
        lines = base_text.splitlines() if base_text else _base_from_hunks(first_hunks)
        base_lines = list(lines)
        blame: list = [None] * len(lines)
        tainted: set[int] = set()
        index = 0
        while index < len(touches):
            sha, excluded, _diff = touches[index]
            hunks = []
            while index < len(touches) and touches[index][0] == sha:
                hunks.extend(parse_hunks(touches[index][2].get("diff") or ""))
                index += 1
            lines, blame, tainted = _apply_commit(lines, blame, tainted, hunks, sha, excluded)
        rendered = "\n".join(
            difflib.unified_diff(base_lines, lines, fromfile=path, tofile=path, n=3, lineterm="")
        )
        for hunk in parse_hunks(rendered):
            added = _added_blame(hunk, blame)
            if added and any(item is not None and _excluded_touch(roles.get(str(item)) or {}) for item in added):
                continue
            start = hunk.new_start
            end = hunk.new_start + max(hunk.new_count, 1) - 1
            if any(start <= line <= end for line in tainted):
                continue
            if not hunk.text.strip():
                continue
            kept.append((path, hunk))
    return kept


def review_comment_sample(
    *,
    sample_id: str,
    group: str,
    machine: str,
    file_name: str,
    diff: str,
    comment: str,
    reviewer: str,
    line_start,
    line_end,
    suggested: str = "",
    context_lines: int | None = None,
    context: str = "",
) -> Sample | None:
    if not comment.strip():
        return None
    if context_lines is not None:
        diff = clip_diff_context(diff, line_start, line_end, context_lines)
        if suggested.strip():
            suggested = "\n".join(suggested.splitlines()[: max(context_lines * 2, 1)])
    lines = ""
    if line_start:
        lines = f"{line_start}-{line_end or line_start}"
    if context.strip():
        user = "Review this change. Name the file and lines when you comment.\n\n" + context.strip()
    else:
        user = (
            "Review this diff. Name the file and lines when you comment.\n\n"
            f"File: {file_name or '(file)'}\n"
            f"```diff\n{diff}\n```"
        )
    body = f"{file_name}:{lines}: {comment.strip()}" if lines else comment.strip()
    if suggested.strip():
        body += f"\n\nSuggested fix:\n```\n{suggested.strip()}\n```"
    messages = _with_system(
        REVIEW_SYSTEM,
        [{"role": "user", "content": user}, {"role": "assistant", "content": body}],
    )
    return Sample(
        id=sample_id,
        task="review",
        source="gitlab",
        machine=machine,
        group=group,
        kind="review",
        messages=messages,
        gold=comment.strip(),
        extra={"reviewer": reviewer, "file": file_name, "lines": lines},
        dpo={
            "prompt": messages[:-1],
            "chosen": [messages[-1]],
            "rejected": [{"role": "assistant", "content": NO_ISSUES}],
        },
    )


def review_negative_sample(
    *,
    sample_id: str,
    group: str,
    machine: str,
    file_name: str,
    diff: str,
    context_lines: int | None = None,
    context: str = "",
) -> Sample | None:
    if not diff.strip() and not context.strip():
        return None
    if context_lines is not None and not context.strip():
        diff = clip_diff_context(diff, None, None, context_lines)
    if context.strip():
        user = "Review this change. Name the file and lines when you comment.\n\n" + context.strip()
    else:
        user = f"Review this diff. Name the file and lines when you comment.\n\nFile: {file_name}\n```diff\n{diff}\n```"
    messages = _with_system(
        REVIEW_SYSTEM,
        [{"role": "user", "content": user}, {"role": "assistant", "content": NO_ISSUES}],
    )
    return Sample(
        id=sample_id,
        task="review",
        source="gitlab",
        machine=machine,
        group=group,
        kind="review-negative",
        messages=messages,
        gold=NO_ISSUES,
        extra={"file": file_name},
        dpo={
            "prompt": messages[:-1],
            "chosen": [messages[-1]],
            "rejected": [{"role": "assistant", "content": "This change introduces a bug."}],
        },
    )

def _anchors(traces: list[dict]) -> list[dict]:
    found = []
    for trace in traces:
        anchor = dict(trace.get("anchor") or {})
        if anchor.get("file"):
            found.append(anchor)
    return found


def from_gitlab_bundle(
    bundle: dict,
    config: SignalConfig,
    *,
    machine: str = "gitlab",
    review_context_lines: int | None = None,
    conventions: str = "",
    context_tokens: int = 8_192,
) -> list[Sample]:
    mr = bundle.get("mr") or {}
    project_id = mr.get("project_id")
    iid = mr.get("iid")
    group = f"mr:{project_id}!{iid}"
    commits = bundle.get("commits") or []
    diffs = bundle.get("diffs_by_sha") or {}
    discussions = bundle.get("discussions") or []
    if bundle.get("traces"):
        traces = bundle["traces"]
    else:
        traces, _roles = linked_review(
            mr,
            discussions,
            commits,
            diffs,
            config,
            files=bundle.get("files") or [],
        )
    samples: list[Sample] = []
    base_files = {
        item.get("path") or "": item.get("content") or ""
        for item in (bundle.get("files") or [])
        if item.get("side") == "base"
    }
    issues = bundle.get("issues") or []
    roles = {row["sha"]: row for row in classify_commits(commits, mr, discussions, config)}
    for commit in commits:
        sha = str(commit.get("id") or commit.get("sha") or "")
        role = roles.get(sha) or {}
        if role.get("role") not in {"team_lead_mrr", "team_lead_other"}:
            continue
        why = "team-lead MRR polish" if role.get("mrr") else "team-lead polish"
        for hunk in iter_hunks(diffs.get(sha) or []):
            sample = style_from_pair(
                sample_id=f"style-{project_id}-{iid}-{sha[:8]}-{hunk['index']}",
                group=group,
                machine=machine,
                kind="polish" if role.get("mrr") else "polish-other",
                before=hunk["before"],
                after=hunk["after"],
                file_name=hunk["file"],
                why=f"{why}: {role.get('message') or ''}".strip(),
            )
            if sample:
                samples.append(sample)
    commented: list[dict] = []
    for trace in traces:
        anchor = trace.get("anchor") or {}
        file_name = anchor.get("file") or ""
        if file_name:
            commented.append(anchor)
        if trace.get("before") and trace.get("after"):
            sample = style_from_pair(
                sample_id=f"style-fix-{project_id}-{iid}-{trace.get('note_id')}",
                group=group,
                machine=machine,
                kind="comment-fix",
                before=trace.get("before") or "",
                after=trace.get("after") or "",
                file_name=file_name,
                why=trace.get("body") or "review comment",
            )
            if sample:
                samples.append(sample)
        diff = trace.get("diff") or trace.get("before") or ""
        line_start = anchor.get("new_line_start") or anchor.get("old_line_start")
        line_end = anchor.get("new_line_end") or anchor.get("old_line_end")
        packed = _pack_review(
            diff=diff,
            file_name=file_name,
            line_start=line_start,
            line_end=line_end,
            mr=mr,
            issues=issues,
            base_files=base_files,
            conventions=conventions,
            context_tokens=context_tokens,
            review_context_lines=review_context_lines,
        )
        if trace.get("author_role") == "author":
            continue
        review = review_comment_sample(
            sample_id=f"review-{project_id}-{iid}-{trace.get('note_id')}",
            group=group,
            machine=machine,
            file_name=file_name,
            diff=diff,
            comment=trace.get("body") or "",
            reviewer=trace.get("reviewer") or trace.get("author_username") or "",
            line_start=line_start,
            line_end=line_end,
            suggested=(trace.get("after") or "") if trace.get("status") == "linked" else "",
            context_lines=None,
            context=packed,
        )
        if review:
            samples.append(review)
        author_username = str((mr.get("author") or {}).get("username") or "")
        for reply in trace.get("replies") or []:
            if author_username and (reply.get("author_username") or "") == author_username:
                continue
            reply_review = review_comment_sample(
                sample_id=f"review-{project_id}-{iid}-{reply.get('id')}",
                group=group,
                machine=machine,
                file_name=file_name,
                diff=diff,
                comment=reply.get("body") or "",
                reviewer=reply.get("author_username") or "",
                line_start=(anchor.get("new_line_start") or anchor.get("old_line_start")),
                line_end=(anchor.get("new_line_end") or anchor.get("old_line_end")),
                context=packed,
            )
            if reply_review:
                samples.append(reply_review)
    if not traces:
        for discussion in discussions:
            for note in discussion.get("notes") or []:
                if note.get("system") or not str(note.get("body") or "").strip():
                    continue
                anchor = extract_anchor(note)
                if anchor.get("file"):
                    commented.append(anchor)
    for index, (path, hunk) in enumerate(_net_negative_hunks(commits, diffs, roles, base_files)):
        start = hunk.new_start
        end = hunk.new_start + max(hunk.new_count, 1) - 1
        if any(item.get("file") == path and _overlaps(start, end, item) for item in commented):
            continue
        negative = review_negative_sample(
            sample_id=f"review-ok-{project_id}-{iid}-{index}",
            group=group,
            machine=machine,
            file_name=path,
            diff=hunk.text,
            context=_pack_review(
                diff=hunk.text,
                file_name=path,
                line_start=start,
                line_end=end,
                mr=mr,
                issues=issues,
                base_files=base_files,
                conventions=conventions,
                context_tokens=context_tokens,
                review_context_lines=review_context_lines,
            ),
        )
        if negative:
            samples.append(negative)
    return samples


def from_gitlab_store(
    root,
    config: SignalConfig | None = None,
    *,
    review_context_lines: int | None = None,
    conventions: str = "",
    context_tokens: int = 8_192,
) -> list[Sample]:
    from pathlib import Path

    root = Path(root)
    if not (root / "store.db").exists():
        return []
    from scripts.ingest.gitlab.store import GitlabStore

    config = config or load_signal_config()
    store = GitlabStore(root)
    try:
        samples = []
        rows = store.list_mrs()
        for row in rows:
            bundle = store.load_bundle(row["project_id"], row["iid"])
            if bundle is None:
                continue
            bundle["traces"] = store.traces_for(row["project_id"], row["iid"])
            samples.extend(
                from_gitlab_bundle(
                    bundle,
                    config,
                    review_context_lines=review_context_lines,
                    conventions=conventions,
                    context_tokens=context_tokens,
                )
            )
        return samples
    finally:
        store.close()

