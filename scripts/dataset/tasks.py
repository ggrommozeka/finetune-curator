"""Agentic, style, and review samples from one OpenCode export and one GitLab store."""

from __future__ import annotations

from dataclasses import dataclass, field

from scripts.dataset.format import clip_diff_context, split_messages, to_tool_message
from scripts.gitlab.classify import classify_commits
from scripts.gitlab.link import parse_hunks
from scripts.gitlab.polish import iter_hunks
from scripts.gitlab.signals import SignalConfig, load_signal_config
from scripts.gitlab.trace import build_traces, extract_anchor
from scripts.rules.sessions import load_jsonl, load_review_decisions, normalize_messages, session_id

AGENTIC_SYSTEM = (
    "You are a coding agent in OpenCode. Complete the user's task. "
    "Use tools when you need to read or change the project, and follow tool results."
)
STYLE_SYSTEM = (
    "Rewrite the code so it follows the project's conventions. "
    "Return only the updated code."
)
REVIEW_SYSTEM = (
    "You review a diff. Point at the file and lines that have a problem and say what to change. "
    "If the change is fine, reply exactly: No issues."
)

NO_ISSUES = "No issues."


@dataclass
class Sample:
    id: str
    task: str
    source: str
    machine: str
    group: str
    messages: list[dict]
    kind: str
    dpo: dict | None = None
    gold: str = ""
    extra: dict = field(default_factory=dict)

    def as_sft(self) -> dict:
        return {
            "id": self.id,
            "task": self.task,
            "source": self.source,
            "machine": self.machine,
            "group": self.group,
            "kind": self.kind,
            "messages": self.messages,
        }

    def as_dpo(self) -> dict | None:
        if not self.dpo:
            return None
        return {
            "id": self.id,
            "task": self.task,
            "source": self.source,
            "machine": self.machine,
            "group": self.group,
            "kind": self.kind,
            "prompt": self.dpo["prompt"],
            "chosen": self.dpo["chosen"],
            "rejected": self.dpo["rejected"],
        }


def _with_system(system: str, messages: list[dict]) -> list[dict]:
    if messages and messages[0].get("role") == "system":
        return messages
    return [{"role": "system", "content": system}, *messages]


def _assistant_after(messages: list[dict], start: int) -> int | None:
    for index in range(start, len(messages)):
        if messages[index].get("role") == "assistant":
            return index
    return None


def _corrections(messages: list[dict]) -> list[tuple[int, int, int]]:
    """Return (user_index, rejected_assistant, chosen_assistant) for follow-up corrections."""
    first_user = next((index for index, message in enumerate(messages) if message.get("role") == "user"), None)
    found = []
    for index, message in enumerate(messages):
        if message.get("role") != "user" or index == first_user:
            continue
        rejected = None
        for earlier in range(index - 1, -1, -1):
            if messages[earlier].get("role") == "assistant":
                rejected = earlier
                break
        chosen = _assistant_after(messages, index + 1)
        if rejected is None or chosen is None:
            continue
        found.append((index, rejected, chosen))
    return found


def _improved(messages: list[dict], pairs: list[tuple[int, int, int]]) -> list[dict]:
    drop = set()
    for user_index, rejected, _chosen in pairs:
        drop.add(user_index)
        drop.add(rejected)
    return [message for index, message in enumerate(messages) if index not in drop]


def agentic_samples(record: dict, *, machine: str, success: bool) -> list[Sample]:
    raw = [to_tool_message(message) for message in normalize_messages(record)]
    raw = [message for message in raw if message.get("content") or message.get("tool_calls")]
    if not raw:
        return []
    sid = session_id(record)
    pairs = _corrections(raw)
    samples = []
    if success and not pairs:
        samples.append(
            Sample(
                id=f"agentic-{sid}",
                task="agentic",
                source="opencode",
                machine=machine,
                group=f"session:{sid}",
                kind="trajectory",
                messages=_with_system(AGENTIC_SYSTEM, raw),
            )
        )
    if pairs:
        improved = _with_system(AGENTIC_SYSTEM, _improved(raw, pairs))
        primary = pairs[0]
        prompt = _with_system(AGENTIC_SYSTEM, raw[: primary[1]])
        dpo = {"prompt": prompt, "chosen": [raw[primary[2]]], "rejected": [raw[primary[1]]]}
        if any(message.get("role") == "assistant" for message in improved):
            samples.append(
                Sample(
                    id=f"agentic-{sid}-improved",
                    task="agentic",
                    source="opencode",
                    machine=machine,
                    group=f"session:{sid}",
                    kind="correction",
                    messages=improved,
                    dpo=dpo,
                    gold=raw[primary[2]].get("content") or "",
                    extra={"correction": raw[primary[0]].get("content") or ""},
                )
            )
        for nth, (user_index, rejected, chosen) in enumerate(pairs[1:], 2):
            step_prompt = _with_system(AGENTIC_SYSTEM, raw[:rejected])
            samples.append(
                Sample(
                    id=f"agentic-{sid}-dpo-{nth}",
                    task="agentic",
                    source="opencode",
                    machine=machine,
                    group=f"session:{sid}",
                    kind="correction-dpo",
                    messages=step_prompt + [raw[chosen]],
                    dpo={"prompt": step_prompt, "chosen": [raw[chosen]], "rejected": [raw[rejected]]},
                    gold=raw[chosen].get("content") or "",
                    extra={"correction": raw[user_index].get("content") or ""},
                )
            )
    return samples


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
) -> Sample | None:
    if not diff.strip():
        return None
    if context_lines is not None:
        diff = clip_diff_context(diff, None, None, context_lines)
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


def _success(record: dict, labels: dict[str, str], min_score: float) -> bool:
    sid = session_id(record)
    label = labels.get(sid, "")
    if label == "keep":
        return True
    if label == "discard":
        return False
    score = record.get("score")
    try:
        return score is not None and score != "" and float(score) >= min_score
    except (TypeError, ValueError):
        return False


def _machine_for(record: dict, machines: dict[str, str]) -> str:
    sid = session_id(record)
    if sid in machines:
        return machines[sid]
    return str(record.get("machine") or record.get("source") or "opencode")


def load_machine_labels(patterns_root) -> dict[str, str]:
    from pathlib import Path

    root = Path(patterns_root) if patterns_root else None
    if root is None or not (root / "store.db").exists():
        return {}
    from scripts.patterns.store import PatternStore

    store = PatternStore(root)
    try:
        labels = {}
        for row in store.sessions():
            sources = row.get("sources") or []
            machine = ",".join(sources) if sources else "opencode"
            for key in (row.get("id"), row.get("session_id"), row.get("global_id")):
                if key:
                    labels[str(key)] = machine
        return labels
    finally:
        store.close()


def from_opencode(path, decisions_path, *, machines: dict[str, str], min_score: float) -> list[Sample]:
    from pathlib import Path

    labels = load_review_decisions(Path(decisions_path))["labels"] if decisions_path else {}
    samples = []
    for record in load_jsonl(Path(path)):
        machine = _machine_for(record, machines)
        samples.extend(agentic_samples(record, machine=machine, success=_success(record, labels, min_score)))
    return samples


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
) -> list[Sample]:
    mr = bundle.get("mr") or {}
    project_id = mr.get("project_id")
    iid = mr.get("iid")
    group = f"mr:{project_id}!{iid}"
    commits = bundle.get("commits") or []
    diffs = bundle.get("diffs_by_sha") or {}
    discussions = bundle.get("discussions") or []
    traces = bundle.get("traces") or build_traces(
        discussions, commits, diffs, mr, files=bundle.get("files") or []
    )
    samples: list[Sample] = []
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
        review = review_comment_sample(
            sample_id=f"review-{project_id}-{iid}-{trace.get('note_id')}",
            group=group,
            machine=machine,
            file_name=file_name,
            diff=diff,
            comment=trace.get("body") or "",
            reviewer=trace.get("reviewer") or trace.get("author_username") or "",
            line_start=(anchor.get("new_line_start") or anchor.get("old_line_start")),
            line_end=(anchor.get("new_line_end") or anchor.get("old_line_end")),
            suggested=trace.get("after") or "",
            context_lines=review_context_lines,
        )
        if review:
            samples.append(review)
        for reply in trace.get("replies") or []:
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
                context_lines=review_context_lines,
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
    seen_diff = set()
    for sha, files in diffs.items():
        for diff in files or []:
            path = diff.get("new_path") or diff.get("old_path") or ""
            for index, hunk in enumerate(parse_hunks(diff.get("diff") or "")):
                if not hunk.text.strip():
                    continue
                start = hunk.new_start
                end = hunk.new_start + max(hunk.new_count, 1) - 1
                if any(item.get("file") == path and _overlaps(start, end, item) for item in commented):
                    continue
                key = (path, hunk.text)
                if key in seen_diff:
                    continue
                seen_diff.add(key)
                negative = review_negative_sample(
                    sample_id=f"review-ok-{project_id}-{iid}-{str(sha)[:8]}-{index}",
                    group=group,
                    machine=machine,
                    file_name=path,
                    diff=hunk.text,
                    context_lines=review_context_lines,
                )
                if negative:
                    samples.append(negative)
    return samples


def from_gitlab_store(root, config: SignalConfig | None = None, *, review_context_lines: int | None = None) -> list[Sample]:
    from pathlib import Path

    root = Path(root)
    if not (root / "store.db").exists():
        return []
    from scripts.gitlab.store import GitlabStore

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
            samples.extend(from_gitlab_bundle(bundle, config, review_context_lines=review_context_lines))
        return samples
    finally:
        store.close()

