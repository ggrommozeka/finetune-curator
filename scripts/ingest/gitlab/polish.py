"""Before/after pairs from team-lead commits. These are not comment fixes.

MRR-titled commits are the primary polish signal. Other team-lead commits are
kept and marked separately. An optional model labels the kind of polish.
"""

from __future__ import annotations

from scripts.ingest.gitlab.anonymize import Anonymizer
from scripts.ingest.gitlab.classify import classify_commits
from scripts.ingest.gitlab.examples import clip
from scripts.ingest.gitlab.link import parse_hunks
from scripts.llm.parsing import strip_thinking
from scripts.ingest.gitlab.signals import SignalConfig

_PROMPT = """Refine this code the way the reviewer did in this commit.

Commit message:
{message}

File: {file}

Code before:
```
{before}
```

Return the updated code."""

_SUMMARY = """Label this expert code edit with one short category.
Choose one of: naming, error handling, simplification, structure, style, tests, other.
Reply with only that label.

Commit message:
{message}

Before:
{before}

After:
{after}
"""


def _sha(commit: dict) -> str:
    return str(commit.get("id") or commit.get("sha") or "")


def iter_hunks(diffs: list[dict]) -> list[dict]:
    found = []
    for diff in diffs or []:
        path = diff.get("new_path") or diff.get("old_path") or ""
        for index, hunk in enumerate(parse_hunks(diff.get("diff") or "")):
            if not (hunk.before.strip() or hunk.after.strip()):
                continue
            if hunk.before.strip() == hunk.after.strip():
                continue
            found.append(
                {
                    "file": path,
                    "index": index,
                    "before": hunk.before,
                    "after": hunk.after,
                    "diff": hunk.text,
                }
            )
    return found


def summarize_polish(complete_fn, message: str, before: str, after: str) -> str:
    """Ask the local model what kind of polish this hunk is. Empty on failure."""
    if complete_fn is None:
        return ""
    try:
        raw = complete_fn(
            [
                {"role": "system", "content": "You label code edits. Reply with one short label."},
                {"role": "user", "content": _SUMMARY.format(message=message, before=before[:2000], after=after[:2000])},
            ]
        )
    except Exception:
        return ""
    text = strip_thinking(str(raw or "")).strip().splitlines()
    if not text:
        return ""
    return text[0].strip().strip(".").lower()[:80]


def build_polish_records(
    mr: dict,
    commits: list[dict],
    diffs_by_sha: dict[str, list],
    discussions: list | None,
    config: SignalConfig,
    *,
    max_chars: int = 8000,
    complete_fn=None,
) -> list[dict]:
    roles = {row["sha"]: row for row in classify_commits(commits, mr, discussions, config)}
    project_id = mr.get("project_id")
    iid = mr.get("iid")
    records = []
    for commit in commits or []:
        sha = _sha(commit)
        role = roles.get(sha) or {}
        if role.get("role") not in {"team_lead_mrr", "team_lead_other"}:
            continue
        message = role.get("message") or ""
        author = {
            "name": role.get("author_name") or "",
            "email": role.get("author_email") or "",
        }
        for hunk in iter_hunks(diffs_by_sha.get(sha) or []):
            before = clip(hunk["before"], max_chars)
            after = clip(hunk["after"], max_chars)
            if not after.strip() or before.strip() == after.strip():
                continue
            label = summarize_polish(complete_fn, message, before, after) if complete_fn else ""
            user = _PROMPT.format(
                message=clip(message, 1000),
                file=hunk["file"] or "(file)",
                before=before,
            )
            record_id = f"gitlab-{project_id}-{iid}-polish-{sha[:12]}-{hunk['index']}"
            records.append(
                {
                    "id": record_id,
                    "source": "gitlab_polish",
                    "kind": "team_lead_polish",
                    "label": "gitlab",
                    "messages": [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": after},
                    ],
                    "gitlab": {
                        "kind": "team_lead_polish",
                        "project_id": project_id,
                        "mr_iid": iid,
                        "note_id": "",
                        "file": hunk["file"],
                        "before": before,
                        "after": after,
                        "diff": clip(hunk["diff"], max_chars),
                        "commit_sha": sha,
                        "commit_message": message,
                        "commit_created_at": role.get("created_at") or "",
                        "role": role.get("role"),
                        "mrr": bool(role.get("mrr")),
                        "polish": label,
                        "link": "polish",
                        "confidence": 0.95 if role.get("mrr") else 0.7,
                        "trainable": True,
                        "web_url": mr.get("web_url"),
                        "commit_author": author,
                    },
                    "_authors": [author, mr.get("author") or {}],
                }
            )
    return records


def anonymize_polish(records: list[dict], anonymizer: Anonymizer) -> list[dict]:
    from scripts.ingest.gitlab.anonymize import anonymize_record

    for record in records:
        for author in record.get("_authors") or []:
            anonymizer.register_author(author)
    return [anonymize_record(record, anonymizer) for record in records]
