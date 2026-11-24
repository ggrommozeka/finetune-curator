"""Build Unsloth-style SFT messages and DPO pairs from a linked review note."""

from __future__ import annotations

from scripts.gitlab.anonymize import Anonymizer, anonymize_record

_PROMPT = """Apply the review comment to the code.

Merge request: {title}
{description}
File: {file}

Review comment:
{comment}

Code before:
```
{before}
```

Return the updated code."""


def clip(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "\n… [truncated]"


def _description(mr: dict, limit: int) -> str:
    text = str(mr.get("description") or "").strip()
    if not text:
        return ""
    return clip(text, min(limit, 1500)) + "\n"


def build_sft_record(mr: dict, note: dict, link: dict, *, max_chars: int = 8000) -> dict | None:
    before = clip(str(link.get("before") or ""), max_chars)
    after = clip(str(link.get("after") or link.get("diff") or ""), max_chars)
    if not after.strip() or before.strip() == after.strip():
        diff_only = clip(str(link.get("diff") or ""), max_chars)
        if not diff_only.strip():
            return None
        if not after.strip():
            after = diff_only
        if before.strip() == after.strip():
            return None
    file_label = link.get("file") or "(merge request)"
    comment = str(note.get("body") or "").strip()
    user = _PROMPT.format(
        title=str(mr.get("title") or "").strip() or f"MR !{mr.get('iid')}",
        description=_description(mr, max_chars),
        file=file_label,
        comment=clip(comment, max_chars),
        before=before or "(no hunk context)",
    )
    project_id = mr.get("project_id")
    iid = mr.get("iid")
    note_id = note.get("id")
    record_id = f"gitlab-{project_id}-{iid}-note-{note_id}"
    author = note.get("author") if isinstance(note.get("author"), dict) else {}
    commit_author = link.get("commit_author") if isinstance(link.get("commit_author"), dict) else {}
    mr_author = mr.get("author") if isinstance(mr.get("author"), dict) else {}
    return {
        "id": record_id,
        "source": "gitlab_review",
        "kind": "comment_fix",
        "label": "gitlab",
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": after},
        ],
        "gitlab": {
            "project_id": project_id,
            "mr_iid": iid,
            "note_id": note_id,
            "file": link.get("file"),
            "line_start": link.get("line_start"),
            "line_end": link.get("line_end"),
            "comment": comment,
            "before": before,
            "after": after,
            "diff": clip(str(link.get("diff") or ""), max_chars),
            "confidence": link.get("confidence"),
            "kind": "comment_fix",
            "link": link.get("link"),
            "commit_sha": link.get("commit_sha"),
            "commit_message": link.get("commit_message"),
            "commit_created_at": link.get("commit_created_at"),
            "note_created_at": note.get("created_at"),
            "resolved": note.get("resolved"),
            "web_url": mr.get("web_url"),
            "reviewer": author.get("username") or "",
            "author": {"username": author.get("username"), "name": author.get("name")},
            "commit_author": {
                "username": commit_author.get("username"),
                "name": commit_author.get("name"),
                "email": commit_author.get("email"),
            }
            if commit_author
            else {},
            "trainable": True,
        },
        "_authors": [author, commit_author, mr_author],
    }


def sft_to_dpo(record: dict) -> dict | None:
    gitlab = record.get("gitlab") or {}
    if gitlab.get("trainable") is False:
        return None
    messages = record.get("messages") or []
    if len(messages) < 2:
        return None
    prompt = messages[0].get("content") or ""
    chosen = messages[1].get("content") or ""
    rejected = gitlab.get("before") or ""
    if not prompt or not chosen or not rejected:
        return None
    if chosen.strip() == rejected.strip():
        return None
    return {
        "id": record.get("id"),
        "source": "gitlab_review",
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def apply_anonymization(records: list[dict], *, strip_urls: bool) -> list[dict]:
    anonymizer = Anonymizer(strip_urls=strip_urls)
    prepared = []
    for record in records:
        for author in record.get("_authors") or []:
            anonymizer.register_author(author)
        prepared.append(record)
    return [anonymize_record(record, anonymizer) for record in prepared]
