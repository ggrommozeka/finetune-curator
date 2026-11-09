"""Per-MR commit roles. Titles never decide who fixed a review comment.

Roles:
- ``author`` — MR author's commit before any review comment
- ``author_fix`` — MR author's commit after a review comment
- ``team_lead_mrr`` — team lead commit whose title matches the MRR regex
- ``team_lead_other`` — any other team lead commit
- ``other``
"""

from __future__ import annotations

from scripts.ingest.gitlab.link import parse_time
from scripts.ingest.gitlab.signals import SignalConfig, is_mrr_title, is_team_lead, mr_author_identities, commit_identities
from scripts.ingest.gitlab.trace import iter_threads


def _moment(value: str):
    try:
        return parse_time(value)
    except ValueError:
        return None


def _commit_time(commit: dict):
    return _moment(str(commit.get("created_at") or commit.get("committed_date") or ""))


def _message(commit: dict) -> str:
    return "\n".join(part for part in (commit.get("message"), commit.get("title")) if part)


def first_review_time(discussions: list | None) -> object | None:
    earliest = None
    for thread in iter_threads(discussions or [], None):
        moment = _moment(str(thread["root"].get("created_at") or ""))
        if moment is None:
            continue
        if earliest is None or moment < earliest:
            earliest = moment
    return earliest


def classify_commits(
    commits: list[dict],
    mr: dict | None,
    discussions: list | None,
    config: SignalConfig,
) -> list[dict]:
    author_ids = mr_author_identities(mr)
    review_at = first_review_time(discussions)
    ordered = sorted(commits or [], key=lambda commit: (_commit_time(commit) is None, _commit_time(commit) or ""))
    rows = []
    for commit in ordered:
        sha = str(commit.get("id") or commit.get("sha") or "")
        message = _message(commit)
        mrr = is_mrr_title(message, config.mrr_pattern)
        moment = _commit_time(commit)
        identities = commit_identities(commit)
        by_author = bool(author_ids and identities & author_ids)
        if is_team_lead(commit, config):
            role = "team_lead_mrr" if mrr else "team_lead_other"
        elif by_author and review_at is not None and moment is not None and moment > review_at:
            role = "author_fix"
        elif by_author:
            role = "author"
        else:
            role = "other"
        rows.append(
            {
                "sha": sha,
                "role": role,
                "mrr": mrr,
                "message": message,
                "created_at": commit.get("created_at") or commit.get("committed_date") or "",
                "author_name": commit.get("author_name") or (commit.get("author") or {}).get("name") or "",
                "author_email": commit.get("author_email") or (commit.get("author") or {}).get("email") or "",
            }
        )
    return rows
