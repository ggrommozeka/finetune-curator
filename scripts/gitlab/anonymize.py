"""Strip author names, emails, and optionally URLs from exported records."""

from __future__ import annotations

import re

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_URL = re.compile(r"https?://[^\s)>\]\"']+")


class Anonymizer:
    def __init__(self, *, strip_urls: bool = False):
        self.strip_urls = strip_urls
        self._ids: dict[str, str] = {}
        self._labels: list[tuple[str, str]] = []
        self._seen_labels: set[str] = set()

    def register_author(self, author: dict | None) -> str:
        author = author or {}
        username = str(author.get("username") or "")
        name = str(author.get("name") or "")
        email = str(author.get("email") or "")
        key = (username or email or name).strip().lower()
        if not key:
            return "dev"
        if key not in self._ids:
            self._ids[key] = f"dev-{len(self._ids) + 1}"
        token = self._ids[key]
        for label in (username, name, email):
            cleaned = label.strip()
            if len(cleaned) < 3:
                continue
            marker = cleaned.lower()
            if marker in self._seen_labels:
                continue
            self._seen_labels.add(marker)
            self._labels.append((cleaned, token))
        return token

    def text(self, value: str) -> str:
        if not value:
            return value
        out = _EMAIL.sub("<email>", value)
        for label, token in sorted(self._labels, key=lambda item: len(item[0]), reverse=True):
            out = re.sub(rf"\b{re.escape(label)}\b", token, out, flags=re.IGNORECASE)
        if self.strip_urls:
            out = _URL.sub("<url>", out)
        return out

    def author_view(self, author: dict | None) -> dict:
        author = author or {}
        if not any(str(author.get(key) or "").strip() for key in ("username", "name", "email")):
            return {}
        token = self.register_author(author)
        return {"username": token, "name": token}


def anonymize_record(record: dict, anonymizer: Anonymizer) -> dict:
    """Return a copy with names, emails, and optional URLs removed."""
    gitlab = dict(record.get("gitlab") or {})
    for author_key in ("author", "commit_author"):
        author = gitlab.get(author_key)
        if isinstance(author, dict) and author:
            gitlab[author_key] = anonymizer.author_view(author)
    if gitlab.get("reviewer"):
        gitlab["reviewer"] = anonymizer.author_view({"username": gitlab["reviewer"]}).get("username") or ""
    if gitlab.get("resolved_by"):
        gitlab["resolved_by"] = anonymizer.author_view({"username": gitlab["resolved_by"]}).get("username") or ""
    replies = []
    for reply in gitlab.get("replies") or []:
        reply = dict(reply)
        view = anonymizer.author_view(
            {"username": reply.get("author_username"), "name": reply.get("author_name")}
        )
        reply["author_username"] = view.get("username") or ""
        reply["author_name"] = view.get("name") or ""
        reply["body"] = anonymizer.text(str(reply.get("body") or ""))
        replies.append(reply)
    if gitlab.get("replies") is not None:
        gitlab["replies"] = replies
    anchor = gitlab.get("anchor")
    if isinstance(anchor, dict):
        anchor = dict(anchor)
        anchor["snippet"] = anonymizer.text(str(anchor.get("snippet") or ""))
        gitlab["anchor"] = anchor
    if anonymizer.strip_urls:
        gitlab.pop("web_url", None)
    for key in ("comment", "before", "after", "diff", "commit_message", "file"):
        if isinstance(gitlab.get(key), str):
            gitlab[key] = anonymizer.text(gitlab[key])
    messages = []
    for message in record.get("messages") or []:
        item = dict(message)
        if isinstance(item.get("content"), str):
            item["content"] = anonymizer.text(item["content"])
        messages.append(item)
    cleaned = dict(record)
    cleaned["messages"] = messages
    cleaned["gitlab"] = gitlab
    cleaned.pop("_authors", None)
    return cleaned
