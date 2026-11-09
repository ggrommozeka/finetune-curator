"""One item shape for OpenCode sessions and GitLab review comments.

Training examples and proposal evidence both carry ``source`` so the viewer
can open the right page. GitLab is optional: this module still works before
that store exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from scripts.ingest.opencode_sessions import first_user_preview, load_jsonl, normalize_messages, session_id


@dataclass
class SourceItem:
    source: str
    item_id: str
    title: str
    text: str
    locator: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "item_id": self.item_id,
            "title": self.title,
            "text": self.text,
            "locator": dict(self.locator),
        }


class Catalog:
    def __init__(self, opencode_path: Path | str, gitlab_root: Path | str | None = None):
        self.opencode_path = Path(opencode_path)
        self.gitlab_root = Path(gitlab_root) if gitlab_root else None

    def opencode_records(self) -> list[dict]:
        rows = load_jsonl(self.opencode_path)
        for row in rows:
            row.setdefault("source", "opencode")
        return rows

    def gitlab_mrs(self) -> list[dict]:
        store = self._gitlab_store()
        if store is None:
            return []
        try:
            rows = store.list_mrs()
        finally:
            store.close()
        for row in rows:
            row["source"] = "gitlab"
        return rows

    def resolve(self, ref: str) -> SourceItem | None:
        ref = str(ref)
        for row in self.opencode_records():
            sid = session_id(row)
            if ref not in {sid, str(row.get("id") or ""), str(row.get("session_id") or "")}:
                continue
            messages = normalize_messages(row)
            return SourceItem(
                source="opencode",
                item_id=sid,
                title=str(row.get("title") or sid),
                text=first_user_preview(messages, 180),
                locator={"session_id": sid},
            )
        gitlab_item = self._resolve_gitlab(ref)
        if gitlab_item is not None:
            return gitlab_item
        return None

    def _gitlab_store(self):
        if self.gitlab_root is None or not (self.gitlab_root / "store.db").exists():
            return None
        try:
            from scripts.ingest.gitlab.store import GitlabStore
        except ImportError:
            return None
        return GitlabStore(self.gitlab_root)

    def _resolve_gitlab(self, ref: str) -> SourceItem | None:
        store = self._gitlab_store()
        if store is None:
            return None
        try:
            for trace in store.traces_for():
                note_id = str(trace.get("note_id") or "")
                keys = {note_id, f"gitlab-{note_id}"}
                if ref not in keys:
                    continue
                anchor = trace.get("anchor") or {}
                project_id = None
                iid = None
                # Traces do not always repeat project ids; look them up from notes.
                row = store._conn.execute(
                    "SELECT project_id, mr_iid FROM notes WHERE note_id = ?",
                    (note_id,),
                ).fetchone()
                if row is not None:
                    project_id = int(row["project_id"])
                    iid = int(row["mr_iid"])
                return SourceItem(
                    source="gitlab",
                    item_id=f"gitlab-{note_id}",
                    title=str(anchor.get("file") or "merge request"),
                    text=str(trace.get("body") or ""),
                    locator={"project_id": project_id, "mr_iid": iid, "note_id": note_id},
                )
        finally:
            store.close()
        return None
