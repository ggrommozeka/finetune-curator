"""Derived GitLab rows: examples, decisions, traces, and the MR list."""

from __future__ import annotations

import json

from datetime import datetime, timezone

from scripts.ingest.gitlab.trace import summarize_traces


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class DerivedStore:
    def list_mrs(
        self,
        *,
        project: str = "",
        author: str = "",
        reviewer: str = "",
        since: str = "",
        until: str = "",
        linked_only: bool = False,
        min_comments: int = 0,
    ) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM merge_requests ORDER BY merged_at DESC, project_id, iid"
        ).fetchall()
        found = []
        want_linked = linked_only
        for row in rows:
            item = self._summary(row)
            if project and project.lower() not in " ".join(
                (item["project_path"], str(item["project_id"]), item["web_url"])
            ).lower():
                continue
            if author and author.lower() not in (item["author_username"] + " " + item["author_name"]).lower():
                continue
            if reviewer and not any(reviewer.lower() in name.lower() for name in item["reviewers"]):
                continue
            if since and (item["merged_at"] or "") < since:
                continue
            if until and (item["merged_at"] or "") > until + "T23:59:59":
                continue
            if want_linked and item["linked_count"] <= 0:
                continue
            if item["comment_count"] < min_comments:
                continue
            found.append(item)
        return found

    def _summary(self, row: sqlite3.Row) -> dict:
        project_id = row["project_id"]
        iid = row["iid"]
        notes = self._conn.execute(
            "SELECT author_username FROM notes WHERE project_id = ? AND mr_iid = ?",
            (project_id, iid),
        ).fetchall()
        reviewers = sorted({note["author_username"] for note in notes if note["author_username"]})
        linked_count = self._conn.execute(
            """
            SELECT COUNT(*) AS n FROM traces
            WHERE project_id = ? AND mr_iid = ? AND status = 'linked'
            """,
            (project_id, iid),
        ).fetchone()["n"]
        roles = self.roles_for(project_id, iid)
        role_counts: dict[str, int] = {}
        mrr_count = 0
        for item in roles:
            role_counts[item["role"]] = role_counts.get(item["role"], 0) + 1
            if item["mrr"]:
                mrr_count += 1
        return {
            "project_id": project_id,
            "iid": iid,
            "project_path": row["project_path"] or "",
            "title": row["title"] or "",
            "description": row["description"] or "",
            "author_username": row["author_username"] or "",
            "author_name": row["author_name"] or "",
            "web_url": row["web_url"] or "",
            "merged_at": row["merged_at"] or "",
            "updated_at": row["updated_at"] or "",
            "comment_count": len(notes),
            "linked_count": int(linked_count),
            "mrr_count": mrr_count,
            "role_counts": role_counts,
            "lead_count": role_counts.get("team_lead_mrr", 0) + role_counts.get("team_lead_other", 0),
            "reviewers": reviewers,
            "ref": f"{project_id}!{iid}",
        }


    def replace_examples(self, project_id: int, iid: int, records: list[dict]) -> None:
        now = utc_now()
        with self._conn:
            self._conn.execute(
                "DELETE FROM examples WHERE project_id = ? AND mr_iid = ?",
                (int(project_id), int(iid)),
            )
            for record in records:
                gitlab = record.get("gitlab") or {}
                self._conn.execute(
                    """
                    INSERT INTO examples (
                      id, project_id, mr_iid, note_id, link, confidence, record_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record["id"],
                        int(project_id),
                        int(iid),
                        str(gitlab.get("note_id") or ""),
                        gitlab.get("link") or "",
                        gitlab.get("confidence"),
                        json.dumps(record, ensure_ascii=False),
                        now,
                    ),
                )

    def set_decision(
        self,
        example_id: str,
        status: str,
        *,
        note: str | None = None,
        user_text: str | None = None,
        assistant_text: str | None = None,
    ) -> None:
        status = status.lower()
        if status not in {"keep", "discard", "undecided"}:
            raise ValueError(f"unknown decision {status}")
        current = self.get_decision(example_id) or {}
        self._conn.execute(
            """
            INSERT INTO example_decisions (
              example_id, status, note, user_text, assistant_text, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(example_id) DO UPDATE SET
              status = excluded.status,
              note = excluded.note,
              user_text = excluded.user_text,
              assistant_text = excluded.assistant_text,
              updated_at = excluded.updated_at
            """,
            (
                example_id,
                status,
                current.get("note", "") if note is None else note,
                current.get("user_text") if user_text is None else user_text,
                current.get("assistant_text") if assistant_text is None else assistant_text,
                utc_now(),
            ),
        )
        self._conn.commit()

    def get_decision(self, example_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM example_decisions WHERE example_id = ?",
            (example_id,),
        ).fetchone()
        return dict(row) if row else None

    def examples_for(self, project_id: int | None = None, iid: int | None = None) -> list[dict]:
        if project_id is None:
            rows = self._conn.execute("SELECT * FROM examples ORDER BY id").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM examples WHERE project_id = ? AND mr_iid = ? ORDER BY id",
                (int(project_id), int(iid)),
            ).fetchall()
        found = []
        for row in rows:
            record = json.loads(row["record_json"])
            decision = self.get_decision(row["id"]) or {
                "status": "undecided",
                "note": "",
                "user_text": None,
                "assistant_text": None,
            }
            found.append({"record": record, "decision": decision, "id": row["id"]})
        return found

    def export_records(self, *, status: str = "keep", refs: list[tuple[int, int]] | None = None) -> list[dict]:
        wanted = {status} if status != "all" else {"keep", "discard", "undecided"}
        allowed = set(refs) if refs else None
        exported = []
        for item in self.examples_for():
            record = item["record"]
            gitlab = record.get("gitlab") or {}
            key = (int(gitlab.get("project_id")), int(gitlab.get("mr_iid")))
            if allowed is not None and key not in allowed:
                continue
            decision = item["decision"]
            state = (decision.get("status") or "undecided").lower()
            if state not in wanted:
                continue
            messages = [dict(message) for message in record.get("messages") or []]
            if decision.get("user_text"):
                if messages:
                    messages[0]["content"] = decision["user_text"]
            if decision.get("assistant_text"):
                if len(messages) > 1:
                    messages[1]["content"] = decision["assistant_text"]
                gitlab = dict(gitlab)
                gitlab["after"] = decision["assistant_text"]
            exported.append({**record, "messages": messages, "gitlab": gitlab, "decision": state})
        return exported

    def overrides_for(self, project_id: int, iid: int) -> dict[str, dict]:
        note_ids = [
            row["note_id"]
            for row in self._conn.execute(
                "SELECT note_id FROM notes WHERE project_id = ? AND mr_iid = ?",
                (int(project_id), int(iid)),
            )
        ]
        if not note_ids:
            return {}
        found = {}
        for row in self._conn.execute(
            f"SELECT * FROM link_overrides WHERE project_id = ? AND note_id IN ({','.join('?' * len(note_ids))})",
            (int(project_id), *note_ids),
        ):
            found[row["note_id"]] = dict(row)
        return found

    def set_link_override(
        self,
        project_id: int,
        note_id: str,
        status: str,
        *,
        commit_sha: str | None = None,
        hunk_index: int | None = None,
    ) -> None:
        allowed = {"linked", "addressed_elsewhere", "not_addressed", "unclear"}
        if status not in allowed:
            raise ValueError(f"unknown link status {status}")
        self._conn.execute(
            """
            INSERT INTO link_overrides (project_id, note_id, status, commit_sha, hunk_index, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_id, note_id) DO UPDATE SET
              status = excluded.status,
              commit_sha = excluded.commit_sha,
              hunk_index = excluded.hunk_index,
              updated_at = excluded.updated_at
            """,
            (int(project_id), str(note_id), status, commit_sha or "", hunk_index, utc_now()),
        )
        self._conn.commit()

    def replace_traces(self, project_id: int, iid: int, traces: list[dict]) -> None:
        with self._conn:
            self._conn.execute(
                "DELETE FROM traces WHERE project_id = ? AND mr_iid = ?",
                (int(project_id), int(iid)),
            )
            for trace in traces:
                anchor = trace.get("anchor") or {}
                self._conn.execute(
                    """
                    INSERT INTO traces (
                      project_id, mr_iid, note_id, author_username, author_role, file_path,
                      status, confidence, trace_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(project_id),
                        int(iid),
                        str(trace.get("note_id")),
                        trace.get("author_username") or "",
                        trace.get("author_role") or "",
                        anchor.get("file") or "",
                        trace.get("status") or "unclear",
                        trace.get("confidence"),
                        json.dumps(trace, ensure_ascii=False),
                    ),
                )

    def note_mr(self, note_id: str) -> tuple[int, int] | None:
        row = self._conn.execute(
            "SELECT project_id, mr_iid FROM notes WHERE note_id = ?",
            (str(note_id),),
        ).fetchone()
        if row is None:
            return None
        return int(row["project_id"]), int(row["mr_iid"])

    def traces_for(self, project_id: int | None = None, iid: int | None = None) -> list[dict]:
        if project_id is None:
            rows = self._conn.execute("SELECT trace_json FROM traces").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT trace_json FROM traces WHERE project_id = ? AND mr_iid = ?",
                (int(project_id), int(iid)),
            ).fetchall()
        traces = [json.loads(row["trace_json"]) for row in rows]
        traces.sort(key=lambda item: item.get("created_at") or "")
        return traces

    def link_stats(self, project_id: int | None = None, iid: int | None = None) -> dict:
        return summarize_traces(self.traces_for(project_id, iid))
