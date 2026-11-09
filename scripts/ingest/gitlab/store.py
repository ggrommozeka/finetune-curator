"""Local GitLab store. Raw API payloads and processing results live in different tables.

The database and raw JSON files under ``data/gitlab/`` are gitignored. Re-running
processing never reads GitLab and never rewrites the raw tables.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from scripts.ingest.gitlab.link import DEFAULT_FIX_PATTERN, is_fix_message
from scripts.ingest.gitlab.trace import author_role, extract_anchor, snippet_for_anchor, summarize_traces

SCHEMA = """
CREATE TABLE IF NOT EXISTS merge_requests (
  project_id INTEGER NOT NULL,
  iid INTEGER NOT NULL,
  project_path TEXT,
  title TEXT,
  description TEXT,
  author_username TEXT,
  author_name TEXT,
  web_url TEXT,
  merged_at TEXT,
  updated_at TEXT,
  created_at TEXT,
  raw_json TEXT NOT NULL,
  discussions_json TEXT NOT NULL,
  commits_json TEXT NOT NULL,
  synced_at TEXT NOT NULL,
  PRIMARY KEY (project_id, iid)
);
CREATE TABLE IF NOT EXISTS notes (
  project_id INTEGER NOT NULL,
  mr_iid INTEGER NOT NULL,
  note_id TEXT NOT NULL,
  discussion_id TEXT,
  author_username TEXT,
  author_name TEXT,
  author_role TEXT,
  body TEXT,
  created_at TEXT,
  resolved INTEGER,
  resolved_by TEXT,
  file_path TEXT,
  base_sha TEXT,
  start_sha TEXT,
  head_sha TEXT,
  old_line_start INTEGER,
  old_line_end INTEGER,
  new_line_start INTEGER,
  new_line_end INTEGER,
  snippet TEXT,
  raw_json TEXT NOT NULL,
  PRIMARY KEY (project_id, note_id)
);
CREATE TABLE IF NOT EXISTS commits (
  project_id INTEGER NOT NULL,
  mr_iid INTEGER NOT NULL,
  sha TEXT NOT NULL,
  message TEXT,
  created_at TEXT,
  author_name TEXT,
  author_email TEXT,
  committer_name TEXT,
  committer_email TEXT,
  PRIMARY KEY (project_id, mr_iid, sha)
);
CREATE TABLE IF NOT EXISTS diffs (
  project_id INTEGER NOT NULL,
  sha TEXT NOT NULL,
  old_path TEXT,
  new_path TEXT,
  diff TEXT,
  raw_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
  project_id INTEGER NOT NULL,
  sha TEXT NOT NULL,
  path TEXT NOT NULL,
  side TEXT NOT NULL,
  content TEXT,
  PRIMARY KEY (project_id, sha, path, side)
);
CREATE TABLE IF NOT EXISTS examples (
  id TEXT PRIMARY KEY,
  project_id INTEGER NOT NULL,
  mr_iid INTEGER NOT NULL,
  note_id TEXT,
  link TEXT,
  confidence REAL,
  record_json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS example_decisions (
  example_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  note TEXT NOT NULL DEFAULT '',
  user_text TEXT,
  assistant_text TEXT,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS traces (
  project_id INTEGER NOT NULL,
  mr_iid INTEGER NOT NULL,
  note_id TEXT NOT NULL,
  author_username TEXT,
  author_role TEXT,
  file_path TEXT,
  status TEXT NOT NULL,
  confidence REAL,
  trace_json TEXT NOT NULL,
  PRIMARY KEY (project_id, note_id)
);
CREATE TABLE IF NOT EXISTS commit_roles (
  project_id INTEGER NOT NULL,
  mr_iid INTEGER NOT NULL,
  sha TEXT NOT NULL,
  role TEXT NOT NULL,
  mrr INTEGER NOT NULL,
  message TEXT,
  author_name TEXT,
  author_email TEXT,
  created_at TEXT,
  PRIMARY KEY (project_id, mr_iid, sha)
);
CREATE TABLE IF NOT EXISTS link_overrides (
  project_id INTEGER NOT NULL,
  note_id TEXT NOT NULL,
  status TEXT NOT NULL,
  commit_sha TEXT,
  hunk_index INTEGER,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (project_id, note_id)
);
"""


def _ensure_columns(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(commits)")}
    for name in ("committer_name", "committer_email"):
        if name not in columns:
            conn.execute(f"ALTER TABLE commits ADD COLUMN {name} TEXT")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_mr_ref(text: str) -> tuple[str, int]:
    """Parse ``project!iid``. Project may be a numeric id or a path."""
    if "!" not in text:
        raise ValueError(f"MR ref {text!r} must look like 123!7 or group/app!7")
    project, iid_text = text.rsplit("!", 1)
    project = project.strip()
    if not project or not iid_text.strip().isdigit():
        raise ValueError(f"MR ref {text!r} must look like 123!7 or group/app!7")
    return project, int(iid_text)


class GitlabStore:
    def __init__(self, root: Path | str = "data/gitlab"):
        self.root = Path(root)
        self.db_path = self.root / "store.db"
        self.raw_root = self.root / "raw"
        self.root.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        _ensure_columns(self._conn)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def has_mr(self, project_id: int, iid: int) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM merge_requests WHERE project_id = ? AND iid = ?",
            (int(project_id), int(iid)),
        ).fetchone()
        return row is not None

    def save_bundle(
        self,
        mr: dict,
        discussions: list,
        commits: list,
        diffs_by_sha: dict[str, list],
        files: list[dict],
        *,
        project_path: str | None = None,
        issues: list | None = None,
    ) -> None:
        project_id = int(mr["project_id"])
        iid = int(mr["iid"])
        author = mr.get("author") or {}
        now = utc_now()
        self._write_raw(project_id, iid, "mr.json", mr)
        self._write_raw(project_id, iid, "discussions.json", discussions)
        self._write_raw(project_id, iid, "commits.json", commits)
        self._write_raw(project_id, iid, "issues.json", issues or [])
        for sha, diffs in diffs_by_sha.items():
            self._write_raw(project_id, iid, f"diffs/{sha}.json", diffs)

        with self._conn:
            self._conn.execute("DELETE FROM notes WHERE project_id = ? AND mr_iid = ?", (project_id, iid))
            self._conn.execute("DELETE FROM commits WHERE project_id = ? AND mr_iid = ?", (project_id, iid))
            self._conn.execute(
                """
                INSERT INTO merge_requests (
                  project_id, iid, project_path, title, description, author_username, author_name,
                  web_url, merged_at, updated_at, created_at, raw_json, discussions_json,
                  commits_json, synced_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(project_id, iid) DO UPDATE SET
                  project_path = excluded.project_path,
                  title = excluded.title,
                  description = excluded.description,
                  author_username = excluded.author_username,
                  author_name = excluded.author_name,
                  web_url = excluded.web_url,
                  merged_at = excluded.merged_at,
                  updated_at = excluded.updated_at,
                  created_at = excluded.created_at,
                  raw_json = excluded.raw_json,
                  discussions_json = excluded.discussions_json,
                  commits_json = excluded.commits_json,
                  synced_at = excluded.synced_at
                """,
                (
                    project_id,
                    iid,
                    project_path or "",
                    mr.get("title") or "",
                    mr.get("description") or "",
                    author.get("username") or "",
                    author.get("name") or "",
                    mr.get("web_url") or "",
                    mr.get("merged_at") or "",
                    mr.get("updated_at") or "",
                    mr.get("created_at") or "",
                    json.dumps(mr, ensure_ascii=False),
                    json.dumps(discussions, ensure_ascii=False),
                    json.dumps(commits, ensure_ascii=False),
                    now,
                ),
            )
            for discussion in discussions or []:
                for note in discussion.get("notes") or []:
                    if not isinstance(note, dict) or note.get("system"):
                        continue
                    note_author = note.get("author") or {}
                    username = note_author.get("username") or ""
                    anchor = extract_anchor(note)
                    anchor["snippet"] = snippet_for_anchor(anchor, files)
                    self._conn.execute(
                        """
                        INSERT OR REPLACE INTO notes (
                          project_id, mr_iid, note_id, discussion_id, author_username, author_name,
                          author_role, body, created_at, resolved, resolved_by, file_path,
                          base_sha, start_sha, head_sha, old_line_start, old_line_end,
                          new_line_start, new_line_end, snippet, raw_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            project_id,
                            iid,
                            str(note.get("id")),
                            str(discussion.get("id") or ""),
                            username,
                            note_author.get("name") or "",
                            author_role(mr, username),
                            note.get("body") or "",
                            note.get("created_at") or "",
                            1 if note.get("resolved") else 0,
                            ((note.get("resolved_by") or {}).get("username") or ""),
                            anchor.get("file") or "",
                            anchor.get("base_sha") or "",
                            anchor.get("start_sha") or "",
                            anchor.get("head_sha") or "",
                            anchor.get("old_line_start"),
                            anchor.get("old_line_end"),
                            anchor.get("new_line_start"),
                            anchor.get("new_line_end"),
                            anchor.get("snippet") or "",
                            json.dumps(note, ensure_ascii=False),
                        ),
                    )
            for commit in commits or []:
                sha = str(commit.get("id") or "")
                if not sha:
                    continue
                message = commit.get("message") or commit.get("title") or ""
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO commits (
                      project_id, mr_iid, sha, message, created_at, author_name, author_email,
                      committer_name, committer_email
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project_id,
                        iid,
                        sha,
                        message,
                        commit.get("created_at") or commit.get("committed_date") or "",
                        commit.get("author_name") or (commit.get("author") or {}).get("name") or "",
                        commit.get("author_email") or (commit.get("author") or {}).get("email") or "",
                        commit.get("committer_name") or (commit.get("committer") or {}).get("name") or "",
                        commit.get("committer_email") or (commit.get("committer") or {}).get("email") or "",
                    ),
                )
            for sha in diffs_by_sha:
                self._conn.execute("DELETE FROM diffs WHERE project_id = ? AND sha = ?", (project_id, sha))
            for sha, diffs in diffs_by_sha.items():
                for diff in diffs or []:
                    self._conn.execute(
                        """
                        INSERT INTO diffs (project_id, sha, old_path, new_path, diff, raw_json)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            project_id,
                            sha,
                            diff.get("old_path") or "",
                            diff.get("new_path") or "",
                            diff.get("diff") or "",
                            json.dumps(diff, ensure_ascii=False),
                        ),
                    )
            for item in files:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO files (project_id, sha, path, side, content)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (project_id, item["sha"], item["path"], item["side"], item.get("content") or ""),
                )

    def _write_raw(self, project_id: int, iid: int, name: str, payload) -> None:
        path = self.raw_root / str(project_id) / str(iid) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def replace_roles(self, project_id: int, iid: int, roles: list[dict]) -> None:
        self._conn.execute(
            "DELETE FROM commit_roles WHERE project_id = ? AND mr_iid = ?",
            (project_id, iid),
        )
        for role in roles:
            self._conn.execute(
                """
                INSERT INTO commit_roles (
                  project_id, mr_iid, sha, role, mrr, message, author_name, author_email, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    iid,
                    role.get("sha") or "",
                    role.get("role") or "other",
                    1 if role.get("mrr") else 0,
                    role.get("message") or "",
                    role.get("author_name") or "",
                    role.get("author_email") or "",
                    role.get("created_at") or "",
                ),
            )
        self._conn.commit()

    def roles_for(self, project_id: int, iid: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM commit_roles WHERE project_id = ? AND mr_iid = ? ORDER BY created_at",
            (project_id, iid),
        ).fetchall()
        return [
            {
                "sha": row["sha"],
                "role": row["role"],
                "mrr": bool(row["mrr"]),
                "message": row["message"] or "",
                "author_name": row["author_name"] or "",
                "author_email": row["author_email"] or "",
                "created_at": row["created_at"] or "",
            }
            for row in rows
        ]

    def list_mrs(
        self,
        *,
        project: str = "",
        author: str = "",
        reviewer: str = "",
        since: str = "",
        until: str = "",
        fixes_only: bool = False,
        min_comments: int = 0,
        fix_pattern: str = DEFAULT_FIX_PATTERN,
    ) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM merge_requests ORDER BY merged_at DESC, project_id, iid"
        ).fetchall()
        found = []
        for row in rows:
            item = self._summary(row, fix_pattern)
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
            if fixes_only and item["fix_count"] <= 0:
                continue
            if item["comment_count"] < min_comments:
                continue
            found.append(item)
        return found

    def _summary(self, row: sqlite3.Row, fix_pattern: str) -> dict:
        project_id = row["project_id"]
        iid = row["iid"]
        notes = self._conn.execute(
            "SELECT author_username FROM notes WHERE project_id = ? AND mr_iid = ?",
            (project_id, iid),
        ).fetchall()
        commits = self._conn.execute(
            "SELECT message FROM commits WHERE project_id = ? AND mr_iid = ?",
            (project_id, iid),
        ).fetchall()
        reviewers = sorted({note["author_username"] for note in notes if note["author_username"]})
        messages = [commit["message"] or "" for commit in commits]
        fix_count = sum(1 for message in messages if is_fix_message(message, fix_pattern))
        from scripts.ingest.gitlab.classify import classify_commits
        from scripts.ingest.gitlab.signals import load_signal_config

        raw_mr = json.loads(row["raw_json"])
        discussions = json.loads(row["discussions_json"])
        commit_rows = json.loads(row["commits_json"])
        classified = classify_commits(commit_rows, raw_mr, discussions, load_signal_config(mrr_pattern=fix_pattern))
        role_counts: dict[str, int] = {}
        for item in classified:
            role_counts[item["role"]] = role_counts.get(item["role"], 0) + 1
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
            "fix_count": fix_count,
            "role_counts": role_counts,
            "lead_count": role_counts.get("team_lead_mrr", 0) + role_counts.get("team_lead_other", 0),
            "reviewers": reviewers,
            "ref": f"{project_id}!{iid}",
        }

    def load_bundle(self, project_id: int, iid: int) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM merge_requests WHERE project_id = ? AND iid = ?",
            (int(project_id), int(iid)),
        ).fetchone()
        if row is None:
            return None
        mr = json.loads(row["raw_json"])
        discussions = json.loads(row["discussions_json"])
        commits = json.loads(row["commits_json"])
        issues_path = self.raw_root / str(int(project_id)) / str(int(iid)) / "issues.json"
        issues = []
        if issues_path.is_file():
            try:
                issues = json.loads(issues_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                issues = []
        diffs_by_sha: dict[str, list] = {}
        for sha_row in self._conn.execute(
            "SELECT DISTINCT sha FROM diffs WHERE project_id = ?",
            (int(project_id),),
        ):
            sha = sha_row["sha"]
            diffs_by_sha.setdefault(sha, [])
        for diff_row in self._conn.execute(
            "SELECT sha, raw_json FROM diffs WHERE project_id = ?",
            (int(project_id),),
        ):
            diffs_by_sha.setdefault(diff_row["sha"], []).append(json.loads(diff_row["raw_json"]))
        # Diffs are stored per commit sha, not per MR. Keep only shas that belong to this MR.
        mr_shas = {str(commit.get("id") or "") for commit in commits}
        diffs_by_sha = {sha: diffs for sha, diffs in diffs_by_sha.items() if sha in mr_shas}
        files = [
            dict(item)
            for item in self._conn.execute(
                "SELECT sha, path, side, content FROM files WHERE project_id = ?",
                (int(project_id),),
            )
            if item["sha"] in mr_shas or item["side"] == "base"
        ]
        return {
            "mr": mr,
            "discussions": discussions,
            "commits": commits,
            "diffs_by_sha": diffs_by_sha,
            "files": files,
            "issues": issues if isinstance(issues, list) else [],
            "project_path": row["project_path"] or "",
        }

    def resolve_ref(self, ref: str) -> tuple[int, int]:
        project, iid = parse_mr_ref(ref)
        if project.isdigit():
            project_id = int(project)
            if not self.has_mr(project_id, iid):
                raise KeyError(ref)
            return project_id, iid
        rows = self._conn.execute(
            """
            SELECT project_id, iid FROM merge_requests
            WHERE iid = ? AND (
              project_path = ? OR web_url LIKE ? OR CAST(project_id AS TEXT) = ?
            )
            """,
            (iid, project, f"%/{project}/%", project),
        ).fetchall()
        if len(rows) != 1:
            raise KeyError(ref)
        return int(rows[0]["project_id"]), int(rows[0]["iid"])

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
