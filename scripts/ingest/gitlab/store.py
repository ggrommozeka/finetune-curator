"""Local GitLab store. Raw API payloads and processing results live in different tables.

The database and raw JSON files under ``data/gitlab/`` are gitignored. Re-running
processing never reads GitLab and never rewrites the raw tables.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from scripts.ingest.gitlab.derived import DerivedStore
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
CREATE TABLE IF NOT EXISTS issues (
  project_id INTEGER NOT NULL,
  mr_iid INTEGER NOT NULL,
  issue_iid INTEGER NOT NULL,
  project_path TEXT NOT NULL DEFAULT '',
  title TEXT,
  description TEXT,
  raw_json TEXT NOT NULL,
  PRIMARY KEY (project_id, mr_iid, project_path, issue_iid)
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


class GitlabStore(DerivedStore):
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
            self._replace_issues(project_id, iid, issues or [])

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

    def _replace_issues(self, project_id: int, iid: int, issues: list) -> None:
        self._conn.execute(
            "DELETE FROM issues WHERE project_id = ? AND mr_iid = ?",
            (project_id, iid),
        )
        for issue in issues or []:
            if not isinstance(issue, dict):
                continue
            try:
                issue_iid = int(issue.get("iid"))
            except (TypeError, ValueError):
                continue
            self._conn.execute(
                """
                INSERT OR REPLACE INTO issues (
                  project_id, mr_iid, issue_iid, project_path, title, description, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    iid,
                    issue_iid,
                    str(issue.get("project_path") or ""),
                    issue.get("title") or "",
                    issue.get("description") or "",
                    json.dumps(issue, ensure_ascii=False),
                ),
            )

    def issues_for(self, project_id: int, iid: int) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT issue_iid, project_path, title, description FROM issues
            WHERE project_id = ? AND mr_iid = ? ORDER BY project_path, issue_iid
            """,
            (int(project_id), int(iid)),
        ).fetchall()
        if rows:
            return [
                {
                    "iid": row["issue_iid"],
                    "project_path": row["project_path"] or "",
                    "title": row["title"] or "",
                    "description": row["description"] or "",
                }
                for row in rows
            ]
        issues_path = self.raw_root / str(int(project_id)) / str(int(iid)) / "issues.json"
        if not issues_path.is_file():
            return []
        try:
            payload = json.loads(issues_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        return payload if isinstance(payload, list) else []

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
        issues = self.issues_for(project_id, iid)
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
        from scripts.ingest.gitlab.sync import _path_from_web

        rows = self._conn.execute(
            "SELECT project_id, iid, project_path, web_url FROM merge_requests WHERE iid = ?",
            (iid,),
        ).fetchall()
        matches = []
        for row in rows:
            stored = row["project_path"] or _path_from_web({"web_url": row["web_url"] or ""})
            if stored == project or str(row["project_id"]) == project:
                matches.append(row)
        if len(matches) != 1:
            raise KeyError(ref)
        return int(matches[0]["project_id"]), int(matches[0]["iid"])

