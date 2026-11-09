"""Local store for imported sessions, annotations, clusters, and adopted versions."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
  label TEXT NOT NULL,
  origin_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  schema_version TEXT,
  imported_at TEXT NOT NULL,
  PRIMARY KEY (label, origin_id)
);
CREATE TABLE IF NOT EXISTS sessions (
  global_id TEXT PRIMARY KEY,
  content_hash TEXT NOT NULL UNIQUE,
  title TEXT,
  project TEXT,
  created_at TEXT,
  score REAL,
  quality_label TEXT,
  message_count INTEGER,
  raw_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS session_sources (
  global_id TEXT NOT NULL,
  label TEXT NOT NULL,
  origin_id TEXT NOT NULL,
  PRIMARY KEY (label, origin_id)
);
CREATE TABLE IF NOT EXISTS annotations (
  global_id TEXT PRIMARY KEY,
  annotation_json TEXT NOT NULL,
  model TEXT,
  annotated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS clusters (
  id TEXT PRIMARY KEY,
  label TEXT,
  size INTEGER NOT NULL,
  frequency INTEGER NOT NULL,
  task_types INTEGER NOT NULL,
  projects INTEGER NOT NULL,
  machines INTEGER NOT NULL,
  fail_rate REAL NOT NULL,
  noise INTEGER NOT NULL,
  summary_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cluster_members (
  cluster_id TEXT NOT NULL,
  global_id TEXT NOT NULL,
  PRIMARY KEY (cluster_id, global_id)
);
CREATE TABLE IF NOT EXISTS proposals (
  id TEXT PRIMARY KEY,
  cluster_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  conflicts_json TEXT NOT NULL,
  evidence_json TEXT NOT NULL,
  status TEXT NOT NULL,
  edited_json TEXT,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS versions (
  id TEXT PRIMARY KEY,
  adopted_at TEXT NOT NULL,
  retired_at TEXT,
  note TEXT,
  manifest_json TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class PatternStore:
    def __init__(self, root: Path | str = "data/patterns"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "store.db"
        self.versions_root = self.root / "versions"
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def seen_origin(self, label: str, origin_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sources WHERE label = ? AND origin_id = ?",
            (label, origin_id),
        ).fetchone()
        return row is not None

    def find_by_hash(self, content_hash: str) -> str | None:
        row = self._conn.execute(
            "SELECT global_id FROM sessions WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
        return row["global_id"] if row else None

    def add_session(
        self,
        *,
        global_id: str,
        content_hash: str,
        origin_id: str,
        source_label: str,
        kind: str,
        schema_version: str,
        record: dict,
    ) -> str:
        """Insert a session or attach a duplicate source. Returns the global id kept."""
        existing = self.find_by_hash(content_hash)
        kept = existing or global_id
        now = utc_now()
        with self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO sources (label, origin_id, kind, schema_version, imported_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (source_label, origin_id, kind, schema_version, now),
            )
            if existing is None:
                self._conn.execute(
                    """
                    INSERT INTO sessions (
                      global_id, content_hash, title, project, created_at, score,
                      quality_label, message_count, raw_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        global_id,
                        content_hash,
                        record.get("title") or "",
                        record.get("project") or "",
                        record.get("created_at") or "",
                        record.get("score"),
                        record.get("label") or "",
                        record.get("message_count") or 0,
                        json.dumps(record, ensure_ascii=False),
                    ),
                )
            self._conn.execute(
                """
                INSERT OR IGNORE INTO session_sources (global_id, label, origin_id)
                VALUES (?, ?, ?)
                """,
                (kept, source_label, origin_id),
            )
        return kept

    def sessions(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM sessions ORDER BY created_at, global_id").fetchall()
        found = []
        for row in rows:
            item = json.loads(row["raw_json"])
            item["global_id"] = row["global_id"]
            item["sources"] = self.sources_for(row["global_id"])
            found.append(item)
        return found

    def sources_for(self, global_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT label FROM session_sources WHERE global_id = ? ORDER BY label",
            (global_id,),
        ).fetchall()
        return [row["label"] for row in rows]

    def put_annotation(self, global_id: str, annotation: dict, model: str) -> None:
        self._conn.execute(
            """
            INSERT INTO annotations (global_id, annotation_json, model, annotated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(global_id) DO UPDATE SET
              annotation_json = excluded.annotation_json,
              model = excluded.model,
              annotated_at = excluded.annotated_at
            """,
            (global_id, json.dumps(annotation, ensure_ascii=False), model, utc_now()),
        )
        self._conn.commit()

    def annotation(self, global_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT annotation_json FROM annotations WHERE global_id = ?",
            (global_id,),
        ).fetchone()
        return json.loads(row["annotation_json"]) if row else None

    def annotated_rows(self) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT s.global_id, s.raw_json, a.annotation_json
            FROM sessions s JOIN annotations a ON a.global_id = s.global_id
            ORDER BY s.global_id
            """
        ).fetchall()
        found = []
        for row in rows:
            record = json.loads(row["raw_json"])
            record["global_id"] = row["global_id"]
            record["sources"] = self.sources_for(row["global_id"])
            record["annotation"] = json.loads(row["annotation_json"])
            found.append(record)
        return found

    def replace_clusters(self, clusters: list[dict]) -> None:
        now = utc_now()
        with self._conn:
            self._conn.execute("DELETE FROM cluster_members")
            self._conn.execute("DELETE FROM clusters")
            for cluster in clusters:
                self._conn.execute(
                    """
                    INSERT INTO clusters (
                      id, label, size, frequency, task_types, projects, machines,
                      fail_rate, noise, summary_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cluster["id"],
                        cluster.get("label") or "",
                        cluster["size"],
                        cluster["frequency"],
                        cluster["task_types"],
                        cluster["projects"],
                        cluster["machines"],
                        cluster["fail_rate"],
                        1 if cluster.get("noise") else 0,
                        json.dumps(cluster, ensure_ascii=False),
                        now,
                    ),
                )
                for gid in cluster.get("session_ids") or []:
                    self._conn.execute(
                        "INSERT INTO cluster_members (cluster_id, global_id) VALUES (?, ?)",
                        (cluster["id"], gid),
                    )

    def clusters(self, *, include_noise: bool = False) -> list[dict]:
        rows = self._conn.execute("SELECT summary_json, noise FROM clusters").fetchall()
        found = []
        for row in rows:
            if row["noise"] and not include_noise:
                continue
            found.append(json.loads(row["summary_json"]))
        found.sort(key=lambda item: (-(item["frequency"] * (item["fail_rate"] or 0)), item["id"]))
        return found

    def put_proposal(self, proposal: dict) -> None:
        self._conn.execute(
            """
            INSERT INTO proposals (
              id, cluster_id, kind, payload_json, conflicts_json, evidence_json, status, edited_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              kind = excluded.kind,
              payload_json = excluded.payload_json,
              conflicts_json = excluded.conflicts_json,
              evidence_json = excluded.evidence_json,
              updated_at = excluded.updated_at
            """,
            (
                proposal["id"],
                proposal["cluster_id"],
                proposal["kind"],
                json.dumps(proposal.get("payload") or {}, ensure_ascii=False),
                json.dumps(proposal.get("conflicts") or [], ensure_ascii=False),
                json.dumps(proposal.get("evidence") or {}, ensure_ascii=False),
                proposal.get("status") or "pending",
                json.dumps(proposal.get("edited") or {}, ensure_ascii=False) if proposal.get("edited") else None,
                utc_now(),
            ),
        )
        self._conn.commit()

    def proposals(self, status: str | None = None) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM proposals ORDER BY id").fetchall()
        found = []
        for row in rows:
            if status and row["status"] != status:
                continue
            edited = json.loads(row["edited_json"]) if row["edited_json"] else {}
            payload = json.loads(row["payload_json"])
            payload.update(edited)
            found.append(
                {
                    "id": row["id"],
                    "cluster_id": row["cluster_id"],
                    "kind": edited.get("kind") or row["kind"],
                    "status": row["status"],
                    "payload": payload,
                    "conflicts": json.loads(row["conflicts_json"]),
                    "evidence": json.loads(row["evidence_json"]),
                }
            )
        return found

    def set_proposal_status(self, proposal_id: str, status: str, edited: dict | None = None) -> None:
        if status not in {"pending", "approved", "rejected"}:
            raise ValueError(status)
        current = self._conn.execute("SELECT edited_json FROM proposals WHERE id = ?", (proposal_id,)).fetchone()
        if current is None:
            raise KeyError(proposal_id)
        previous = json.loads(current["edited_json"]) if current["edited_json"] else {}
        if edited:
            previous.update(edited)
        self._conn.execute(
            "UPDATE proposals SET status = ?, edited_json = ?, updated_at = ? WHERE id = ?",
            (status, json.dumps(previous, ensure_ascii=False), utc_now(), proposal_id),
        )
        self._conn.commit()

    def add_version(self, version_id: str, manifest: dict, note: str = "") -> None:
        self._conn.execute(
            """
            INSERT INTO versions (id, adopted_at, retired_at, note, manifest_json)
            VALUES (?, ?, NULL, ?, ?)
            """,
            (version_id, utc_now(), note, json.dumps(manifest, ensure_ascii=False)),
        )
        self._conn.commit()

    def versions(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM versions ORDER BY adopted_at").fetchall()
        found = []
        for row in rows:
            found.append(
                {
                    "id": row["id"],
                    "adopted_at": row["adopted_at"],
                    "retired_at": row["retired_at"],
                    "note": row["note"] or "",
                    "manifest": json.loads(row["manifest_json"]),
                }
            )
        return found

    def retire_after(self, version_id: str) -> None:
        rows = self.versions()
        seen = False
        now = utc_now()
        for row in rows:
            if row["id"] == version_id:
                seen = True
                self._conn.execute("UPDATE versions SET retired_at = NULL WHERE id = ?", (version_id,))
                continue
            if seen and not row["retired_at"]:
                self._conn.execute("UPDATE versions SET retired_at = ? WHERE id = ?", (now, row["id"]))
        self._conn.commit()
