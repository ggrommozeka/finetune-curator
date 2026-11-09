"""Import LogMiner JSONL and OpenCode SQLite histories from several machines."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from scripts.patterns.store import PatternStore
from scripts.rules.sessions import normalize_messages, session_id


def content_hash(messages: list) -> str:
    blob = json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()


def global_id(content_hash_value: str) -> str:
    return "s_" + content_hash_value[:16]


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchall()
    if not rows:
        return set()
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def detect_opencode_schema(conn: sqlite3.Connection) -> str:
    """Return a schema label or raise if the file is not an OpenCode DB we know."""
    message_cols = _table_columns(conn, "message")
    part_cols = _table_columns(conn, "part")
    session_cols = _table_columns(conn, "session")
    if not session_cols and not message_cols:
        tables = [
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
        ]
        raise ValueError("Unrecognized OpenCode database. Tables: " + ", ".join(tables))
    if part_cols and "data" in part_cols and "data" in message_cols:
        return "opencode-v2"
    if "role" in message_cols and "content" in message_cols:
        return "opencode-v1"
    if "messages" in session_cols:
        return "opencode-blob"
    tables = [
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
    ]
    raise ValueError("Unrecognized OpenCode schema. Tables: " + ", ".join(tables))


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _rows_from_v2(conn: sqlite3.Connection) -> list[dict]:
    session_cols = _table_columns(conn, "session")
    title_col = "title" if "title" in session_cols else None
    created_col = next((name for name in ("time_created", "created_at", "time_updated") if name in session_cols), None)
    project_col = next((name for name in ("project_id", "directory", "slug") if name in session_cols), None)
    sessions = conn.execute("SELECT * FROM session").fetchall()
    records = []
    for session in sessions:
        keys = session.keys()
        sid = session["id"]
        parts = conn.execute(
            "SELECT data FROM part WHERE session_id = ? ORDER BY time_created, id" if "time_created" in _table_columns(conn, "part") else "SELECT data FROM part WHERE session_id = ? ORDER BY id",
            (sid,),
        ).fetchall()
        messages = []
        for part in parts:
            try:
                payload = json.loads(part["data"])
            except (TypeError, json.JSONDecodeError):
                continue
            role = payload.get("role") or payload.get("type") or "assistant"
            content = payload.get("text") or payload.get("content") or payload.get("output") or ""
            if content:
                messages.append({"role": str(role), "content": _text(content)})
        records.append(
            {
                "id": str(sid),
                "title": session[title_col] if title_col else "",
                "project": str(session[project_col]) if project_col else "",
                "created_at": str(session[created_col]) if created_col else "",
                "messages": messages,
            }
        )
    return records


def _rows_from_v1(conn: sqlite3.Connection) -> list[dict]:
    message_cols = _table_columns(conn, "message")
    session_key = "session_id" if "session_id" in message_cols else "sessionID"
    order = "id" if "id" in message_cols else session_key
    buckets: dict[str, list] = {}
    for row in conn.execute(f"SELECT * FROM message ORDER BY {order}"):
        sid = str(row[session_key])
        buckets.setdefault(sid, []).append(
            {"role": row["role"] or "user", "content": _text(row["content"])}
        )
    return [{"id": sid, "title": "", "project": "", "created_at": "", "messages": msgs} for sid, msgs in buckets.items()]


def _rows_from_blob(conn: sqlite3.Connection) -> list[dict]:
    records = []
    for row in conn.execute("SELECT * FROM session"):
        raw = row["messages"]
        try:
            messages = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            messages = []
        records.append(
            {
                "id": str(row["id"]),
                "title": row["title"] if "title" in row.keys() else "",
                "project": "",
                "created_at": "",
                "messages": messages or [],
            }
        )
    return records


def read_opencode_db(path: Path) -> tuple[str, list[dict]]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        schema = detect_opencode_schema(conn)
        if schema == "opencode-v2":
            return schema, _rows_from_v2(conn)
        if schema == "opencode-v1":
            return schema, _rows_from_v1(conn)
        return schema, _rows_from_blob(conn)
    finally:
        conn.close()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _record_from_export(row: dict) -> dict:
    messages = normalize_messages(row)
    return {
        "id": session_id(row),
        "title": row.get("title") or "",
        "project": row.get("project") or row.get("cwd") or "",
        "created_at": str(row.get("created_at") or row.get("time_created") or ""),
        "score": row.get("score"),
        "label": row.get("label") or "",
        "messages": messages,
        "message_count": len(messages),
    }


def ingest_records(
    store: PatternStore,
    records: list[dict],
    *,
    source_label: str,
    kind: str,
    schema_version: str,
) -> dict:
    added = attached = skipped = 0
    for record in records:
        origin_id = str(record.get("id") or "")
        if not origin_id:
            continue
        if store.seen_origin(source_label, origin_id):
            skipped += 1
            continue
        messages = record.get("messages") or []
        digest = content_hash(messages)
        gid = global_id(digest)
        existed = store.find_by_hash(digest) is not None
        store.add_session(
            global_id=gid,
            content_hash=digest,
            origin_id=origin_id,
            source_label=source_label,
            kind=kind,
            schema_version=schema_version,
            record={**record, "global_id": gid, "message_count": len(messages)},
        )
        if existed:
            attached += 1
        else:
            added += 1
    return {"added": added, "attached": attached, "skipped": skipped, "schema": schema_version}


def import_path(store: PatternStore, path: Path, *, source_label: str) -> dict:
    if path.suffix == ".jsonl":
        records = [_record_from_export(row) for row in read_jsonl(path)]
        return ingest_records(store, records, source_label=source_label, kind="jsonl", schema_version="logminer")
    schema, records = read_opencode_db(path)
    normalized = []
    for record in records:
        messages = normalize_messages(record)
        normalized.append({**record, "messages": messages, "message_count": len(messages)})
    return ingest_records(store, normalized, source_label=source_label, kind="opencode-db", schema_version=schema)


def import_gitlab_traces(store: PatternStore, gitlab_root: Path) -> dict:
    """Turn linked GitLab review traces into sessions tagged source=gitlab."""
    db_path = gitlab_root / "store.db"
    if not db_path.exists():
        return {"added": 0, "attached": 0, "skipped": 0, "schema": "gitlab-missing"}
    from scripts.gitlab.store import GitlabStore

    gitlab = GitlabStore(gitlab_root)
    try:
        traces = [trace for trace in gitlab.traces_for() if trace.get("status") == "linked"]
        polish = [
            item["record"]
            for item in gitlab.examples_for()
            if (item.get("record") or {}).get("source") == "gitlab_polish"
        ]
    finally:
        gitlab.close()
    records = []
    for record in polish:
        gitlab_meta = record.get("gitlab") or {}
        messages = record.get("messages") or []
        records.append(
            {
                "id": record.get("id"),
                "title": (gitlab_meta.get("commit_message") or "team-lead polish")[:80],
                "project": gitlab_meta.get("file") or "gitlab",
                "created_at": gitlab_meta.get("commit_created_at") or "",
                "messages": messages,
                "source_kind": "gitlab_polish",
            }
        )
    for trace in traces:
        anchor = trace.get("anchor") or {}
        body = "\n".join(
            part
            for part in (
                f"Reviewer ({trace.get('author_role') or 'commenter'}): {trace.get('body') or ''}",
                f"File: {anchor.get('file') or ''}",
                f"Before:\n{trace.get('before') or ''}",
                f"After:\n{trace.get('after') or ''}",
            )
            if part
        )
        records.append(
            {
                "id": f"gitlab-{trace.get('note_id')}",
                "title": (trace.get("body") or "")[:80],
                "project": anchor.get("file") or "gitlab",
                "created_at": trace.get("created_at") or "",
                "messages": [
                    {"role": "user", "content": body},
                    {"role": "assistant", "content": trace.get("after") or ""},
                ],
                "source_kind": "gitlab_review",
            }
        )
    return ingest_records(store, records, source_label="gitlab", kind="gitlab", schema_version="gitlab-traces")
