"""Import LogMiner JSONL and OpenCode SQLite histories from several machines."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from scripts.patterns.store import PatternStore
from scripts.rules.sessions import normalize_messages, session_id, source_model_info


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


def _json_obj(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _tool_call(payload: dict) -> dict | None:
    kind = str(payload.get("type") or "").lower().replace("_", "-")
    name = payload.get("tool") or payload.get("name") or ""
    function = payload.get("function") if isinstance(payload.get("function"), dict) else {}
    name = name or function.get("name") or ""
    if kind not in {"tool", "tool-call", "tool-use"} and not name:
        return None
    state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
    arguments = payload.get("input")
    if arguments is None:
        arguments = payload.get("arguments") or function.get("arguments") or state.get("input") or state.get("args") or {}
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    call_id = str(payload.get("callID") or payload.get("call_id") or payload.get("id") or name or "call")
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": str(name or "tool"), "arguments": arguments},
    }


def _tool_output(payload: dict) -> str:
    state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
    output = payload.get("output")
    if output is None:
        output = state.get("output") if state.get("output") is not None else state.get("result")
    if output is None:
        output = payload.get("content") or ""
    return _text(output)


def _turns_from_parts(parts: list[dict], default_role: str = "assistant") -> list[dict]:
    """Rebuild chat turns, including tool calls and tool results, from typed parts."""
    messages: list[dict] = []
    current: dict | None = None

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        if current.get("content") or current.get("tool_calls"):
            messages.append(current)
        current = None

    for payload in parts:
        kind = str(payload.get("type") or "").lower().replace("_", "-")
        role = str(payload.get("role") or default_role or "assistant")
        if kind in {"tool-result", "tool-output"} or role == "tool":
            flush()
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(payload.get("callID") or payload.get("call_id") or payload.get("tool_call_id") or ""),
                    "content": _tool_output(payload),
                }
            )
            continue
        call = _tool_call(payload)
        if call and kind in {"tool", "tool-call", "tool-use"}:
            if current is None or current.get("role") != "assistant":
                flush()
                current = {"role": "assistant", "content": ""}
            current.setdefault("tool_calls", []).append(call)
            if kind == "tool" and (payload.get("output") is not None or payload.get("state")):
                output = _tool_output(payload)
                flush()
                if output:
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
            continue
        text = payload.get("text") if payload.get("text") is not None else payload.get("content") or ""
        if current is not None and current.get("role") != role:
            flush()
        if current is None:
            current = {"role": role, "content": ""}
        if text:
            piece = _text(text)
            current["content"] = piece if not current.get("content") else current["content"] + "\n" + piece
    flush()
    return messages


def _part_payloads(conn: sqlite3.Connection, session_id_value: str, message_id: str | None = None) -> list[dict]:
    part_cols = _table_columns(conn, "part")
    if not part_cols or "data" not in part_cols:
        return []
    order = "time_created, id" if "time_created" in part_cols and "id" in part_cols else "id" if "id" in part_cols else "rowid"
    if message_id is not None and "message_id" in part_cols:
        query = f"SELECT data FROM part WHERE message_id = ? ORDER BY {order}"
        rows = conn.execute(query, (message_id,)).fetchall()
    elif "session_id" in part_cols:
        query = f"SELECT data FROM part WHERE session_id = ? ORDER BY {order}"
        rows = conn.execute(query, (session_id_value,)).fetchall()
    else:
        return []
    payloads = []
    for row in rows:
        payload = _json_obj(row["data"])
        if payload:
            payloads.append(payload)
    return payloads


def _attach_model(record: dict, session=None) -> dict:
    if session is not None and "model" in session.keys() and session["model"]:
        record["model"] = _json_obj(session["model"]) or session["model"]
    info = source_model_info(record)
    if info["label"]:
        record["source_model"] = info["label"]
        record["source_provider"] = info["provider"]
        if not isinstance(record.get("model"), (dict, str)):
            record["model"] = {"providerID": info["provider"], "modelID": info["model"]}
    return record


def _rows_from_v2(conn: sqlite3.Connection) -> list[dict]:
    session_cols = _table_columns(conn, "session")
    message_cols = _table_columns(conn, "message")
    title_col = "title" if "title" in session_cols else None
    created_col = next((name for name in ("time_created", "created_at", "time_updated") if name in session_cols), None)
    project_col = next((name for name in ("project_id", "directory", "slug") if name in session_cols), None)
    sessions = conn.execute("SELECT * FROM session").fetchall()
    records = []
    for session in sessions:
        sid = session["id"]
        messages: list[dict] = []
        if message_cols and "session_id" in message_cols:
            order = "time_created, id" if "time_created" in message_cols else "id"
            rows = conn.execute(
                f"SELECT * FROM message WHERE session_id = ? ORDER BY {order}",
                (sid,),
            ).fetchall()
            for row in rows:
                data = _json_obj(row["data"]) if "data" in row.keys() else {}
                role = str(data.get("role") or (row["role"] if "role" in row.keys() else "") or "assistant")
                payloads = _part_payloads(conn, str(sid), str(row["id"]) if "id" in row.keys() else None)
                if not payloads and "message_id" not in _table_columns(conn, "part"):
                    payloads = []
                turns = _turns_from_parts(payloads, default_role=role)
                if not turns and (data.get("content") or data.get("text")):
                    turns = [{"role": role, "content": _text(data.get("content") or data.get("text"))}]
                for turn in turns:
                    if turn.get("role") != "tool" and not any(part.get("role") for part in payloads):
                        turn["role"] = role
                    model = data.get("model") or (row["model"] if "model" in row.keys() else None)
                    if model and "model" not in turn:
                        turn["model"] = model
                messages.extend(turns)
        if not messages:
            messages = _turns_from_parts(_part_payloads(conn, str(sid)))
        record = {
            "id": str(sid),
            "title": session[title_col] if title_col else "",
            "project": str(session[project_col]) if project_col else "",
            "created_at": str(session[created_col]) if created_col else "",
            "messages": messages,
        }
        records.append(_attach_model(record, session))
    return records


def _rows_from_v1(conn: sqlite3.Connection) -> list[dict]:
    message_cols = _table_columns(conn, "message")
    session_key = "session_id" if "session_id" in message_cols else "sessionID"
    order = "id" if "id" in message_cols else session_key
    buckets: dict[str, list] = {}
    for row in conn.execute(f"SELECT * FROM message ORDER BY {order}"):
        sid = str(row[session_key])
        message = {"role": row["role"] or "user", "content": _text(row["content"])}
        if "model" in row.keys() and row["model"]:
            message["model"] = row["model"]
        calls = row["tool_calls"] if "tool_calls" in row.keys() else None
        if calls:
            if isinstance(calls, str):
                try:
                    message["tool_calls"] = json.loads(calls)
                except json.JSONDecodeError:
                    message["tool_calls"] = []
            else:
                message["tool_calls"] = calls
        buckets.setdefault(sid, []).append(message)
    records = []
    for sid, msgs in buckets.items():
        records.append(_attach_model({"id": sid, "title": "", "project": "", "created_at": "", "messages": msgs}))
    return records


def _rows_from_blob(conn: sqlite3.Connection) -> list[dict]:
    records = []
    for row in conn.execute("SELECT * FROM session"):
        raw = row["messages"]
        try:
            messages = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            messages = []
        record = {
            "id": str(row["id"]),
            "title": row["title"] if "title" in row.keys() else "",
            "project": "",
            "created_at": "",
            "messages": messages or [],
        }
        records.append(_attach_model(record, row))
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
        "machine": row.get("machine") or "",
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
        if not record.get("machine"):
            record["machine"] = source_label
        record = _attach_model(record)
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
                "reviewer": trace.get("reviewer") or trace.get("author_username") or "",
            }
        )
    return ingest_records(store, records, source_label="gitlab", kind="gitlab", schema_version="gitlab-traces")
