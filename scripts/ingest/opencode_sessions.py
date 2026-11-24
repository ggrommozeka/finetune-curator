"""Select, render, and pack OpenCode sessions for rule extraction."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

DEFAULT_CONTEXT_BUDGET = 170_000
DEFAULT_CHARS_PER_TOKEN = 3.0
DEFAULT_OVERLAP_TOKENS = 400
DEFAULT_MAX_SCORE = 0.6
DEFAULT_SOURCES = ("discarded", "low-score", "notes")


def estimate_tokens(text: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Conservative token estimate. Lower chars-per-token counts more tokens."""
    if not text:
        return 0
    cpt = chars_per_token if chars_per_token > 0 else DEFAULT_CHARS_PER_TOKEN
    return max(1, int(math.ceil(len(text) / cpt)))


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_review_decisions(path: Path) -> dict:
    """Load viewer decisions.json. Missing or unreadable files yield empty maps."""
    empty = {"labels": {}, "notes": {}}
    if not path.exists():
        return empty
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return empty
    if not isinstance(data, dict):
        return empty
    if data and all(isinstance(v, str) for v in data.values()):
        return {"labels": {str(k): v.lower() for k, v in data.items()}, "notes": {}}
    labels = {str(k): str(v).lower() for k, v in (data.get("labels") or {}).items()}
    notes = {str(k): str(v) for k, v in (data.get("notes") or {}).items()}
    return {"labels": labels, "notes": notes}


def source_model_info(record: dict) -> dict:
    """Provider and model that produced a session, from the record or its messages."""
    provider = ""
    name = ""
    model = record.get("model")
    if isinstance(model, dict):
        provider = str(model.get("providerID") or model.get("provider") or model.get("provider_id") or "")
        name = str(model.get("modelID") or model.get("model_id") or model.get("id") or model.get("name") or "")
    elif isinstance(model, str) and model.strip():
        text = model.strip()
        if "/" in text:
            provider, name = text.split("/", 1)
        else:
            name = text
    if not provider:
        provider = str(record.get("provider") or record.get("providerID") or record.get("provider_id") or "")
    if not name:
        name = str(record.get("model_id") or record.get("modelID") or "")
    blobs = []
    for key in ("info", "metadata"):
        blob = record.get(key)
        if isinstance(blob, dict):
            blobs.append(blob)
    for message in record.get("messages") or []:
        if isinstance(message, dict):
            blobs.append(message)
            if isinstance(message.get("info"), dict):
                blobs.append(message["info"])
    for blob in blobs:
        nested = blob.get("model")
        if isinstance(nested, dict):
            provider = provider or str(nested.get("providerID") or nested.get("provider") or "")
            name = name or str(nested.get("modelID") or nested.get("id") or nested.get("name") or "")
        elif isinstance(nested, str) and "/" in nested and not (provider and name):
            provider, name = nested.split("/", 1)
        elif isinstance(nested, str) and not name:
            name = nested
        provider = provider or str(blob.get("providerID") or blob.get("provider") or "")
        name = name or str(blob.get("modelID") or blob.get("model_id") or "")
    label = "/".join(part for part in (provider.strip(), name.strip()) if part)
    return {"provider": provider.strip(), "model": name.strip(), "label": label}


def session_id(rec: dict) -> str:
    return str(
        rec.get("id")
        or rec.get("session_id")
        or f"idx-{abs(hash(json.dumps(rec, sort_keys=True, default=str))) % 10**8}"
    )


def normalize_messages(rec: dict) -> list[dict]:
    msgs = rec.get("messages") or []
    if not msgs:
        return []
    if isinstance(msgs[0], str):
        out = []
        for raw in msgs:
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                out.append({"role": "unknown", "content": raw})
        return out
    return msgs


def first_user_preview(msgs: list[dict], n: int = 90) -> str:
    for message in msgs:
        if message.get("role") == "user":
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            content = re.sub(r"\s+", " ", content).strip()
            return (content[:n] + "…") if len(content) > n else content
    return "—"


def message_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("text") is not None:
                parts.append(str(part.get("text")))
            else:
                parts.append(json.dumps(part, ensure_ascii=False))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False)


def _score(rec: dict) -> float | None:
    raw = rec.get("score")
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def select_sessions(
    rows: list[dict],
    decisions: dict | None,
    *,
    sources: tuple[str, ...] | list[str] = DEFAULT_SOURCES,
    max_score: float = DEFAULT_MAX_SCORE,
    max_sessions: int = 0,
) -> list[dict]:
    """Pick sessions that are likely to contain harness mistakes.

    Sources:
    - discarded: reviewer marked the session discard
    - low-score: score <= max_score, or LogMiner label bad/maybe
    - notes: reviewer left a note (including on kept sessions)
    - gitlab_review: exported GitLab MR review examples (source gitlab_review)
    """
    decisions = decisions or {}
    labels = {str(k): str(v).lower() for k, v in (decisions.get("labels") or {}).items()}
    notes = {str(k): str(v) for k, v in (decisions.get("notes") or {}).items()}
    wanted = {s.strip().lower() for s in sources if s and s.strip()}
    chosen: list[tuple[tuple, dict]] = []
    seen: set[str] = set()

    for rec in rows:
        sid = session_id(rec)
        if sid in seen:
            continue
        decision = labels.get(sid, "")
        note = (notes.get(sid) or "").strip()
        score = _score(rec)
        label = str(rec.get("label") or "").lower()
        take = False
        if "discarded" in wanted and decision == "discard":
            take = True
        if "low-score" in wanted and (
            (score is not None and score <= max_score) or label in {"bad", "maybe"}
        ):
            take = True
        if "notes" in wanted and note:
            take = True
        is_gitlab = rec.get("source") == "gitlab_review" or bool(rec.get("gitlab"))
        if "gitlab_review" in wanted and is_gitlab:
            take = True
        if not take:
            continue
        seen.add(sid)
        if decision == "discard":
            rank = 0
        elif note or (is_gitlab and "gitlab_review" in wanted):
            rank = 1
        else:
            rank = 2
        score_key = score if score is not None else 1.0
        chosen.append(((rank, score_key, len(chosen)), rec))

    chosen.sort(key=lambda item: item[0])
    records = [rec for _key, rec in chosen]
    if max_sessions and max_sessions > 0:
        records = records[:max_sessions]
    return records


def render_session(rec: dict, decision: str | None, note: str | None) -> tuple[str, str, list[str]]:
    """Return (session_id, preamble, message_blocks)."""
    sid = session_id(rec)
    lines = [f"SESSION {sid}"]
    score = rec.get("score")
    if score is not None and score != "":
        lines.append(f"score: {score}")
    if rec.get("label"):
        lines.append(f"label: {rec.get('label')}")
    if decision:
        lines.append(f"reviewer_decision: {decision}")
    if note and str(note).strip():
        lines.append(f"reviewer_note: {str(note).strip()}")
    reasons = rec.get("reasons") or []
    if reasons:
        lines.append("logminer_reasons: " + "; ".join(str(item) for item in reasons))
    tools = rec.get("tools_used") or []
    if tools:
        lines.append("tools_used: " + ", ".join(str(item) for item in tools))
    gitlab = rec.get("gitlab") if isinstance(rec.get("gitlab"), dict) else {}
    if gitlab.get("comment"):
        lines.append("review_comment: " + str(gitlab["comment"]).strip())
    if gitlab.get("file"):
        lines.append("review_file: " + str(gitlab["file"]))
    if gitlab.get("link"):
        lines.append("review_link: " + str(gitlab["link"]))
    preamble = "\n".join(lines) + "\n"

    blocks: list[str] = []
    for index, message in enumerate(normalize_messages(rec), 1):
        role = str(message.get("role") or "?").lower()
        text = message_text(message.get("content")).strip()
        if role == "system" and len(text) > 3000:
            text = text[:3000] + "\n… [system prompt truncated]"
        if text:
            blocks.append(f"[{index} {role}]\n{text}")
        if role == "assistant" and message.get("tool_calls"):
            dumped = json.dumps(message["tool_calls"], ensure_ascii=False)
            blocks.append(f"[{index} tool_calls]\n{dumped}")
    if not blocks:
        blocks = ["(no messages)"]
    return sid, preamble, blocks


def _max_chars(tokens: int, chars_per_token: float) -> int:
    if tokens <= 0:
        return 1
    return max(1, int(tokens * chars_per_token) - 1)


def _split_piece(text: str, max_chars: int) -> list[str]:
    step = max(1, max_chars)
    if len(text) <= step:
        return [text]
    return [text[i : i + step] for i in range(0, len(text), step)]


def _tail(pieces: list[str], overlap_tokens: int, chars_per_token: float) -> str:
    if overlap_tokens <= 0 or not pieces:
        return ""
    blob = "\n".join(pieces)
    chars = max(1, int(overlap_tokens * chars_per_token))
    return "(overlap from previous chunk)\n" + blob[-chars:]


def chunk_blocks(
    session_id_value: str,
    preamble: str,
    blocks: list[str],
    *,
    budget_tokens: int,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[str]:
    """Split one session into chunks that fit the input budget, on message boundaries."""
    if not preamble.endswith("\n"):
        preamble += "\n"
    part_line = "part: 99/99\n"
    overhead = estimate_tokens(preamble + part_line, chars_per_token)
    if overhead >= budget_tokens:
        piece_budget = max(1, budget_tokens // 2)
    else:
        piece_budget = max(1, budget_tokens - overhead)
    piece_chars = _max_chars(piece_budget, chars_per_token)

    pieces: list[str] = []
    for block in blocks:
        block = block.strip("\n")
        if not block:
            continue
        if estimate_tokens(block + "\n", chars_per_token) <= piece_budget:
            pieces.append(block)
        else:
            pieces.extend(_split_piece(block, piece_chars))
    if not pieces:
        pieces = ["(empty session)"]

    groups: list[list[str]] = []
    current: list[str] = []
    current_tokens = overhead

    def _flush() -> None:
        nonlocal current, current_tokens
        if current:
            groups.append(current)
        current = []
        current_tokens = overhead

    for piece in pieces:
        piece_tokens = estimate_tokens(piece + "\n", chars_per_token)
        if current and current_tokens + piece_tokens > budget_tokens:
            previous = current
            _flush()
            if overlap_tokens > 0:
                tail = _tail(previous, overlap_tokens, chars_per_token)
                tail_tokens = estimate_tokens(tail + "\n", chars_per_token) if tail else 0
                if tail and overhead + tail_tokens + piece_tokens <= budget_tokens:
                    current = [tail]
                    current_tokens = overhead + tail_tokens
        current.append(piece)
        current_tokens += piece_tokens
    if current:
        groups.append(current)

    total = len(groups)
    rendered: list[str] = []
    for index, group in enumerate(groups, 1):
        body = "\n".join(group)
        rendered.append(f"{preamble}part: {index}/{total}\n{body}")
    # Keep the id on every chunk even if the caller preamble omitted it.
    for index, chunk in enumerate(rendered):
        if f"SESSION {session_id_value}" not in chunk:
            rendered[index] = f"SESSION {session_id_value}\n" + chunk
    return rendered


def pack_batches(
    chunks: list[str],
    budget_tokens: int,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> list[list[str]]:
    """Greedy pack of chunks. A chunk larger than the budget stays in its own batch."""
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    separator = estimate_tokens("\n\n", chars_per_token)
    for chunk in chunks:
        tokens = estimate_tokens(chunk, chars_per_token)
        extra = tokens if not current else tokens + separator
        if current and current_tokens + extra > budget_tokens:
            batches.append(current)
            current = [chunk]
            current_tokens = tokens
        else:
            current.append(chunk)
            current_tokens += extra
    if current:
        batches.append(current)
    return batches


def build_transcripts(
    rows: list[dict],
    decisions: dict | None,
    *,
    sources: tuple[str, ...] | list[str] = DEFAULT_SOURCES,
    max_score: float = DEFAULT_MAX_SCORE,
    max_sessions: int = 0,
    context_budget: int = DEFAULT_CONTEXT_BUDGET,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    batch: bool = True,
) -> tuple[list[str], list[str]]:
    """Return (transcripts, source_session_ids) ready to send to the model."""
    decisions = decisions or {"labels": {}, "notes": {}}
    labels = decisions.get("labels") or {}
    notes = decisions.get("notes") or {}
    selected = select_sessions(
        rows,
        decisions,
        sources=sources,
        max_score=max_score,
        max_sessions=max_sessions,
    )
    source_ids = [session_id(rec) for rec in selected]
    chunks: list[str] = []
    for rec in selected:
        sid = session_id(rec)
        _sid, preamble, blocks = render_session(rec, labels.get(sid), notes.get(sid))
        chunks.extend(
            chunk_blocks(
                sid,
                preamble,
                blocks,
                budget_tokens=context_budget,
                chars_per_token=chars_per_token,
                overlap_tokens=overlap_tokens,
            )
        )
    if batch:
        groups = pack_batches(chunks, context_budget, chars_per_token)
    else:
        groups = [[chunk] for chunk in chunks]
    transcripts = ["\n\n".join(group) for group in groups]
    return transcripts, source_ids
