"""Per-session annotation with the local OpenAI-compatible model. Results are cached."""

from __future__ import annotations

from scripts.patterns.store import PatternStore
from scripts.rules.parsing import ModelOutputError, extract_json
from scripts.rules.sessions import message_text

ANNOTATION_SYSTEM = """You annotate one coding-agent session for later pattern mining.
Return ONLY JSON with these fields:
{"task_type":"bugfix|feature|refactor|review|ops|other","domain":"short","project":"short","languages":["..."],"tools":["..."],"outcome":"success|partial|fail","failure_step":"where it went wrong, or empty","error_type":"wrong-tool|loop|skipped-verification|misunderstood-task|environment|none|other","user_correction":"how the user corrected the agent, or empty","what_worked":"what worked if the session succeeded, or empty"}
Use short phrases. Do not copy secrets.
"""


def _transcript(record: dict, limit: int = 12000) -> str:
    parts = []
    for message in record.get("messages") or []:
        role = message.get("role") or "?"
        text = message_text(message.get("content"))
        parts.append(f"[{role}]\n{text}")
    blob = "\n".join(parts)
    return blob[:limit]


def _coerce(payload: dict) -> dict:
    def text(key: str) -> str:
        value = payload.get(key) or ""
        return str(value).strip()

    def listing(key: str) -> list[str]:
        value = payload.get(key) or []
        if isinstance(value, str):
            value = [value]
        return [str(item).strip() for item in value if str(item).strip()]

    outcome = text("outcome").lower() or "partial"
    if outcome not in {"success", "partial", "fail"}:
        outcome = "partial"
    return {
        "task_type": text("task_type") or "other",
        "domain": text("domain"),
        "project": text("project"),
        "languages": listing("languages"),
        "tools": listing("tools"),
        "outcome": outcome,
        "failure_step": text("failure_step"),
        "error_type": text("error_type") or "none",
        "user_correction": text("user_correction"),
        "what_worked": text("what_worked"),
    }


def annotation_text(annotation: dict) -> str:
    return " ".join(
        str(annotation.get(key) or "")
        for key in ("task_type", "domain", "project", "error_type", "failure_step", "user_correction", "what_worked")
    )


def annotate_session(record: dict, complete_fn) -> dict:
    transcript = _transcript(record)
    messages = [
        {"role": "system", "content": ANNOTATION_SYSTEM},
        {"role": "user", "content": "SESSION " + record["global_id"] + "\n" + transcript + "\n/no_think"},
    ]
    last_error: Exception | None = None
    for _attempt in range(3):
        raw = complete_fn(messages)
        try:
            payload = extract_json(raw)
            if not isinstance(payload, dict):
                raise ModelOutputError("annotation must be an object")
            return _coerce(payload)
        except ModelOutputError as exc:
            last_error = exc
            messages = messages + [
                {"role": "assistant", "content": (raw or "")[:2000]},
                {"role": "user", "content": "Return only the JSON object.\n/no_think"},
            ]
    raise ModelOutputError(str(last_error))


def annotate_pending(store: PatternStore, complete_fn, *, model: str = "local", refresh: bool = False) -> dict:
    done = failed = skipped = 0
    for record in store.sessions():
        if not refresh and store.annotation(record["global_id"]):
            skipped += 1
            continue
        try:
            annotation = annotate_session(record, complete_fn)
        except ModelOutputError:
            failed += 1
            continue
        if not annotation.get("project"):
            annotation["project"] = record.get("project") or ""
        store.put_annotation(record["global_id"], annotation, model)
        done += 1
    return {"annotated": done, "skipped": skipped, "failed": failed}

