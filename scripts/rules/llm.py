"""Rule extraction on top of the shared chat client.

Chat completion itself lives in ``scripts.llm.client``. This module keeps the
rule prompt, JSON retry, and the names older callers import.
"""

from __future__ import annotations

import re

from scripts.llm.client import build_chat_kwargs, chat_complete, thinking_controls
from scripts.llm.parsing import ModelOutputError, extract_json
from scripts.proposals.merge import slug_category

__all__ = [
    "SYSTEM_PROMPT",
    "build_chat_kwargs",
    "chat_complete",
    "coerce_rules",
    "request_rules",
    "session_ids_in_transcript",
    "thinking_controls",
]

SYSTEM_PROMPT = """You extract durable rules that should be added to an AI coding agent's system prompt (AGENTS.md or the harness system prompt).

The sessions below were discarded, scored poorly, or annotated by a reviewer. They show mistakes: ignored corrections, loops, tool misuse, unsafe edits, missing verification, and similar failures.

Write rules that prevent those mistakes next time. Each rule must be:
- one concise imperative instruction the agent can follow
- general (not a one-off task step, and not a file path copied from the session)
- grounded in the sessions you were given

Do not copy secrets, tokens, keys, or long file contents. Evidence is a short excerpt (max 240 characters).

Respond with ONLY a JSON object, no markdown:
{"rules":[{"text":"...","category":"tool-use|looping|verification|editing|planning|communication|context|safety|other","rationale":"...","session_ids":["..."],"evidence":"..."}]}

Use session ids exactly as labeled SESSION <id>. If nothing in the batch is worth a harness rule, return {"rules":[]}.
Write the rule text in the language of the reviewer notes and user corrections when those are present; otherwise use English.
Prefer a handful of strong rules over a long list. At most 8 rules.
"""

SESSION_RE = re.compile(r"^SESSION\s+(\S+)", re.MULTILINE)
_RETRY_HINT = (
    "That was not valid JSON. Return ONLY a JSON object "
    '{"rules":[{"text":"...","category":"...","rationale":"...","session_ids":["..."],"evidence":"..."}]} '
    "with no markdown and no thinking."
)


def session_ids_in_transcript(text: str) -> list[str]:
    seen: list[str] = []
    for match in SESSION_RE.finditer(text or ""):
        sid = match.group(1)
        if sid not in seen:
            seen.append(sid)
    return seen


def coerce_rules(payload, allowed: list[str] | None = None) -> list[dict]:
    """Normalize a parsed model payload into rule dicts (no ids yet)."""
    if isinstance(payload, dict):
        raw = payload.get("rules", payload.get("items"))
        if raw is None and payload.get("text"):
            raw = [payload]
        if raw is None:
            raise ModelOutputError("JSON object has no 'rules' list")
    elif isinstance(payload, list):
        raw = payload
    else:
        raise ModelOutputError("JSON must be an object or a list")
    if not isinstance(raw, list):
        raise ModelOutputError("'rules' must be a list")

    allowed_set = set(allowed or [])
    rules: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("rule") or item.get("instruction") or ""
        text = re.sub(r"\s+", " ", str(text)).strip()
        if not text:
            continue
        rationale = str(item.get("rationale") or item.get("why") or "").strip()[:800]
        evidence = item.get("evidence") or item.get("excerpt") or item.get("quote") or ""
        evidence = re.sub(r"\s+", " ", str(evidence)).strip()[:500]
        sids = item.get("session_ids", item.get("session_id", item.get("sessions", [])))
        if isinstance(sids, str):
            sids = [sids]
        if not isinstance(sids, list):
            sids = []
        cleaned: list[str] = []
        for sid in sids:
            sid = str(sid).strip()
            if not sid:
                continue
            if allowed_set and sid not in allowed_set:
                continue
            if sid not in cleaned:
                cleaned.append(sid)
        if not cleaned and allowed:
            cleaned = list(allowed)
        rules.append(
            {
                "text": text,
                "category": slug_category(str(item.get("category") or "other")),
                "rationale": rationale,
                "session_ids": cleaned,
                "evidence": evidence,
            }
        )
    return rules


def request_rules(
    complete_fn,
    transcript: str,
    *,
    disable_thinking: bool = True,
    attempts: int = 3,
) -> list[dict]:
    """Call ``complete_fn(messages) -> str`` and parse rules, retrying on bad JSON."""
    suffix, _extra = thinking_controls(disable_thinking)
    user = "Extract harness rules from these sessions.\n\n" + transcript.strip() + suffix
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    allowed = session_ids_in_transcript(transcript)
    last_error: Exception | None = None
    tries = max(1, attempts)
    for attempt in range(1, tries + 1):
        raw = complete_fn(messages)
        try:
            return coerce_rules(extract_json(raw), allowed)
        except ModelOutputError as exc:
            last_error = exc
            if attempt == tries:
                break
            messages = messages + [
                {"role": "assistant", "content": (raw or "")[:4000]},
                {"role": "user", "content": _RETRY_HINT + suffix},
            ]
    raise ModelOutputError(
        f"model did not return valid rules JSON after {tries} attempts: {last_error}"
    )
